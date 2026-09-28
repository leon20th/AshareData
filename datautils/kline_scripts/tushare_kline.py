# -*- coding: utf-8 -*-
"""tushare_kline —— 日K tushare 版构建器（字段自维护：缺口自动判定、最小请求增量抓取）

设计原则
  1. 每个字段只对外暴露一个 `update_<字段>` 接口：该补哪一段、用哪个 tushare 接口，全在接口内部判断；
     调用方不需要知道增量/全量——重复跑幂等。
  2. 字段 ↔ tushare 接口：

     字段（13 列 schema）          接口            说明
     ---------------------------   -------------   -----------------------------------------------
     OHLCV + 成交额/前收/涨跌幅    daily           未复权；vol 手→股（×100）、amount 千元→元（×1000）
     换手率 turn                   daily_basic     turnover_rate（=成交量/流通股，与 v2 自算同口径）
     isST（历史）                  namechange      曾用名时间线 → 名称 ST 前缀；目标日行以名单现名覆盖
     复权因子（aux，读时复权用）    adj_factor      全市场按交易日；不做复权计算，读时另折算
     名单/上市退市日                stock_basic     L/D/P 三态
     tradestatus（停牌占位行）      ——              由本地交易日历 + daily 缺行派生（与 v2 同语义）
     交易日历/目标日                ——              沿用本地 a_open（baostock 维护），与全站一致
     m15（15 分钟线）              ——              新浪补缺口（近 ~64 交易日窗）+ baostock 老段；不复权原值

  3. 抓取策略（最小请求）：
     - daily：按票缺口贪心分批（多代码一次调用，6000 行/次上限）；只请求缺失区间（冷建多取库起点前一段）。
     - turn ：按缺失票日汇总 → 按交易日全市场各取一次、现场回填（不落缓存，无持久状态）。
     - adj  ：按交易日缺口全市场各取一次；已抓日期记 _aux/adj_fetch_log.json。
     - namechange：按公告年份分段续传，水位记 _aux/namechange_log.json。
     - m15 ：缺口分窗路由——近 ~64 交易日窗内缺口用新浪（快、可并发）；更早缺口/整段用 baostock（单线程慢扫、逐票续传）。
  4. 口径与 v2（quick_kline）逐位对齐：13 列 schema 与列序、未复权原值、每交易日一行、
     停牌占位（volume==0：价格 carry、amount=0、turn/pctChg 空、tradestatus=0）、
     覆盖 [max(库起点, 上市日), min(目标日, 退市日)]、首行前收基准（库起点前收盘，_aux/prev_close）。
     当日（目标日==今天）源未入库时不落占位行，留缺口下轮补（源已入库则当晚补齐停牌票）。
  5. 老数据不移植（用户定案）：fresh 建库，不读旧库做历史覆盖。

产物: daily_kline_ts/{sh.600000.csv} 13 列 + m15_kline_ts/{sh.600000.csv} 9 列（均为未复权原值）
辅助: daily_kline_ts/_aux/（meta_tickers / adj_factor+日志 / namechange+日志 / prev_close）
用法:
  python AshareData/datautils/kline_scripts/tushare_kline.py [--codes 600519,sz.000001] [--skip turn,isst]
  python AshareData/datautils/kline_scripts/tushare_kline.py --latest   # 只问库状态，不抓数据
  python AshareData/datautils/kline_scripts/tushare_kline.py --probe    # 连通性/权限自检（不写数据）
token: 环境变量 TUSHARE_TOKEN 优先，其次 AshareData/.keys/.tushare_token（单行纯文本；配置见 setup.sh）
参考耗时（全市场 2020 起冷建）：日线部分 ≈5.1k 次调用、约 15 分钟；m15 老段走 baostock 单线程（逐票慢扫，全市场需数天，可 --codes 分批续传）。
备注: m15 不用 tushare 分钟接口（stk_mins 需单独权限）；新浪/baostock 均为不复权原值，复权读出口用 _aux/adj_factor 折算。
"""
import argparse
import bisect
import json
import os
import queue
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

from AshareData.paths import ASHARE_ROOT, DAILY_KLINE_TS_DIR, M15_KLINE_TS_DIR
from AshareData.utils.exchanges_utils.a_open import get_target_trade_date, get_trade_date_list
from AshareData.utils.log_util import get_logger
from AshareData.utils.read_file_utils import read_last_lines

logger = get_logger('日Kts')

TOKEN_FILE = os.path.join(ASHARE_ROOT, '.keys', '.tushare_token')
TS_DIR = DAILY_KLINE_TS_DIR
AUX_DIR = os.path.join(TS_DIR, '_aux')
META_F = os.path.join(AUX_DIR, 'meta_tickers.parquet')
ADJ_F = os.path.join(AUX_DIR, 'adj_factor.parquet')
ADJ_LOG = os.path.join(AUX_DIR, 'adj_fetch_log.json')
NAME_F = os.path.join(AUX_DIR, 'namechange.parquet')
NAME_LOG = os.path.join(AUX_DIR, 'namechange_log.json')
PREV_F = os.path.join(AUX_DIR, 'prev_close.parquet')

M15_DIR = M15_KLINE_TS_DIR
M15_COLS = ['date', 'time', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
SINA_KLINE = 'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData'
SINA_MAX_BARS = 1023             # 新浪单次请求上限（≈64 个交易日的 15 分钟 bar）
SINA_WIN = 62                    # 新浪可回补窗口（交易日数）
BARS_PER_DAY = 16
BS_TIMEOUT = 120                 # baostock 单次查询看门狗（连接半开有卡死前科）

COLS = ['date', 'code', 'open', 'high', 'low', 'close', 'preclose',
        'volume', 'amount', 'turn', 'tradestatus', 'pctChg', 'isST']
FLOOR = '2020-01-02'            # 库起点（与全站消费链对齐）
PRE_FLOOR = '2019-10-01'        # 冷建时多取一段：供"首行前收基准"（库起点前末收盘）
BATCH_ROWS = 5500               # daily 单次调用行数上限（接口 6000，留余量）
BATCH_CODES = 50                # daily 单次调用代码数上限
FLUSH_N = 300                   # adj 抓取过程每 N 个交易日落一次盘（控内存）
ST_RE = re.compile(r'^(S\*?ST|\*?ST)')   # 名称 ST 前缀（含 SST / S*ST 老式标记）
WORKERS = 6                     # 并发抓取线程数（总频率由 _pace 统一限速）
PACE_GAP = 0.15                 # 全局最小请求间隔（秒）≈ 400 次/分，留出 500/分限频余量

_CAL = None
_PRO = None
_PACE_LK = threading.Lock()
_PACE_LAST = [0.0]
_BS_ON = False                  # baostock 登录态（卡死/断线后强制重登）
_TLS = threading.local()


# ==================== 通用工具 ====================

def _cal():
    """全部交易日（'YYYY-MM-DD' 升序，含已发布的前瞻日期）。"""
    global _CAL
    if _CAL is None:
        _CAL = sorted(f'{d[:4]}-{d[4:6]}-{d[6:8]}' for d in get_trade_date_list())
    return _CAL


def _iso(d):
    d = str(d).strip()
    return f'{d[:4]}-{d[4:6]}-{d[6:8]}' if len(d) == 8 and d.isdigit() else d[:10]


def _ymd(d):
    return str(d).replace('-', '')


def _d8(x):
    """日期值（'YYYYMMDD' / 'YYYY-MM-DD' / 带 .0 的 float / NaN）→ 'YYYY-MM-DD' 或 NaN。"""
    if x is None:
        return np.nan
    try:
        if pd.isna(x):
            return np.nan
    except (TypeError, ValueError):
        return np.nan
    s = str(x).strip()
    if s.endswith('.0'):
        s = s[:-2]
    s = s.replace('-', '')
    return _iso(s) if (len(s) == 8 and s.isdigit()) else np.nan


def _ts_code(code):
    a, b = code.split('.')
    return f'{b}.{a.upper()}'


def _local(ts_code):
    a, b = ts_code.split('.')
    return f'{b.lower()}.{a}'


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _dur(sec):
    sec = int(sec)
    if sec < 60:
        return f'{sec}s'
    if sec < 3600:
        return f'{sec // 60}m{sec % 60:02d}s'
    return f'{sec // 3600}h{sec % 3600 // 60:02d}m'


def _progress(i, n, t0, tag, extra='', every=500):
    """粗粒度进度：每轮迭代开头调用（i 从 1 起），首尾必打、每 every 只打一行。"""
    if i != 1 and i != n and i % every:
        return
    el = time.time() - t0
    tail = f' ETA {_dur(el / (i - 1) * (n - i + 1))}' if 1 < i < n else ''
    logger.info(f'{tag} {i}/{n}' + (f' {extra}' if extra else '') + f' 已用 {_dur(el)}{tail}')


def norm_codes(text):
    """'600519,sz.000001' → ['sh.600519', 'sz.000001']。"""
    out = []
    for tok in re.split(r'[,\s]+', text or ''):
        tok = tok.strip()
        if not tok:
            continue
        if '.' in tok:
            a, b = tok.lower().split('.')
            if a.isdigit():          # '600000.sh' / '600000.SH'
                a, b = b, a
            if a not in ('sh', 'sz', 'bj') or not b.isdigit():
                raise ValueError(f'无法识别代码: {tok}')
            code = f'{a}.{b}'
        elif tok.startswith('6'):
            code = f'sh.{tok}'
        elif tok.startswith(('0', '2', '3')):
            code = f'sz.{tok}'
        else:
            raise ValueError(f'无法识别代码: {tok}')
        out.append(code)
    return sorted(set(out))


# ==================== tushare 调用层（重试 / 频控 / 权限识别） ====================

def _pro():
    """tushare 客户端（懒初始化）：环境变量 TUSHARE_TOKEN 优先，其次 .keys/.tushare_token。"""
    global _PRO
    if _PRO is None:
        token = os.environ.get('TUSHARE_TOKEN', '').strip()
        if not token and os.path.exists(TOKEN_FILE):
            token = open(TOKEN_FILE, encoding='utf-8').read().strip()
        if not token:
            raise RuntimeError(f'缺少 tushare token：写入 {TOKEN_FILE}（单行纯文本）'
                               f'或设环境变量 TUSHARE_TOKEN（配置见 AshareData/setup.sh）')
        import tushare as ts
        _PRO = ts.pro_api(token)
    return _PRO


def _pace():
    """全局请求节奏（多线程共享）：保证跨接口总频率不超过 ~400 次/分（限频 500/分）。"""
    while True:
        with _PACE_LK:
            now = time.time()
            wait = _PACE_LAST[0] + PACE_GAP - now
            if wait <= 0:
                _PACE_LAST[0] = now
                return
        time.sleep(wait)


def _call(fn, what, retries=3):
    """tushare 调用重试：频控等待重试；积分/权限/token 类错误直接抛（重试无意义）。

    SDK 对服务端错误抛 Exception(msg)；HTTP 非 2xx 会被静默吞成空表（由缺口检查兜底重试）。
    """
    last = None
    for k in range(retries):
        try:
            _pace()
            return fn()
        except Exception as e:
            msg = str(e)
            if ('每分钟' in msg) or ('最多访问' in msg and '每天' not in msg):
                wait = 20 + 15 * k
                logger.warning(f'[tushare] {what} 触发频控，{wait}s 后重试: {msg[:70]}')
                time.sleep(wait)
                last = e
                continue
            if any(w in msg for w in ('积分', '权限', 'TOKEN', 'token', '无效', '每天最多', '总量', '过期')):
                raise RuntimeError(f'tushare {what} 不可用（积分/权限/token）: {msg[:200]}')
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f'tushare {what} 重试 {retries} 次仍失败: {last}')


def _ts_stock_basic(status):
    return _call(lambda: _pro().stock_basic(exchange='', list_status=status,
                                            fields='ts_code,name,list_date,delist_date'),
                 f'stock_basic({status})')


def _ts_daily(ts_codes, start, end):
    """多代码一次调用（行数上限由调用方贪心分批保证）。"""
    return _call(lambda: _pro().daily(ts_code=','.join(ts_codes), start_date=start, end_date=end), 'daily')


def _ts_daily_basic(date):
    return _call(lambda: _pro().daily_basic(trade_date=date, fields='ts_code,trade_date,turnover_rate'),
                 'daily_basic')


def _ts_adj(date):
    return _call(lambda: _pro().adj_factor(trade_date=date), 'adj_factor')


def _ts_namechange(start, end):
    """全市场按公告日段（不带 ts_code）——用于构建曾用名时间线。"""
    return _call(lambda: _pro().namechange(start_date=start, end_date=end), 'namechange')


# ==================== 本地读写 ====================

def read_ts(code):
    p = os.path.join(TS_DIR, f'{code}.csv')
    if not os.path.exists(p):
        return pd.DataFrame(columns=COLS)
    # float_precision='round_trip'：精确往返解析，保证"重算值 == 读到值"的幂等比较（同 v2）
    return pd.read_csv(p, dtype={'date': str}, float_precision='round_trip')


def write_ts(code, df):
    """落盘归一（volume==0 即停牌判定信号）：非成交行量额归 0、比率列留空；真成交行原样。"""
    df = df.reindex(columns=COLS).sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    nv = pd.to_numeric(df['volume'], errors='coerce')
    off = ~(nv > 0)
    if off.any():
        df.loc[off, 'volume'] = 0.0
        df.loc[off, 'amount'] = 0.0
        df.loc[off, 'turn'] = np.nan
        df.loc[off, 'pctChg'] = np.nan
    os.makedirs(TS_DIR, exist_ok=True)
    df.to_csv(os.path.join(TS_DIR, f'{code}.csv'), index=False)


# ==================== baostock 直连（m15 老段） ====================

def _bs_ensure_login():
    """baostock 登录（全局一次；卡死/断线后由 _bs_rows 置回 False 强制重登）。"""
    global _BS_ON
    if _BS_ON:
        return
    import baostock as bs
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f'baostock 登录失败: {lg.error_msg}')
    _BS_ON = True


def _bs_call(fn, timeout=BS_TIMEOUT):
    """守护线程里跑 baostock 调用：连接半开卡死时超时放弃，不阻塞整轮。返回 (ok, 结果或异常)。"""
    q = queue.Queue(maxsize=1)

    def run():
        try:
            q.put((True, fn()))
        except Exception as e:
            q.put((False, e))

    threading.Thread(target=run, daemon=True).start()
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return False, TimeoutError(f'baostock 调用超时>{timeout}s')


def _bs_rows(code, fields, start, end, frequency='d', adjustflag='3', retries=3):
    """baostock 拉一段 K 线（逐段串行；超时看门狗 + 断线重登录重试）。空段返回 []，失败抛异常。"""
    import baostock as bs   # noqa: F401  延迟导入：不用 m15 的环境不需要装 baostock
    global _BS_ON
    last = None
    for attempt in range(retries):
        try:
            _bs_ensure_login()
            ok, rs = _bs_call(lambda: bs.query_history_k_data_plus(
                code, fields, start_date=start, end_date=end,
                frequency=frequency, adjustflag=adjustflag))
            if not ok:
                raise rs
            if rs.error_code != '0':
                raise RuntimeError(rs.error_msg)

            def drain():
                out = []
                while rs.error_code == '0' and rs.next():
                    out.append(rs.get_row_data())
                return out

            ok, rows = _bs_call(drain)
            if not ok:
                raise rows
            return rows
        except Exception as e:
            last = e
            _BS_ON = False                      # 卡死/断线后下一轮强制重登录
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f'{code} {frequency} {start}~{end} 拉取失败: {last}')


def _fail(tag, code, err, failed, cap=3):
    """逐票失败（网络/服务等不可避免的缺失）：前 cap 条详列，之后只计数（末尾汇总）。"""
    failed.append(code)
    if len(failed) <= cap:
        logger.warning(f'{tag} {code} 失败: {err}')
    elif len(failed) == cap + 1:
        logger.warning(f'{tag} 失败已超 {cap} 条，后续只计数')


# ==================== 构建器 ====================

class TushareKline:
    """tushare 版日K构建器：字段自维护（缺口判定 + 最小请求 + 幂等）。

    统一签名 update_<字段>(codes=None)：codes 为 None 时取 meta 全市场沪深（未上市/退市票按各自边界处理）。
    """

    FIELDS = (('daily', 'OHLCV+前收/涨跌幅+停牌占位'), ('turn', '换手率'),
              ('adj', '复权因子（读时复权用）'), ('isst', 'isST'),
              ('m15', '15分钟线（新浪补全 / baostock 老段）'))

    def __init__(self, force=False):
        self.force = force                 # 强制重取（忽略缺口/已抓日志）
        self._meta_df = None
        self._prev = None                  # prev_close 文件缓存
        self._prev_mem = {}                # 冷建过程中新得的"库起点前收盘"
        self._lock = threading.Lock()      # 保护 _prev_mem 的并发读写
        self._tgt = None

    # ==================== 对外：字段接口 ====================

    def run(self, codes=None, skip=()):
        t0 = time.time()
        self.update_meta()
        codes = self._codes(codes)
        if not codes:
            logger.warning('票池为空 → 结束')
            return
        logger.info(f'== tushare 日K | {len(codes)} 只 | 目标日 {self._target_date()} ==')
        for i, (name, desc) in enumerate(self.FIELDS, 1):
            if name in skip:
                logger.info(f'== [{i}/{len(self.FIELDS)}] 跳过 {name} ==')
                continue
            t = time.time()
            getattr(self, f'update_{name}')(codes)
            logger.info(f'== [{i}/{len(self.FIELDS)}] {name}（{desc}）完成，用时 {_dur(time.time() - t)} ==')
        logger.info(f'== 完成，共 {_dur(time.time() - t0)} ==')

    def update_meta(self, codes=None):
        """名单 + 上市/退市日（stock_basic 三态 L/D/P）。"""
        frames = []
        for st in ('L', 'D', 'P'):
            d = _ts_stock_basic(st)
            if d is None or not len(d):
                continue
            d = d.copy()
            d['list_status'] = st
            frames.append(d)
        if not frames:
            raise RuntimeError('stock_basic 三态均无返回')
        df = pd.concat(frames, ignore_index=True)
        df = df[df['ts_code'].str.endswith(('.SH', '.SZ'))]
        df = df[~df['ts_code'].str[:3].isin(['900', '200'])]      # 去 B 股（900xxx.SH / 200xxx.SZ）
        df = df[['ts_code', 'name', 'list_date', 'delist_date', 'list_status']].sort_values('ts_code')
        os.makedirs(AUX_DIR, exist_ok=True)
        df.to_parquet(META_F)
        n = df['list_status'].value_counts().to_dict()
        logger.info(f'[meta] {len(df)} 只（' + '，'.join(f'{k}={v}' for k, v in sorted(n.items())) + '）')

    def update_daily(self, codes=None):
        """OHLCV+前收/涨跌幅（含停牌占位行）：按票缺口贪心分批，多代码一次调用。"""
        codes = self._codes(codes)
        tgt = self._target_date()
        cal = _cal()
        plan, cold = {}, {}
        for code in codes:
            df = read_ts(code)
            s, e = self._start_of(code), self._end_of(code)
            if s is None:
                if df.empty:
                    continue                     # 名单无上市日且本地无数据：不要求任何行
                s = df['date'].iloc[0]
            if e < s:
                continue                         # 未上市 / 退市早于库起点
            if self.force and len(df):
                fs = s
            else:
                have = set(df['date'])
                lo, hi = bisect.bisect_left(cal, s), bisect.bisect_right(cal, e)
                fs = next((d for d in cal[lo:hi] if d not in have), None)
                if fs is None:
                    continue
            cold[code] = df.empty
            if df.empty and s == FLOOR:
                fs = PRE_FLOOR                   # 冷建多取一段 → 首行前收基准
            plan[code] = (fs, e)
        if not plan:
            logger.info('[daily] 各票均无缺口 → 跳过')
            return
        n_cold = sum(cold.values())
        batches = self._batch_daily(plan, cal, tgt)
        logger.info(f'[daily] 需补 {len(plan)} 只' + (f'（其中冷建 {n_cold} 只）' if n_cold else '')
                    + f'，分 {len(batches)} 批（{WORKERS} 线程并发）')

        def work(batch):
            fs = min(plan[c][0] for c in batch)
            resp = _ts_daily([_ts_code(c) for c in batch], _ymd(fs), _ymd(tgt))
            pub_today = len(resp) > 0 and str(resp['trade_date'].max()) == _ymd(tgt)
            by = {ts: g for ts, g in resp.groupby('ts_code', observed=True)} if len(resp) else {}
            return sum(1 for c in batch if self._apply_daily(c, by.get(_ts_code(c)), pub_today))

        n, t0 = 0, time.time()
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for bi, got in enumerate(ex.map(work, batches), 1):
                n += got
                _progress(bi, len(batches), t0, '[daily]', f'已写 {n}')
        prev_known = self._load_prev()
        sent = [c for c in plan if cold.get(c) and self._start_of(c) == FLOOR
                and c not in self._prev_mem and c not in prev_known]
        self._prev_mem.update({c: np.nan for c in sent})   # 无基准哨兵：不每轮重查
        n_new = len(self._prev_mem) - len(sent)
        self._flush_prev()
        logger.info(f'[daily] 写 {n}/{len(plan)} 只'
                    + (f'（首行前收：新得 {n_new}、哨兵 {len(sent)}）' if (n_new or sent) else ''))

    def update_turn(self, codes=None):
        """换手率：daily_basic.turnover_rate，按缺失票日汇总 → 按交易日全市场抓取、现场回填（不落缓存）。"""
        codes = self._codes(codes)
        dates, hit = set(), set()
        for code in codes:
            df = read_ts(code)
            if df.empty:
                continue
            v = pd.to_numeric(df['volume'], errors='coerce')
            sel = (v > 0)
            if not self.force:
                sel &= pd.to_numeric(df['turn'], errors='coerce').isna()
            if sel.any():
                dates.update(df.loc[sel, 'date'])
                hit.add(code)
        if not dates:
            logger.info('[turn] 无需回填 → 跳过')
            return
        logger.info(f'[turn] {len(dates)} 个交易日 / {len(hit)} 只有缺口')
        got = self._fetch_basic(sorted(dates))          # 现场抓取（内存），抓完即回填
        cache = {c: g.set_index('date')['turnover_rate'].to_dict()
                 for c, g in got.groupby('code', sort=False, observed=True)} if len(got) else {}
        n, skip, t0 = 0, 0, time.time()
        for i, code in enumerate(sorted(hit), 1):
            _progress(i, len(hit), t0, '[turn]', f'已写 {n} 跳过 {skip}')
            df = read_ts(code)
            if df.empty:
                continue
            m = cache.get(code)
            if not m:
                skip += 1
                continue
            di = df['date'].str.replace('-', '').astype(int)
            rate = di.map(m)
            v = pd.to_numeric(df['volume'], errors='coerce')
            t = pd.to_numeric(df['turn'], errors='coerce').to_numpy()
            if self.force:
                upd = (v > 0).to_numpy() & rate.notna().to_numpy()
            else:
                upd = (v > 0).to_numpy() & np.isnan(t) & rate.notna().to_numpy()
            if not upd.any():
                skip += 1
                continue
            nt = np.where(upd, rate.to_numpy(), t)
            if np.array_equal(nt, t, equal_nan=True):
                skip += 1
                continue
            df['turn'] = nt
            write_ts(code, df)
            n += 1
        logger.info(f'[turn] 写 {n} 只' + (f'，无变化/缺源跳过 {skip} 只' if skip else ''))

    def update_adj(self, codes=None):
        """复权因子（读时复权用）：按交易日缺口全市场抓取；已抓日期记日志。"""
        cal, tgt = _cal(), self._target_date()
        done = set() if self.force else (set(json.load(open(ADJ_LOG))) if os.path.exists(ADJ_LOG) else set())
        todo = [d for d in cal if FLOOR <= d <= tgt and d not in done]
        if not todo:
            logger.info('[adj] 各交易日均已覆盖 → 跳过')
            return
        logger.info(f'[adj] 需抓 {len(todo)} 个交易日（全市场/日一次，{WORKERS} 线程）')
        os.makedirs(AUX_DIR, exist_ok=True)
        lock, buf, fetched = threading.Lock(), [], []
        t0, cnt, n_rows = time.time(), [0], [0]

        def work(dt):
            r = _ts_adj(_ymd(dt))
            rows = None
            if r is not None and len(r):
                rows = pd.DataFrame({
                    'code': [_local(x) for x in r['ts_code']],
                    'date': pd.to_numeric(r['trade_date'], errors='coerce').astype('int32'),
                    'adj_factor': pd.to_numeric(r['adj_factor'], errors='coerce')})
            with lock:
                cnt[0] += 1
                _progress(cnt[0], len(todo), t0, '[adj]', f'新增 {n_rows[0]} 行', every=100)
                if rows is None:
                    logger.warning(f'[adj] {dt} 无返回（留待下轮重试）')
                    return
                buf.append(rows)
                fetched.append(dt)
                n_rows[0] += len(rows)
                if len(buf) >= FLUSH_N:
                    self._flush_adj(buf)
                    buf.clear()

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(work, todo))
        if buf:
            self._flush_adj(buf)
        done |= set(fetched)
        json.dump(sorted(done), open(ADJ_LOG, 'w'))
        logger.info(f'[adj] 新增 {n_rows[0]} 行（覆盖 {len(done)} 个交易日）')

    def update_isst(self, codes=None):
        """isST：namechange 曾用名时间线映射（历史），目标日行以名单现名覆盖；仅变化时重写。"""
        codes = self._codes(codes)
        self._fetch_namechange()
        tgt = self._target_date()
        ev = pd.read_parquet(NAME_F) if os.path.exists(NAME_F) else pd.DataFrame(
            columns=['code', 'name', 'start_date', 'end_date', 'ann_date', 'change_reason'])
        if len(ev):
            ev = ev.copy()
            ev['flag'] = ev['name'].map(lambda x: int(bool(ST_RE.match(str(x).strip().upper()))))
        groups = ({c: g.sort_values('start_date') for c, g in ev.groupby('code', sort=False, observed=True)}
                  if len(ev) else {})
        n, t0 = 0, time.time()
        for i, code in enumerate(codes, 1):
            _progress(i, len(codes), t0, '[isst]', f'已写 {n}')
            df = read_ts(code)
            if df.empty:
                continue
            g = groups.get(code)
            if g is not None and len(g):
                starts = np.array(g['start_date'].astype(str).tolist())
                pos = np.searchsorted(starts, np.array(df['date'].tolist()), side='right') - 1
                isst = np.where(pos >= 0, g['flag'].to_numpy()[np.clip(pos, 0, None)], 0).astype(int)
            else:
                isst = np.zeros(len(df), dtype=int)
            r = self._row(code)
            cur = str(r['name']) if r is not None else ''
            hit = (df['date'] == tgt).to_numpy()
            if hit.any() and cur and cur != 'nan':
                v = int(bool(ST_RE.match(cur.strip().upper())))
                if isst[hit][0] != v:
                    logger.info(f'[isst] 名单现名与时间线不一致（以现名覆盖）: {code} → {cur}')
                    isst = np.where(hit, v, isst)
            old = pd.to_numeric(df['isST'], errors='coerce').to_numpy()
            if np.array_equal(old, isst.astype(float), equal_nan=True):
                continue
            df['isST'] = isst
            write_ts(code, df)
            n += 1
        logger.info(f'[isst] 写 {n} 只（共 {len(codes)}）')

    def update_m15(self, codes=None):
        """15 分钟线（不复权原值）：近 64 交易日窗内缺口→新浪（并发）；更早缺口/整段→baostock（单线程，逐票续传）。"""
        codes = self._codes(codes)
        cal = _cal()
        sina, bs_plan, failed = {}, [], []
        for code in codes:
            try:
                plan = self._plan_m15(code, cal, self._end_of(code))
            except Exception as e:
                _fail('[m15]', code, e, failed)
                continue
            if plan is None:
                continue
            if plan[0] == 'sina':
                sina[code] = plan[1:]
            else:
                bs_plan.append((code, plan[1], plan[2]))
        logger.info(f'[m15] 新浪 {len(sina)} 只 / baostock {len(bs_plan)} 只'
                    + ('（baostock 单线程慢扫、逐票落盘续传，可 --codes 分批）' if bs_plan else ''))
        rows_by_code = {}
        done, t0 = 0, time.time()
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(self._fetch_m15, c, dl): (c, fm) for c, (dl, fm) in sina.items()}
            for f in as_completed(futs):
                done += 1
                c, fm = futs[f]
                _progress(done, len(futs), t0, '[m15] 新浪', f'已得 {len(rows_by_code)} 失败 {len(failed)}')
                try:
                    rows_by_code[c] = (f.result(), fm)
                except Exception as e:
                    _fail('[m15]', c, e, failed)
        nw = 0
        for code, (rows, fm) in rows_by_code.items():
            if rows is None or rows.empty:
                continue
            if self._m15_upsert(code, rows, force_merge=fm):
                nw += 1
        t0 = time.time()
        for i, (code, start, end) in enumerate(bs_plan, 1):
            try:
                rows = self._fetch_m15_bs(code, start, end)
                if len(rows) and self._m15_upsert(code, rows, force_merge=True):
                    nw += 1
            except Exception as e:
                _fail('[m15]', code, e, failed)
            _progress(i, len(bs_plan), t0, '[m15] baostock',
                      f'已写 {nw} 失败 {len(failed)} 未处理 {len(bs_plan) - i}', every=3)
        logger.info(f'[m15] 更新 {nw}/{len(codes)} 只，失败 {len(failed)} 只（重跑续传）')

    # ==================== 内部：边界判定 ====================

    def _codes(self, codes=None):
        if codes:
            return sorted(codes)
        m = self._meta()
        out = []
        for ts in m.index:
            c = _local(ts)
            s, e = self._start_of(c), self._end_of(c)
            has = os.path.exists(os.path.join(TS_DIR, f'{c}.csv'))
            if s is None and not has:
                continue                             # 未上市/预留码（名单无上市日）：不要求
            if s is not None and s > e and not has:
                continue                             # 上市晚于目标日 或 退市早于库起点
            out.append(c)
        if not out and not os.path.exists(TS_DIR):
            logger.warning(f'票池为空（{TS_DIR} 尚无数据）')
        return sorted(out)

    def _meta(self):
        if self._meta_df is None:
            if not os.path.exists(META_F):
                raise RuntimeError(f'缺少名单（{META_F}）——先跑一次完整 run 生成 meta')
            self._meta_df = pd.read_parquet(META_F).set_index('ts_code')
        return self._meta_df

    def _row(self, code):
        m = self._meta()
        ts = _ts_code(code)
        return m.loc[ts] if ts in m.index else None

    def _start_of(self, code):
        """该票应覆盖的首个交易日 = max(库起点, 上市日)；无上市日 → None。"""
        r = self._row(code)
        ld = _d8(r['list_date']) if r is not None else np.nan
        return max(FLOOR, ld) if isinstance(ld, str) else None

    def _end_of(self, code):
        """该票应覆盖的最后交易日 = min(目标日, 退市日)。"""
        r = self._row(code)
        dd = _d8(r['delist_date']) if r is not None else np.nan
        return min(self._target_date(), dd) if isinstance(dd, str) else self._target_date()

    def _target_date(self):
        if self._tgt is None:
            t = get_target_trade_date()
            self._tgt = pd.to_datetime(t).strftime('%Y-%m-%d') if t else pd.Timestamp.now().strftime('%Y-%m-%d')
        return self._tgt

    # ==================== 内部：daily 分批 / 合并 / 占位 ====================

    def _batch_daily(self, plan, cal, tgt):
        """按单次 6000 行上限贪心分批（行数 ≈ 区间交易日数；PRE_FLOOR 段计入）。"""
        batches, cur, rows = [], [], 0
        for code in sorted(plan):
            fs, e = plan[code]
            d = max(1, bisect.bisect_right(cal, min(e, tgt)) - bisect.bisect_left(cal, fs))
            if cur and (rows + d > BATCH_ROWS or len(cur) >= BATCH_CODES):
                batches.append(cur)
                cur, rows = [], 0
            cur.append(code)
            rows += d
        if cur:
            batches.append(cur)
        return batches

    def _apply_daily(self, code, sub, pub_today):
        """合并一批抓取行 → 归位（补占位）→ 写盘。返回是否写盘。"""
        rows = None
        if sub is not None and len(sub):
            d = pd.DataFrame({
                'date': [_iso(str(x)) for x in sub['trade_date']],
                'code': code,
                'open': pd.to_numeric(sub['open'], errors='coerce'),
                'high': pd.to_numeric(sub['high'], errors='coerce'),
                'low': pd.to_numeric(sub['low'], errors='coerce'),
                'close': pd.to_numeric(sub['close'], errors='coerce'),
                'preclose': pd.to_numeric(sub['pre_close'], errors='coerce'),
                # 手→股 / 千元→元；取整消除 float 乘法的 1ulp 噪声（成交量本身即整数股）
                'volume': np.round(pd.to_numeric(sub['vol'], errors='coerce') * 100, 0),
                'amount': np.round(pd.to_numeric(sub['amount'], errors='coerce') * 1000, 4),
                'turn': np.nan, 'tradestatus': 0,
                'pctChg': pd.to_numeric(sub['pct_chg'], errors='coerce'),
                'isST': np.nan})
            d = d.sort_values('date').drop_duplicates('date', keep='last')
            st = self._start_of(code)
            if st == FLOOR and code not in self._prev_mem:
                pre = d[d['date'] < FLOOR]
                if len(pre):
                    v = _num(pre['close'].iloc[-1])
                    if v is not None:
                        with self._lock:
                            self._prev_mem[code] = v              # 首行前收基准
            if st is not None:
                d = d[d['date'] >= st]
            rows = d.reset_index(drop=True)
        old = read_ts(code)
        if rows is None or not len(rows):
            if old.empty:
                return False
            merged = old
        else:
            merged = rows if old.empty else pd.concat([old, rows], ignore_index=True)
        out = self._materialize(merged, code, pub_today)
        if out.empty:
            return False
        write_ts(code, out)
        return True

    def _materialize(self, df, code, pub_today):
        """补齐 [起点, 终点] 每交易日一行：真成交行原样，其余（停牌/缺行/坏行）落停牌占位行。

        真成交 = volume>0 且 close>0；占位行 = 价格列 carry（上一真实收盘，首个真实行之前用
        库起点前收盘）、volume=amount=0、turn/pctChg 空、tradestatus=0。
        右边界 = min(目标日, 退市日, 日历末日)；D/P 状态无退市日时不越过最后一个真实行。
        当日（== 实际今天）源未入库（pub_today=False）时不落占位，留缺口下轮补。
        """
        df = df.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
        if df.empty:
            return df
        vols = pd.to_numeric(df['volume'], errors='coerce').to_numpy()
        closes = pd.to_numeric(df['close'], errors='coerce').to_numpy()
        real = (vols > 0) & np.isfinite(closes) & (closes > 0)
        cal = _cal()
        s = self._start_of(code)
        d0 = df['date'].iloc[0]
        if s:
            d0 = min(d0, s)
        e = min(self._end_of(code), cal[-1])
        r = self._row(code)
        if r is not None and str(r['list_status']) in ('D', 'P') and not isinstance(_d8(r['delist_date']), str):
            rd = df.loc[real, 'date']
            if len(rd):
                e = min(e, rd.iloc[-1])
        if e < d0:
            e = df['date'].iloc[-1]
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        have = set(df['date'])
        missing = [x for x in cal if d0 <= x <= e and x not in have and (x < today or (x == today and pub_today))]
        # 占位日期 = 整日缺行 + 非成交行（volume==0 的停牌行、有量无价的坏行一并归位）
        dates = sorted([d for d, ok in zip(df['date'].tolist(), real) if not ok] + missing)
        if dates:
            real_dates = df.loc[real, 'date'].to_numpy()
            real_closes = closes[real]
            if len(real_dates):
                pos = np.searchsorted(real_dates, np.asarray(dates), side='left') - 1
                carry = np.where(pos >= 0, real_closes[np.clip(pos, 0, None)], np.nan)
            else:
                carry = np.full(len(dates), np.nan)
            # 无真实行可依时：沿用行内原价，再回落库起点前收盘，别把价格抹成空
            own = pd.Series(closes, index=df['date']).reindex(dates).to_numpy()
            carry = np.where(np.isfinite(carry), carry, np.where(own > 0, own, np.nan))
            base = self._load_prev().get(code)
            if base is not None and _num(base) is not None and base > 0:
                carry = np.where(np.isfinite(carry), carry, float(base))
        else:
            carry = np.array([])
        out = df[real].reset_index(drop=True)              # 真成交行原样（含已算好的派生列）
        if dates:
            place = pd.DataFrame({'date': dates, 'code': df['code'].iloc[0],
                                  'open': carry, 'high': carry, 'low': carry, 'close': carry,
                                  'preclose': carry, 'volume': 0.0, 'amount': 0.0,
                                  'turn': np.nan, 'pctChg': np.nan, 'tradestatus': 0, 'isST': np.nan})
            out = place if out.empty else pd.concat([out, place], ignore_index=True)
        out = out.sort_values('date').reset_index(drop=True)
        out['tradestatus'] = (pd.to_numeric(out['volume'], errors='coerce') > 0).astype(int)
        return out

    # ==================== 内部：turn / adj 辅助表 ====================

    def _fetch_basic(self, dates):
        """按交易日抓 daily_basic（全市场一次/日，6 线程 + 全局限速）→ 紧凑 DataFrame（内存，不落盘）。

        现场抓取、现场回填：缺口 = 空 turn 的票日，抓完立即写回主表；无持久缓存 → 状态最小，
        中途崩溃则下轮按"仍有空 turn"自然重跑（幂等，无脏状态）。
        """
        frames, lock = [], threading.Lock()
        t0, cnt = time.time(), [0]

        def work(dt):
            r = _ts_daily_basic(_ymd(dt))
            rows = None
            if r is not None and len(r):
                rows = pd.DataFrame({
                    'code': [_local(x) for x in r['ts_code']],
                    'date': pd.to_numeric(r['trade_date'], errors='coerce').astype('int32'),
                    'turnover_rate': pd.to_numeric(r['turnover_rate'], errors='coerce')})
            with lock:
                cnt[0] += 1
                _progress(cnt[0], len(dates), t0, '[turn] 抓取', dt, every=100)
                if rows is None:
                    logger.warning(f'[turn] {dt} daily_basic 无返回（换手率留空，下轮再试）')
                    return
                frames.append(rows)

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(work, dates))
        if not frames:
            return pd.DataFrame(columns=['code', 'date', 'turnover_rate'])
        df = pd.concat(frames, ignore_index=True).drop_duplicates(['code', 'date'], keep='last')
        df['code'] = df['code'].astype('category')
        df['date'] = df['date'].astype('int32')
        return df

    def _flush_adj(self, frames):
        new = pd.concat(frames, ignore_index=True)
        old = pd.read_parquet(ADJ_F) if os.path.exists(ADJ_F) else None
        both = new if old is None else pd.concat([old, new], ignore_index=True)
        both = both.drop_duplicates(['code', 'date'], keep='last')
        both['code'] = both['code'].astype('category')
        both['date'] = both['date'].astype('int32')
        both['adj_factor'] = pd.to_numeric(both['adj_factor'], errors='coerce')
        both.sort_values(['code', 'date']).to_parquet(ADJ_F)

    # ==================== 内部：namechange ====================

    def _fetch_namechange(self):
        """曾用名事件表维护：按公告年份分段抓取（全市场，不带 ts_code），水位记 NAME_LOG。"""
        os.makedirs(AUX_DIR, exist_ok=True)
        log = json.load(open(NAME_LOG)) if os.path.exists(NAME_LOG) else {}
        today = pd.Timestamp.now().strftime('%Y%m%d')
        covered = None if self.force else log.get('covered_to')
        if covered and covered >= today:
            return
        cs = '19900101' if not covered else (pd.Timestamp(covered) + pd.Timedelta(days=1)).strftime('%Y%m%d')
        frames = []
        while cs <= today:
            ce = min(f'{cs[:4]}1231', today)
            r = _ts_namechange(cs, ce)
            if r is not None and len(r):
                frames.append(r)
            cs = f'{int(cs[:4]) + 1}0101'
            time.sleep(0.05)
        if frames:
            d = pd.concat(frames, ignore_index=True)
            d = d[d['ts_code'].str.endswith(('.SH', '.SZ'))]
            d = d.copy()
            d['code'] = [_local(x) for x in d['ts_code']]
            for c in ('start_date', 'end_date', 'ann_date'):
                d[c] = [_d8(x) for x in d[c]]
            d = d[['code', 'name', 'start_date', 'end_date', 'ann_date', 'change_reason']]
            d = d[d['start_date'].notna()]
            old = pd.read_parquet(NAME_F) if os.path.exists(NAME_F) else None
            both = d if old is None else pd.concat([old, d], ignore_index=True)
            both = both.drop_duplicates(['code', 'start_date'], keep='last').sort_values(['code', 'start_date'])
            both.to_parquet(NAME_F)
            logger.info(f'[isst] 曾用名事件累计 {len(both)} 条（本次 +{len(d)}）')
        else:
            logger.warning('[isst] namechange 无返回')
        json.dump({'covered_to': today}, open(NAME_LOG, 'w'))

    # ==================== 内部：prev_close ====================

    def _load_prev(self):
        """{code: 库起点前收盘}（首行前收基准 + 首个真实行之前的占位 carry）。"""
        if self._prev is None:
            self._prev = {}
            if os.path.exists(PREV_F):
                p = pd.read_parquet(PREV_F)
                self._prev = dict(zip(p['code'], p['prev_close']))
        d = dict(self._prev)
        with self._lock:
            d.update(self._prev_mem)
        return d

    def _flush_prev(self):
        if not self._prev_mem:
            return
        add = pd.DataFrame({'code': list(self._prev_mem), 'prev_close': list(self._prev_mem.values())})
        old = pd.read_parquet(PREV_F) if os.path.exists(PREV_F) else None
        both = add if old is None else pd.concat([old, add], ignore_index=True)
        both = both.drop_duplicates('code', keep='last').sort_values('code')
        os.makedirs(AUX_DIR, exist_ok=True)
        both.to_parquet(PREV_F)
        self._prev_mem = {}
        self._prev = None                     # 使下次 _load_prev 重读

    # ==================== 内部：m15（新浪补缺口 / baostock 老段） ====================

    def _plan_m15(self, code, cal, end_td):
        """该票 m15 缺口计划 → ('sina', datalen, force_merge) / ('bs', start, end) / None（已齐）。

        新浪只支持"最近 N 根"（≈64 交易日窗）：窗内缺口用新浪（快、可并发）；
        窗口外的老缺口与库起点起整段用 baostock（慢、单线程、但全，可续传）。
        只要求「真成交日」（日线 volume>0）的分钟 bar：停牌日没有任何源的 bar，不算缺口。
        """
        path = os.path.join(M15_DIR, f'{code}.csv')
        start_td = self._start_of(code) or FLOOR
        i_end = bisect.bisect_left(cal, end_td)
        if not os.path.exists(path):
            return ('bs', start_td, end_td)
        dates = set(pd.read_csv(path, usecols=['date'], dtype={'date': str})['date'])
        if not dates:
            return ('bs', start_td, end_td)
        dp = os.path.join(TS_DIR, f'{code}.csv')
        if os.path.exists(dp):
            dd = pd.read_csv(dp, usecols=['date', 'volume'], dtype={'date': str})
            expect = set(dd.loc[pd.to_numeric(dd['volume'], errors='coerce') > 0, 'date'])
        else:
            expect = None                       # 无日线参照 → 按全部交易日要求
        lo = self._start_of(code) or min(dates)
        i0 = bisect.bisect_left(cal, lo)
        miss = [d for d in cal[i0:i_end + 1] if d not in dates and (expect is None or d in expect)]
        if not miss:
            return None
        if miss[0] < cal[max(0, i_end - SINA_WIN)]:
            return ('bs', miss[0], end_td)             # 老缺口超出新浪窗 → baostock 一段补齐
        datalen = min(SINA_MAX_BARS, (i_end - bisect.bisect_left(cal, miss[0]) + 1) * BARS_PER_DAY + BARS_PER_DAY)
        return ('sina', datalen, min(miss) <= max(dates))   # 缺口中段→合并；仅尾部→追加

    def _fetch_m15(self, code, datalen):
        sym = code.replace('.', '')
        s = getattr(_TLS, 'sina', None)
        if s is None:
            s = _TLS.sina = requests.Session()
            s.headers['User-Agent'] = 'Mozilla/5.0'
        last = None
        for _ in range(3):
            try:
                r = s.get(SINA_KLINE, params={'symbol': sym, 'scale': '15', 'ma': 'no', 'datalen': str(datalen)},
                          timeout=10)
                arr = json.loads(r.text.strip())
                if not isinstance(arr, list) or not arr:
                    return pd.DataFrame()
                rows = [{'date': it['day'][:10],
                         'time': int(it['day'][11:13]) * 100 + int(it['day'][14:16]),
                         'open': float(it['open']), 'high': float(it['high']), 'low': float(it['low']),
                         'close': float(it['close']), 'volume': int(float(it['volume'])),
                         'amount': float(it['amount'])} for it in arr]
                time.sleep(random.uniform(0.02, 0.06))
                return pd.DataFrame(rows)
            except Exception as e:
                last = e
                time.sleep(random.uniform(0.5, 1.5))
        raise last

    def _fetch_m15_bs(self, code, start, end):
        """baostock 15 分钟一段（不复权 adjustflag='3'，与新浪同口径）→ 与新浪同构的行帧（time=HHMM 整数）。"""
        rows = _bs_rows(code, 'date,time,open,high,low,close,volume,amount', start, end,
                        frequency='15', adjustflag='3')
        out = [{'date': r[0], 'time': int(r[1][8:12]),
                'open': _num(r[2]), 'high': _num(r[3]), 'low': _num(r[4]),
                'close': _num(r[5]), 'volume': _num(r[6]), 'amount': _num(r[7])}
               for r in rows if _num(r[5]) is not None]
        return pd.DataFrame(out)

    def _m15_upsert(self, code, rows, force_merge=False):
        path = os.path.join(M15_DIR, f'{code}.csv')
        os.makedirs(M15_DIR, exist_ok=True)   # to_csv 不会建父目录；空库首跑否则 OSError
        if not os.path.exists(path):
            rows.insert(1, 'code', code)
            rows[M15_COLS].to_csv(path, index=False)
            return True
        tail = read_last_lines(path, n_lines=1)
        last = tail[-1].split(',') if tail else None
        last_dt = (last[0], int(last[1])) if last and len(last) >= 2 else None
        new_rows = rows if last_dt is None else rows[[(r.date, r.time) > last_dt for r in rows.itertuples()]]
        if len(new_rows) == 0 and not force_merge:
            return False
        if not force_merge and list(new_rows.index) == list(range(len(rows) - len(new_rows), len(rows))):
            out = new_rows.copy()   # 抓取窗与本地衔接（新区间=连续后缀）→ 直接追加
            out.insert(1, 'code', code)
            out[M15_COLS].to_csv(path, mode='a', header=False, index=False)
            return True
        old = pd.read_csv(path, dtype={'date': str})
        m = pd.concat([old, rows.assign(code=code)], ignore_index=True)
        m = m.drop_duplicates(['date', 'time'], keep='last').sort_values(['date', 'time'])
        m[M15_COLS].to_csv(path, index=False)
        return True


# ==================== 状态问答 / 自检 ====================

def is_latest(quiet=False):
    """整库各字段是否都无缺口（复用与 update_* 同一套缺口判定）。

    逐票：①覆盖区间 [上市日或库起点, 退市日或目标日] 无缺日；②真成交行（volume>0）的
    preclose/pctChg/amount/turn 非空；③isST 非空；④有成交的票 m15 无缺口（同一个 _plan_m15）。
    豁免「重跑也补不回来」的票：退市、本地无数据（源里没有该票）、turn 整票全缺（源不覆盖）。
    注：代价是要读全表，比抽末行慢。
    """
    if not os.path.exists(META_F):
        if not quiet:
            logger.warning('[is_latest] 尚未初始化（无 meta 名单）→ False')
        return False
    b = TushareKline()
    tgt = b._target_date()
    codes = b._codes()
    if not codes:
        if not quiet:
            logger.warning(f'[is_latest] 票池为空（{TS_DIR}）→ False')
        return False
    cal = _cal()
    lags, exempt, n_data = {}, {}, 0
    for code in codes:
        r = b._row(code)
        if r is not None and str(r['list_status']) == 'D':
            exempt.setdefault('delisted', []).append(code)
            continue
        df = read_ts(code)
        if df.empty:
            exempt.setdefault('no_data', []).append(code)      # 源里没有该票（或尚未建库）
            continue
        n_data += 1
        lo = b._start_of(code) or df['date'].iloc[0]
        hi = b._end_of(code)
        have = set(df['date'])
        if any(lo <= d <= hi and d not in have for d in cal):
            lags.setdefault('date', []).append(code)
        traded = df[pd.to_numeric(df['volume'], errors='coerce') > 0]   # 真成交行（volume>0）
        if len(traded):
            for col in ('preclose', 'pctChg'):
                blank = pd.to_numeric(traded[col], errors='coerce').isna()
                if blank.any() and not (blank.sum() == 1 and bool(blank.iloc[0])):
                    lags.setdefault(col, []).append(code)      # 首行无前收（新股）当例外
            if pd.to_numeric(traded['amount'], errors='coerce').isna().any():
                lags.setdefault('amount', []).append(code)
            blank_turn = pd.to_numeric(traded['turn'], errors='coerce').isna()
            if blank_turn.any():
                if blank_turn.all():
                    exempt.setdefault('turn', []).append(code)  # 整票无换手率（源缺）→ 算不出
                else:
                    lags.setdefault('turn', []).append(code)
            if b._plan_m15(code, cal, b._end_of(code)) is not None:
                lags.setdefault('m15', []).append(code)
        if pd.to_numeric(df['isST'], errors='coerce').isna().any():
            lags.setdefault('isST', []).append(code)
    if not n_data:
        if not quiet:
            logger.warning(f'[is_latest] 库尚无任何行情（{TS_DIR}）→ False')
        return False
    note = '，'.join(f'{k}={len(v)}' for k, v in sorted(exempt.items()))
    if not lags:
        if not quiet:
            logger.info(f'[is_latest] 全部字段已达 {tgt}（{len(codes)} 只'
                        + (f'；已豁免 {note}' if note else '') + '）')
        return True
    if not quiet:
        for k, v in sorted(lags.items()):
            logger.warning(f'[is_latest] {k} 未达 {tgt}: {len(v)} 只（样例: {v[:5]}）')
        if note:
            logger.info(f'[is_latest] 已豁免（源不提供/退市/无数据）: {note}')
    return False


def probe():
    """连通性/权限自检：逐个接口调一次并打印结果（不写任何数据）。"""
    src = ('环境变量 TUSHARE_TOKEN' if os.environ.get('TUSHARE_TOKEN')
           else (TOKEN_FILE if os.path.exists(TOKEN_FILE) else None))
    if src is None:
        logger.error(f'[probe] 未找到 token：写入 {TOKEN_FILE}（单行纯文本）或设环境变量 TUSHARE_TOKEN')
        return 1
    logger.info(f'[probe] token 来源: {src}')
    cal = _cal()
    today = pd.Timestamp.now().strftime('%Y-%m-%d')
    d = ([x for x in cal if x <= today] or [today])[-1]
    ok = True

    def check(name, fn):
        nonlocal ok
        try:
            r = fn()
            logger.info(f'[probe] {name}: OK（{len(r) if r is not None else 0} 行）')
        except Exception as e:
            ok = False
            logger.error(f'[probe] {name}: 失败 — {e}')

    check('stock_basic(L)', lambda: _ts_stock_basic('L'))
    check(f'daily(600000.SH @ {d})', lambda: _ts_daily(['600000.SH'], _ymd(d), _ymd(d)))
    check(f'daily_basic({d})', lambda: _ts_daily_basic(_ymd(d)))
    check(f'adj_factor({d})', lambda: _ts_adj(_ymd(d)))
    check('namechange(全市场按公告年份段)', lambda: _ts_namechange(f'{int(d[:4]) - 1}0101', _ymd(d)))
    check('namechange(单票)', lambda: _call(lambda: _pro().namechange(ts_code='600000.SH'), 'namechange'))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description='日K tushare 版构建器（字段自维护：缺口与最小请求）')
    ap.add_argument('--codes', default='', help='逗号分隔，如 600519,sz.000001（缺省=全市场沪深）')
    ap.add_argument('--skip', default='', help='跳过字段，如 turn,isst')
    ap.add_argument('--force', action='store_true', help='强制重取（忽略缺口/已抓日志）')
    ap.add_argument('--latest', action='store_true', help='只做整库状态问答（is_latest），不抓数据')
    ap.add_argument('--probe', action='store_true', help='接口连通/权限自检（不写数据）')
    a = ap.parse_args()
    try:
        if a.latest:
            sys.exit(0 if is_latest() else 1)
        if a.probe:
            sys.exit(probe())
        codes = norm_codes(a.codes) if a.codes else None
        TushareKline(force=a.force).run(codes, skip=[s for s in a.skip.split(',') if s])
    except Exception as e:
        logger.error(f'fatal: {type(e).__name__}: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
