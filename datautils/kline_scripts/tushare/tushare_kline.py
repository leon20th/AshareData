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
     isST（历史）                  stock_st        各交易日风险警示名单（tushare 官方接口，原样落库、不做推断）
     复权因子（aux，读时复权用）    adj_factor      全市场按交易日；不做复权计算，读时另折算
     名单/上市退市日                stock_basic     L/D/P 三态
     tradestatus（停牌占位行）      ——              由本地交易日历 + daily 缺行派生（与 v2 同语义）
     交易日历/目标日                ——              沿用本地 a_open（baostock 维护），与全站一致

  3. 抓取策略（最小请求）：
     - daily：按票缺口贪心分批（多代码一次调用，6000 行/次上限）；只请求缺失区间（冷建多取库起点前一段）。
             turn（daily_basic）与 isST（stock_st）都是"当日全市场"接口，故先按本轮要新增的交易日
             各抓一次（多线程 + 全局限速），落盘时与 OHLCV 一并写好 —— 每只票每轮**只写一次**。
     - adj  ：按交易日缺口全市场各取一次；已抓日期记 _aux/adj_fetch_log.json。
  4. 口径与 v2（quick_kline）逐位对齐：13 列 schema 与列序、未复权原值、每交易日一行、
     停牌占位（volume==0：价格 carry、amount=0、turn/pctChg 空、tradestatus=0）、
     覆盖 [max(库起点, 上市日), min(目标日, 退市日)]、首行前收基准（库起点前收盘，_aux/prev_close）。
     当日（目标日==今天）源未入库时不落占位行，留缺口下轮补（源已入库则当晚补齐停牌票）。
  5. 老数据不移植（用户定案）：fresh 建库，不读旧库做历史覆盖。

产物: daily_kline_ts/{sh.600000.csv} 13 列（未复权原值）；m15 由共用模块 kline_scripts/m15/update_m15.py 维护（m15_kline_ts，两管线共用）
辅助: daily_kline_ts/_aux/（meta_tickers / adj_factor+日志 / prev_close）
用法:
  python AshareData/datautils/kline_scripts/tushare/tushare_kline.py [--codes 600519,sz.000001] [--skip adj]
  python AshareData/datautils/kline_scripts/tushare/tushare_kline.py --latest   # 只问库状态，不抓数据
  python AshareData/datautils/kline_scripts/tushare/tushare_kline.py --probe    # 连通性/权限自检（不写数据）
token: 环境变量 TUSHARE_TOKEN 优先，其次 AshareData/.keys/.tushare_token（单行纯文本；配置见 setup.sh）
参考耗时（全市场 2020 起冷建）：日线部分 ≈5.1k 次调用、约 15 分钟；turn/isST 各 1636 次当日接口调用（多线程 ≈10 分钟）。此后增量只补新交易日，且每票每轮只写一次。
"""
import argparse
import bisect
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from AshareData.paths import ASHARE_ROOT, DAILY_KLINE_TS_DIR
from AshareData.utils.exchanges_utils.a_open import get_target_trade_date, get_trade_date_list
from AshareData.utils.log_util import get_logger

logger = get_logger('日Kts')

TOKEN_FILE = os.path.join(ASHARE_ROOT, '.keys', '.tushare_token')
TS_DIR = DAILY_KLINE_TS_DIR
AUX_DIR = os.path.join(TS_DIR, '_aux')
META_F = os.path.join(AUX_DIR, 'meta_tickers.parquet')
ADJ_F = os.path.join(AUX_DIR, 'adj_factor.parquet')
ADJ_LOG = os.path.join(AUX_DIR, 'adj_fetch_log.json')
PREV_F = os.path.join(AUX_DIR, 'prev_close.parquet')


COLS = ['date', 'code', 'open', 'high', 'low', 'close', 'preclose',
        'volume', 'amount', 'turn', 'tradestatus', 'pctChg', 'isST']
FLOOR = '2020-01-02'            # 库起点（与全站消费链对齐）
PRE_FLOOR = '2019-10-01'        # 冷建时多取一段：供"首行前收基准"（库起点前末收盘）
BATCH_ROWS = 5500               # daily 单次调用行数上限（接口 6000，留余量）
BATCH_CODES = 50                # daily 单次调用代码数上限
FLUSH_N = 300                   # adj 抓取过程每 N 个交易日落一次盘（控内存）
WORKERS = 6                     # 并发抓取线程数（总频率由 _pace 统一限速）
PACE_GAP = 0.15                 # 全局最小请求间隔（秒）≈ 400 次/分，留出 500/分限频余量

_CAL = None
_PRO = None
_PACE_LK = threading.Lock()
_PACE_LAST = [0.0]


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
    """tushare 客户端（懒初始化）：环境变量 TUSHARE_TOKEN 优先，其次 .keys/.tushare_token。

    注：本项目 kline_scripts/tushare/ 与 PyPI 包同名，导入前剔除遮蔽路径，确保拿到真包。
    """
    global _PRO
    if _PRO is None:
        token = os.environ.get('TUSHARE_TOKEN', '').strip()
        if not token and os.path.exists(TOKEN_FILE):
            token = open(TOKEN_FILE, encoding='utf-8').read().strip()
        if not token:
            raise RuntimeError(f'缺少 tushare token：写入 {TOKEN_FILE}（单行纯文本）'
                               f'或设环境变量 TUSHARE_TOKEN（配置见 AshareData/setup.sh）')
        # 本项目 kline_scripts/tushare/ 与 PyPI 包同名：当 kline_scripts/ 位于 sys.path 前部
        # （如从该目录下的脚本启动）时，裸 import 会命中本地包；先剔除遮蔽路径与假缓存
        import sys as _sys
        _mod = _sys.modules.get('tushare')
        if _mod is not None and not hasattr(_mod, 'pro_api'):
            del _sys.modules['tushare']
        _shadow = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # .../kline_scripts
        _saved = _sys.path[:]
        try:
            _sys.path[:] = [p for p in _sys.path if os.path.abspath(p or '.') != _shadow]
            import tushare as ts
        finally:
            _sys.path[:] = _saved
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


def _ts_stock_st(date):
    """ST 名单（tushare 官方 ST 接口）——按交易日返回风险警示板股票。

    字段：ts_code/name/trade_date/type/type_name（type='ST'，type_name='风险警示板'）。
    覆盖 2020-01-02 起（与库起点一致）；不带 trade_date 时可按 ts_code 取单票全历史。
    这是 tushare 口径下 isST 的**权威来源**（无需从 name 前缀推断）。
    """
    return _call(lambda: _pro().stock_st(trade_date=date, fields='trade_date,ts_code,name'),
                 'stock_st')


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


# ==================== 构建器 ====================

class TushareKline:
    """tushare 版日K构建器：字段自维护（缺口判定 + 最小请求 + 幂等）。

    统一签名 update_<字段>(codes=None)：codes 为 None 时取 meta 全市场沪深（未上市/退市票按各自边界处理）。
    """

    FIELDS = (('daily', 'OHLCV+前收/涨跌幅+停牌占位+换手率+isST'), ('adj', '复权因子（读时复权用）'))

    def __init__(self, force=False):
        self.force = force                 # 强制重取（忽略缺口/已抓日志）
        self._meta_df = None
        self._prev = None                  # prev_close 文件缓存
        self._prev_mem = {}                # 冷建过程中新得的"库起点前收盘"
        self._lock = threading.Lock()      # 保护 _prev_mem 的并发读写
        self._st_mem = {}                  # {交易日: set(本地代码)}：stock_st 当日名单（本次运行内缓存）
        self._st_lk = threading.Lock()     # 保护 _st_mem 的并发填充
        self._turn_mem = {}                # {交易日: Series(本地代码 → 换手率)}：daily_basic（本次运行内缓存）
        self._turn_lk = threading.Lock()   # 保护 _turn_mem 的并发填充
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
                if fs is None and not df.empty and self._turn_lag(df, e):
                    fs = str(e)                  # 换手率滞后自愈：目标日已成交但当日晚间才出换手率
                if fs is None:
                    continue
            cold[code] = df.empty
            if df.empty and s == FLOOR:
                fs = PRE_FLOOR                   # 冷建多取一段 → 首行前收基准
            plan[code] = (fs, e)
        if not plan:
            logger.info('[daily] 各票均无缺口 → 跳过')
            return
        # turn / isST 都是"当日全市场"接口：先按本轮要新增的交易日抓齐，落盘时一次写成（不二次重写）
        need = self._plan_dates(plan, cal, tgt)
        self._fetch_st(need)
        self._fetch_turn(need)
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

    # ==================== 内部：turn / isST（当日全市场接口） ====================

    def _turn_lag(self, df, e):
        """末行是否"目标日已成交但换手率留空"：daily_basic 当晚才出时用它让下轮重取该日补上。

        仅当该票换手率有覆盖（源覆盖该票）才算缺口——整票无覆盖的票豁免，避免每轮空转。
        """
        last = df.iloc[-1]
        if str(last['date']) != str(e) or not _num(last['volume']):
            return False
        if _num(last['turn']) is not None:
            return False
        return bool(pd.to_numeric(df['turn'], errors='coerce').notna().any())

    def _fetch_turn(self, dates):
        """按交易日抓换手率（daily_basic 的 turnover_rate）→ {交易日: Series(本地代码 → 换手率)}。

        只抓没抓过的日子（多线程 + 全局限速由 `_fetch_basic` 内部负责）；源缺该日则不缓存，
        该行留空、下轮由 `_turn_lag` 触发重取。
        """
        todo = [d for d in dates if d not in self._turn_mem]
        if not todo:
            return
        got = self._fetch_basic(sorted(todo))
        for dt, g in got.groupby('date', sort=False, observed=True):
            with self._turn_lk:
                self._turn_mem[_iso(str(dt))] = pd.Series(g['turnover_rate'].to_numpy(),
                                                          index=g['code'].to_numpy())

    def _plan_dates(self, plan, cal, tgt):
        """本轮各票要新增的交易日并集（各票 [起点, 终点] 窗口，裁到库起点）——isST 只需这些天。"""
        mask = np.zeros(len(cal), dtype=bool)
        for fs, e in plan.values():
            lo, hi = bisect.bisect_left(cal, max(fs, FLOOR)), bisect.bisect_right(cal, min(e, tgt))
            if hi > lo:
                mask[lo:hi] = True
        return [cal[i] for i in np.flatnonzero(mask)]

    def _fetch_st(self, dates):
        """按交易日抓 ST 名单（tushare 官方 stock_st）→ {交易日: set(本地代码)}，本次运行内缓存。

        只抓没抓过的日子（6 线程 + 全局限速，与 daily_basic/adj 同模式）；返回空表视为该日
        源暂缺（不缓存），其行不落盘、下轮再来。
        """
        todo = [d for d in dates if d not in self._st_mem]
        if not todo:
            return
        t0, cnt = time.time(), [0]

        def work(dt):
            try:
                r = _ts_stock_st(_ymd(dt))
                bad = None if (r is not None and len(r)) else '无返回'
            except Exception as e:
                r, bad = None, f'失败 {str(e)[:80]}'
            with self._st_lk:
                cnt[0] += 1
                _progress(cnt[0], len(todo), t0, '[isST] 抓 ST 名单', dt, every=200)
                if bad:
                    logger.warning(f'[isST] {dt} stock_st {bad}（该日留缺口下轮补）')
                    return
                self._st_mem[dt] = {_local(x) for x in r['ts_code']}

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            list(ex.map(work, todo))
        logger.info(f'[isST] ST 名单：本轮新抓 {len(todo)} 个交易日（缓存 {len(self._st_mem)} 天）')

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
        # turn / isST 都直接取当日接口数据：旧行沿用已写好的值，其余按当日数据落值 → 落盘只写一次
        pv = {c: v for c, v in zip(old['date'], pd.to_numeric(old['turn'], errors='coerce'))} if len(old) else {}
        out['turn'] = out['date'].map(pv)
        pv = {c: v for c, v in zip(old['date'], pd.to_numeric(old['isST'], errors='coerce'))} if len(old) else {}
        out['isST'] = out['date'].map(pv)
        lag_t = (pd.to_numeric(out['volume'], errors='coerce') > 0) & out['turn'].isna()
        if lag_t.any():
            self._fetch_turn(sorted(set(out.loc[lag_t, 'date'])))
            hit = lag_t & out['date'].isin(list(self._turn_mem))
            out.loc[hit, 'turn'] = [self._turn_mem[d].get(c, np.nan)
                                    for c, d in zip(out.loc[hit, 'code'], out.loc[hit, 'date'])]
        lag_s = out['isST'].isna()
        if lag_s.any():
            self._fetch_st(sorted(set(out.loc[lag_s, 'date'])))
            hit = lag_s & out['date'].isin(list(self._st_mem))
            out.loc[hit, 'isST'] = [1.0 if c in self._st_mem[d] else 0.0
                                    for c, d in zip(out.loc[hit, 'code'], out.loc[hit, 'date'])]
            out = out[out['isST'].notna()].reset_index(drop=True)   # ST 源缺日不落盘 → 留缺口下轮补
            if out.empty:
                return False
        out['isST'] = out['isST'].astype(int)
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

# ==================== 状态问答 / 自检 ====================

def is_latest(quiet=False):
    """整库各字段是否都无缺口（复用与 update_* 同一套缺口判定）。

    逐票：①覆盖区间 [上市日或库起点, 退市日或目标日] 无缺日；②真成交行（volume>0）的
    preclose/pctChg/amount/turn 非空；③isST 非空。（m15 由共用模块 kline_scripts/m15/update_m15.py 自检）
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
    check('stock_st(ST名单)', lambda: _call(lambda: _pro().stock_st(trade_date='20260930',
                                                                   fields='trade_date,ts_code,name'), 'stock_st'))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description='日K tushare 版构建器（字段自维护：缺口与最小请求）')
    ap.add_argument('--codes', default='', help='逗号分隔，如 600519,sz.000001（缺省=全市场沪深）')
    ap.add_argument('--skip', default='', help='跳过字段，目前仅 adj（turn/isST 已随 daily 一次写入）')
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
