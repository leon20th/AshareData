# -*- coding: utf-8 -*-
"""update_m15 —— 共用 15 分钟线更新器（新浪 + baostock；两套日K管线共用）。

数据源与路由:
  - 近 ~64 交易日窗内缺口 → 新浪（getKLineData，快、并发 6 线程）
  - 更早缺口 → baostock（只拉「最早~最晚缺口」最小区间，单线程、逐票续传）
  - 缺口判定以日线参照库（--daily-ref）的 volume>0 为「应有 bar」标准：停牌日不算缺口。
  - 产物固定写共用库 m15_kline_ts/（tushare 管线与 mix_source 管线共用同一份 m15）。

源坏自动跳过（2026-09-29 新增）:
  - baostock 成功返回但缺口日期无数据（源缺，如 2021-08-24 等历史事故日）→ 日期记入
    m15_kline_ts/_aux/m15_unavailable.json；此后各轮自动跳过（不再重复徒劳查询）。
  - 拉取异常（网络/超时）不记录（下轮自然重试）；--force 忽略记录强制重试全部。
  - 安全阀：单轮新增记录超阈值（>120 只 或 >400 日期）视为源头大面积异常 → 拒绝写入
    跳过表并告警（防止把「源暂时挂了」误记为永久跳过）。

用法:
  python update_m15.py                       # 全市场增量（默认参照 tushare 日线）
  python update_m15.py --daily-ref mix       # 参照 daily_kline_v2（mix_source 管线）
  python update_m15.py --codes 600519,sz.000001
  python update_m15.py --force               # 忽略"源不可得"记录，强制重试
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

from AshareData.paths import DAILY_KLINE_TS_DIR, DAILY_KLINE_V2_DIR, M15_KLINE_TS_DIR
from AshareData.utils.exchanges_utils.a_open import get_target_trade_date, get_trade_date_list
from AshareData.utils.log_util import get_logger
from AshareData.utils.read_file_utils import read_last_lines

logger = get_logger('m15更新')

# ==================== 常量 ====================

M15_DIR = M15_KLINE_TS_DIR                  # 共用 m15 库（tushare 与 mix_source 两管线共用）
M15_COLS = ['date', 'time', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
FLOOR = '2020-01-02'                        # 库起点（与消费链对齐）
UNAVAIL_F = os.path.join(M15_DIR, '_aux', 'm15_unavailable.json')   # 源不可得跳过表
SINA_KLINE = 'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData'
SINA_MAX_BARS = 1023             # 新浪单次请求上限（≈64 个交易日的 15 分钟 bar）
SINA_WIN = 62                    # 新浪可回补窗口（交易日数）
BARS_PER_DAY = 16
BS_TIMEOUT = 120                 # baostock 单次查询看门狗（连接半开有卡死前科）
MAX_NEW_CODES = 120              # 跳过表单轮新增票数安全阀（正常首轮 ~86 只）
MAX_NEW_DATES = 400              # 跳过表单轮新增日期数安全阀（正常首轮 ~90 个）

_CAL = None
_BS_ON = False                   # baostock 登录态（卡死/断线后强制重登）
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


def _fail(tag, code, err, failed, cap=3):
    """逐票失败（网络/服务等不可避免的缺失）：前 cap 条详列，之后只计数（末尾汇总）。"""
    failed.append(code)
    if len(failed) <= cap:
        logger.warning(f'{tag} {code} 失败: {err}')
    elif len(failed) == cap + 1:
        logger.warning(f'{tag} 失败已超 {cap} 条，后续只计数')


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


# ==================== 更新器 ====================

class M15Updater:
    """共用 m15 更新器：新浪窗内补 / baostock 老段，缺口最小请求 + 源缺自动跳过。"""

    def __init__(self, force=False, daily_ref='tushare'):
        self.force = force                  # 忽略"源不可得"跳过记录，强制重试
        self.ts_dir = DAILY_KLINE_TS_DIR if daily_ref == 'tushare' else DAILY_KLINE_V2_DIR
        self.meta_f = os.path.join(self.ts_dir, '_aux', 'meta_tickers.parquet')
        self.unavail = self._load_unavail()
        self.new_unavail = {}               # 本轮新增（提交时合并落盘）
        self._meta_df = None
        self._tgt = None

    # ------------------------------------------------------------------
    # 源缺跳过表
    # ------------------------------------------------------------------

    @staticmethod
    def _load_unavail():
        if os.path.exists(UNAVAIL_F):
            try:
                return json.load(open(UNAVAIL_F, encoding='utf-8'))
            except Exception as e:
                logger.warning(f'[m15] 跳过表读取失败（按空处理）: {e}')
        return {}

    def _commit_unavail(self):
        """本轮新增的「源缺」日期合并写入跳过表（带大面积异常安全阀）。"""
        new = self.new_unavail
        if not new:
            return
        n_dates = sum(len(v) for v in new.values())
        if len(new) > MAX_NEW_CODES or n_dates > MAX_NEW_DATES:
            logger.warning(f'[m15] 疑似数据源大面积异常（本轮新增源缺 {len(new)} 只/{n_dates} 个日期 超阈值 '
                           f'{MAX_NEW_CODES}只/{MAX_NEW_DATES}日期）——暂不写入跳过表，下轮自动重试')
            return
        for c, ds in new.items():
            self.unavail.setdefault(c, {}).update(ds)
        os.makedirs(os.path.dirname(UNAVAIL_F), exist_ok=True)
        json.dump(self.unavail, open(UNAVAIL_F, 'w', encoding='utf-8'), ensure_ascii=False, indent=0)
        logger.info(f'[m15] 跳过表更新：本轮新增 {len(new)} 只/{n_dates} 个源缺日期（累计 {len(self.unavail)} 只）')

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def update_m15(self, codes=None):
        """15 分钟线（不复权原值）：窗内缺口→新浪（并发）；更早缺口→baostock 最小缺口区间（逐票续传）。

        源缺自动跳过：baostock 成功返回但缺口日期无数据 → 记入跳过表，此后各轮不再请求。
        """
        codes = self._codes(codes)
        cal = _cal()
        sina, bs_plan, failed = {}, [], []
        n_skipped = 0
        for code in codes:
            try:
                plan = self._plan_m15(code, cal, self._end_of(code))
            except Exception as e:
                _fail('[m15]', code, e, failed)
                continue
            if plan is None:
                if code in self.unavail and not self.force:
                    n_skipped += 1
                continue
            if plan[0] == 'sina':
                sina[code] = plan[1:]
            else:
                bs_plan.append((code, *plan[1:]))
        logger.info(f'[m15] 新浪 {len(sina)} 只 / baostock {len(bs_plan)} 只'
                    + (f'（跳过表命中 {n_skipped} 只）' if n_skipped else '')
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
        for i, (code, start, end, miss) in enumerate(bs_plan, 1):
            try:
                rows = self._fetch_m15_bs(code, start, end)
                if miss is not None:
                    got = set(rows['date']) if len(rows) else set()
                    lack = [d for d in miss if d not in got]
                    if lack:
                        self.new_unavail.setdefault(code, {})
                        today = pd.Timestamp.now().strftime('%Y-%m-%d')
                        for d in lack:
                            self.new_unavail[code][d] = today     # 源缺：记录并跳过后续轮次
                    if len(rows):
                        rows = rows[[d in miss for d in rows['date']]]
                if len(rows) and self._m15_upsert(code, rows, force_merge=True):
                    nw += 1
            except Exception as e:
                _fail('[m15]', code, e, failed)                    # 异常不记录（下轮自然重试）
            _progress(i, len(bs_plan), t0, '[m15] baostock',
                      f'已写 {nw} 失败 {len(failed)} 未处理 {len(bs_plan) - i}', every=3)
        self._commit_unavail()
        if self.unavail:
            n_dates = sum(len(v) for v in self.unavail.values())
            logger.info(f'[m15] 源缺跳过表：累计 {len(self.unavail)} 只 / {n_dates} 个日期（--force 可强制重试）')
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
            has = os.path.exists(os.path.join(self.ts_dir, f'{c}.csv'))
            if s is None and not has:
                continue                             # 未上市/预留码（名单无上市日）：不要求
            if s is not None and s > e and not has:
                continue                             # 上市晚于目标日 或 退市早于库起点
            out.append(c)
        if not out and not os.path.exists(self.ts_dir):
            logger.warning(f'票池为空（{self.ts_dir} 尚无数据）')
        return sorted(out)

    def _meta(self):
        if self._meta_df is None:
            if not os.path.exists(self.meta_f):
                raise RuntimeError(f'缺少名单（{self.meta_f}）——先跑一次对应管线的日K run 生成 meta')
            df = pd.read_parquet(self.meta_f)
            if 'ts_code' not in df.columns and 'thscode' in df.columns:
                df = df.rename(columns={'thscode': 'ts_code'})   # v2(扶摇) meta 兼容
            self._meta_df = df.set_index('ts_code')
        return self._meta_df

    def _row(self, code):
        m = self._meta()
        ts = _ts_code(code)
        return m.loc[ts] if ts in m.index else None

    def _start_of(self, code):
        """该票应覆盖的首个交易日 = max(库起点, 上市日)；无上市日 → None。"""
        r = self._row(code)
        ld = _d8(r.get('list_date')) if r is not None else np.nan
        return max(FLOOR, ld) if isinstance(ld, str) else None

    def _end_of(self, code):
        """该票应覆盖的最后交易日 = min(目标日, 退市日)；无退市日 → 目标日。"""
        r = self._row(code)
        dd = _d8(r.get('delist_date')) if r is not None else np.nan
        return min(self._target_date(), dd) if isinstance(dd, str) else self._target_date()

    def _target_date(self):
        if self._tgt is None:
            t = get_target_trade_date()
            self._tgt = pd.to_datetime(t).strftime('%Y-%m-%d') if t else pd.Timestamp.now().strftime('%Y-%m-%d')
        return self._tgt

    # ==================== 内部：m15（新浪补缺口 / baostock 老段） ====================

    def _plan_m15(self, code, cal, end_td):
        """该票 m15 缺口计划 → ('sina', datalen, force_merge) / ('bs', start, end, miss_set|None) / None（已齐）。

        新浪只支持"最近 N 根"（≈64 交易日窗）：窗内缺口用新浪（快、可并发）；
        窗口外的老缺口用 baostock 只拉「最早~最晚缺口」区间、库起点起整段才拉全（单线程、可续传）。
        已记入"源缺跳过表"的日期直接剔除（--force 时忽略）。
        只要求「真成交日」（日线 volume>0）的分钟 bar：停牌日没有任何源的 bar，不算缺口。
        """
        path = os.path.join(M15_DIR, f'{code}.csv')
        start_td = self._start_of(code) or FLOOR
        i_end = bisect.bisect_left(cal, end_td)
        if not os.path.exists(path):
            return ('bs', start_td, end_td, None)
        dates = set(pd.read_csv(path, usecols=['date'], dtype={'date': str})['date'])
        if not dates:
            return ('bs', start_td, end_td, None)
        dp = os.path.join(self.ts_dir, f'{code}.csv')
        if os.path.exists(dp):
            dd = pd.read_csv(dp, usecols=['date', 'volume'], dtype={'date': str})
            expect = set(dd.loc[pd.to_numeric(dd['volume'], errors='coerce') > 0, 'date'])
        else:
            # 参照库无该票文件（且 m15 已有数据）→ 不巡：无 volume 参照会把停牌日误判为缺口
            # （如 v2 库不含的个别票；其缺口巡检由相关参照库存在的管线负责）
            return None
        lo = self._start_of(code) or min(dates)
        i0 = bisect.bisect_left(cal, lo)
        miss = [d for d in cal[i0:i_end + 1] if d not in dates and (expect is None or d in expect)]
        if not miss:
            return None
        if not self.force and code in self.unavail:
            skip = self.unavail[code]
            miss = [d for d in miss if d not in skip]
            if not miss:
                return None
        if miss[0] < cal[max(0, i_end - SINA_WIN)]:
            return ('bs', miss[0], miss[-1], set(miss))   # 老缺口超出新浪窗 → baostock 只拉缺口区间
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


def main():
    ap = argparse.ArgumentParser(description='共用 m15 更新器（新浪+baostock；两套日K管线共用同一份 m15 库）')
    ap.add_argument('--codes', default='', help='逗号分隔，如 600519,sz.000001（缺省=全市场沪深）')
    ap.add_argument('--daily-ref', default='tushare', choices=['tushare', 'mix'],
                    help='缺口参照日线库：tushare（daily_kline_ts，默认）/ mix（daily_kline_v2）')
    ap.add_argument('--force', action='store_true', help='忽略"源不可得"跳过记录，强制重试')
    a = ap.parse_args()
    try:
        codes = norm_codes(a.codes) if a.codes else None
        M15Updater(force=a.force, daily_ref=a.daily_ref).update_m15(codes)
    except Exception as e:
        logger.error(f'fatal: {type(e).__name__}: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
