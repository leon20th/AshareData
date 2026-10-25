"""三大指数日线 → `loader_npy/index_daily` 键表（按 trade_date_idx，无 code 轴）。

覆盖：上证指数 `000001.SH`、深证成指 `399001.SZ`、创业板指 `399006.SZ`。

数据源：
  1. **富瑶 API**（主源，仅近 5 年）`GET /api/a-share-index/prices/historical`
     —— 返回的 `date_ms` 比真实交易日早 1 个日历日（已量化验证：平移 +1 天后
     与 baostock 重叠段收盘差 ~0.005）。
  2. **baostock**（回填富瑶 5 年窗口之外，2010 起全历史；创业板自 2010-06-01）。
     指数用 `adjustflag='3'`（不复权），字段 `date,open,high,low,close`。

增量纪律：本地表已覆盖的区间不重复请求，只补缺口；`--force` 才全量重建。
两源重叠段逐日交叉校验并打印最大收盘差（>0.01 报警）。

列（float32，价格原值 O(1e3~1e4)，不做归一化——下游自行决定）：
    sh_close, sh_ret1, sz_close, sz_ret1, cyb_close, cyb_ret1
`*_ret1` = 当日收盘 / 上一 A 股交易日收盘 − 1。
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from bisect import bisect_right  # noqa: F401
from datetime import date as _date, timedelta

import numpy as np
import pandas as pd

from AshareData.datautils.dataloaders.feature_build.feature_utils import (
    get_trade_date_idx)
from AshareData.datautils.dataloaders.feature_build.loader_tables import (
    read_key_table_df, save_key_table)
from AshareData.paths import ASHARE_ROOT
from AshareData.utils.exchanges_utils import a_open

TABLE = 'index_daily'
SUB_COLUMNS = ['trade_date_idx', 'sh_close', 'sh_ret1', 'sz_close', 'sz_ret1',
               'cyb_close', 'cyb_ret1']

FUYAO_BASE = 'https://fuyao.aicubes.cn'
FUYAO_KEY_FILE = os.path.join(ASHARE_ROOT, '.keys', '.fuyao_api.json')
FY_CODE = {'sh': '000001.SH', 'sz': '399001.SZ', 'cyb': '399006.SZ'}
BS_CODE = {'sh': 'sh.000001', 'sz': 'sz.399001', 'cyb': 'sz.399006'}
FUYAO_SHIFT_DAYS = 1          # date_ms 比真实交易日早 1 个日历日（实测）
HIST_START = _date(2010, 1, 1)
BS_TIMEOUT = 120              # 单次查询看门狗（连接半开有卡死前科）
BS_FIELDS = 'date,open,high,low,close'
_BS_ON = False                # baostock 登录态


def _fuyao(code: str, d0: _date, d1: _date) -> pd.DataFrame:
    """富瑶指数日线 → DataFrame(index=真实交易日, cols=open/high/low/close)。"""
    key = json.load(open(FUYAO_KEY_FILE, encoding='utf-8'))['行情数据']
    s = int(time.mktime(d0.timetuple()) * 1000)
    e = int(time.mktime(d1.timetuple()) * 1000)
    url = (f'{FUYAO_BASE}/api/a-share-index/prices/historical?thscode={code}'
           f'&interval=1d&start={s}&end={e}')
    req = urllib.request.Request(url, headers={'X-api-key': key})
    with urllib.request.urlopen(req, timeout=40) as r:
        js = json.load(r)
    it = js.get('data', {}).get('item', [])
    if not it:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close'])
    df = pd.DataFrame(it)
    df['date'] = (pd.to_datetime(df['date_ms'], unit='ms').dt.normalize()
                  + pd.Timedelta(FUYAO_SHIFT_DAYS, unit='D'))
    df = df.rename(columns={'open_price': 'open', 'high_price': 'high',
                            'low_price': 'low', 'close_price': 'close'})
    return (df.drop_duplicates('date').set_index('date').sort_index()
            [['open', 'high', 'low', 'close']])


def _bs_call(fn, timeout=BS_TIMEOUT):
    """守护线程里跑 baostock 调用：连接半开卡死时超时放弃，不阻塞整轮。"""
    import queue

    q = queue.Queue(maxsize=1)

    def run():
        try:
            q.put((True, fn()))
        except Exception as e:  # noqa: BLE001
            q.put((False, e))

    threading.Thread(target=run, daemon=True).start()
    try:
        return q.get(timeout=timeout)
    except queue.Empty:
        return False, TimeoutError(f'baostock 调用超时>{timeout}s')


def _bs_login():
    global _BS_ON
    if _BS_ON:
        return
    import baostock as bs
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f'baostock 登录失败: {lg.error_msg}')
    _BS_ON = True


def _baostock(code: str, d0: _date, d1: _date) -> pd.DataFrame:
    """baostock 指数日线（不复权）。空段返回空表，连接异常抛错。"""
    import baostock as bs

    _bs_login()
    ok, rs = _bs_call(lambda: bs.query_history_k_data_plus(
        code, BS_FIELDS, start_date=d0.isoformat(), end_date=d1.isoformat(),
        frequency='d', adjustflag='3'))
    if not ok:
        raise rs
    if rs.error_code != '0':
        raise RuntimeError(f'baostock {code}: {rs.error_msg}')
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    if not rows:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close'])
    df = pd.DataFrame(rows, columns=BS_FIELDS.split(','))
    df['date'] = pd.to_datetime(df['date'])
    for c in ('open', 'high', 'low', 'close'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return (df.dropna(subset=['close']).drop_duplicates('date')
            .set_index('date').sort_index()[['open', 'high', 'low', 'close']])


def build_index_daily(force_rebuild: bool = False) -> pd.DataFrame:
    """构建/更新 `index_daily` 键表。富瑶为主，baostock 回填，重叠段交叉校验。

    增量纪律：先查本地水位。已到当天 → 直接返回（不发请求）；否则只请求
    `[水位-10天, 今天]` 这段（重叠 10 天用于校验，不重拉全史）。
    """
    existing = None if force_rebuild else read_key_table_df(TABLE)
    today = _date.today()
    today_s = today.strftime('%Y%m%d')

    # 预期应有的最新交易日 = 日历里 <= 今天 的最后一天（躲周末/节假日）
    cal = a_open.trade_date_list
    expect_s = cal[bisect_right(cal, today_s) - 1]

    last = None
    if existing is not None and len(existing):
        last_tdi = int(existing['trade_date_idx'].max())
        last_s = cal[last_tdi]            # 键表无 date 列，查日历反解
        if last_s >= expect_s:
            print(f'[index_daily] 本地已到 {last_s}（预期 {expect_s}），'
                  f'无新交易日，跳过', flush=True)
            return existing
        last = pd.Timestamp(last_s)

    # 本次实际要拉取的起止（无本地表则拉全史）
    want_lo = (HIST_START if last is None
               else (last - pd.Timedelta(10, unit='D')).date())
    # 富瑶硬限制：只给近 5 年。留 2 周余量，否则起始日落在窗口外会返回 0 行。
    fy_lo = today - timedelta(days=1800)

    per, src = {}, {}
    for k in ('sh', 'sz', 'cyb'):
        f = _fuyao(FY_CODE[k], max(fy_lo, want_lo), today) \
            if want_lo < today else pd.DataFrame(columns=['open', 'close'])
        b = _baostock(BS_CODE[k], want_lo, today)
        if len(f) and len(b):
            both = f.index.intersection(b.index)
            if len(both):
                err = (f.loc[both, 'close'] - b.loc[both, 'close']).abs().max()
                flag = '' if err < 0.01 else '  ⚠️ 两源不一致'
                print(f'[index_daily] 校验 {k}: 富瑶 vs baostock 重叠 '
                      f'{len(both)} 日, 最大收盘差 {err:.4f}{flag}', flush=True)
        # 富瑶优先，其窗口外（或未拉到）用 baostock 补齐
        keep_b = b[b.index < pd.Timestamp(max(fy_lo, want_lo))]
        parts = [x for x in (keep_b, f) if len(x)]
        per[k] = pd.concat(parts) if parts else pd.DataFrame(
            columns=['open', 'high', 'low', 'close'])
        src[k] = f'富瑶 {len(f)} + baostock {len(keep_b)} → {len(per[k])} 行'
        print(f'[index_daily] {k}: {src[k]}', flush=True)

    all_d = sorted(set().union(*[set(d.index) for d in per.values()]))
    df = pd.DataFrame(index=pd.DatetimeIndex(all_d))
    for k in ('sh', 'sz', 'cyb'):
        df[f'{k}_close'] = (per[k]['close'].reindex(df.index) if len(per[k])
                            else np.nan)

    df = df.reset_index().rename(columns={'index': 'date'})
    df['trade_date_idx'] = get_trade_date_idx(
        df['date'].dt.strftime('%Y%m%d').tolist())
    df = df[df['trade_date_idx'] >= 0].sort_values('trade_date_idx')

    if existing is not None and len(existing):
        old = existing[~existing['trade_date_idx'].isin(df['trade_date_idx'])]
        if len(old):
            df = pd.concat([old, df], ignore_index=True)
            df = df.sort_values('trade_date_idx').drop_duplicates(
                'trade_date_idx', keep='last')

    # date 列一律由日历重建（旧行合并后没有 date）
    df = df.sort_values('trade_date_idx').reset_index(drop=True)
    df['date'] = pd.to_datetime(
        [cal[int(i)] for i in df['trade_date_idx']], format='%Y%m%d')
    for k in ('sh', 'sz', 'cyb'):
        df[f'{k}_ret1'] = df[f'{k}_close'].pct_change(fill_method=None)

    save_key_table(TABLE, df, SUB_COLUMNS, np.nan, source='build_index_daily',
                   extra={'latest_date': df['date'].max().strftime('%Y%m%d'),
                          'sources': {'fuyao': f'>={fy_lo}',
                                      'baostock': f'<{fy_lo}'},
                          'fuyao_date_shift_days': FUYAO_SHIFT_DAYS})
    print(f'[index_daily] 落盘 {len(df)} 行  '
          f'{df["date"].min().date()} → {df["date"].max().date()}', flush=True)
    return df


if __name__ == '__main__':
    import sys
    build_index_daily(force_rebuild='--force' in sys.argv)
