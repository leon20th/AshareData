# -*- coding: utf-8 -*-
"""kline_v2_reader —— v2 日线读出口（读时复权）。

v2 主表（daily_kline_v2/*.csv）存**未复权原值**；本模块提供与旧库等价的**前复权**读数：

    qfq(t) = raw(t) × ∏_{e > t} f_e

- 因子边界与数值**全部由 v2 自带的官方除权参考价推导**：真实行（tradestatus=1）相邻过渡处
  preclose[r] ≠ close[前一真实行] 即为一次除权（复权因子 f = preclose/前收；非除权日 preclose 严格等于昨收）
- 停牌期间拟除权的股票，因子在**复牌行**一次性生效（与交易所/旧库口径一致）
- 锚点 = 序列末日（末日价格=原值，与旧库约定一致）
- 停牌占位行的价格列=carry（原值），处于同一因子段内，行间仍严格连续

API:
    read_daily(code, adjust='qfq')  -> 13 列 DataFrame（date 为 'YYYY-MM-DD' 字符串，与旧库 CSV 同构）
    read_daily(code, adjust='raw')  -> 原值直读
    close_rows(code)                -> [(date, qfq_close), ...]（供逐行裁剪的消费方）
"""
import os

import numpy as np
import pandas as pd

from AshareData.datautils.kline_scripts.quick_kline import V2_DIR

PRICE_COLS = ['open', 'high', 'low', 'close', 'preclose']


def _suffix_factors(dates, closes, precloses, trade):
    """F[i] = ∏_{e > dates[i]} f_e；e 取真实行过渡处的因子边界（停牌期间推迟到复牌行）。"""
    dates = np.asarray(dates)
    closes = np.asarray(closes, dtype=float)
    precloses = np.asarray(precloses, dtype=float)
    real = np.asarray(trade, dtype=float) == 1
    prev_real = np.full(len(dates), np.nan)          # 每个真实行取前一真实行收盘
    idx = np.flatnonzero(real)
    if len(idx) > 1:
        prev_real[idx[1:]] = closes[idx[:-1]]
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = precloses / prev_real
    bd = real & np.isfinite(ratio) & (precloses > 0) & (prev_real > 0) & (np.abs(ratio - 1.0) > 1e-10)
    if not bd.any():
        return np.ones(len(dates))
    f = ratio[bd]
    suffix = np.concatenate([np.cumprod(f[::-1])[::-1], [1.0]])
    k = np.searchsorted(dates[bd], dates, side='right')
    return suffix[k]


def read_daily(code, adjust='qfq'):
    """读单只 v2 日线；adjust='qfq' 前复权（默认，与旧库等价）/ 'raw' 原值。"""
    p = os.path.join(V2_DIR, f'{code}.csv')
    df = pd.read_csv(p, dtype={'date': str})
    df = df.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    if adjust == 'raw' or df.empty:
        return df
    dates = df['date'].to_numpy()
    closes = pd.to_numeric(df['close'], errors='coerce').to_numpy()
    precloses = pd.to_numeric(df['preclose'], errors='coerce').to_numpy()
    trade = pd.to_numeric(df['tradestatus'], errors='coerce').to_numpy()
    F = _suffix_factors(dates, closes, precloses, trade)
    for c in PRICE_COLS:
        df[c] = pd.to_numeric(df[c], errors='coerce') * F
    return df


def close_rows(code, tail_bytes=16384):
    """[(date, qfq_close), ...]（轻量尾窗读：只取 date/close/preclose/tradestatus，窗口内重建因子）。

    正确性：因子只影响“更早”的行（边界只作用于日期早于它的行），窗口内任意行所需的
    边界部在窗口内；仅当窗口首行恰为新边界（其基被截断）时跳过该边界——它不影响窗口内任何行。
    默认窗口 16KB（≈260 行 ≈ 1 年），供 Lushan 表格逐行消费（全市场 ~5000 只 / ~0.5s）。
    """
    p = os.path.join(V2_DIR, f'{code}.csv')
    with open(p, 'rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - tail_bytes))
        text = f.read().decode('utf-8', errors='ignore')
    lines = text.splitlines()[1:]        # 首行 = 表头（全读）或被截断行（尾读）
    dates, closes, precloses, trade = [], [], [], []
    for line in lines:
        parts = line.split(',')
        if len(parts) < 11:
            continue
        try:
            closes.append(float(parts[5]))
            precloses.append(float(parts[6]) if parts[6].strip() else float('nan'))
            trade.append(float(parts[10]) if parts[10].strip() else float('nan'))
        except ValueError:
            continue
        dates.append(parts[0])
    if not dates:
        return []
    closes = np.asarray(closes)
    F = _suffix_factors(dates, closes, precloses, trade)
    return list(zip(dates, (closes * F).tolist()))
