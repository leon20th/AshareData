import numpy as np
from AshareData.utils.exchanges_utils import a_open, stock_utils

def get_code_price_limit(codes, isST, isNew):
    return [stock_utils.get_price_limit(code, is_st, is_new) for code, is_st, is_new in zip(codes, isST, isNew)]

def get_trade_date_idx(dates):
    return [a_open.get_trade_date_idx(d) for d in dates]

def get_code_idx(codes):
    return [stock_utils.get_code_idx(code) for code in codes]

def isNew(date_idx):
    # 算差分 如果date_idx[i+1] - date_idx[i] > 30 or i == 0, then isNew[i:i+5] = 1
    n = len(date_idx)
    flags = np.zeros(n, dtype=int)
    arr = np.array(date_idx)
    diff = np.diff(arr)
    gap_idx = np.where(diff > 30)[0] + 1  # 新数据从 gap 后一位开始
    offsets = np.arange(5)
    idx = gap_idx[:, None] + offsets[None, :]  # (n_gaps, 5)
    idx = idx[idx < n]
    flags[0] = 1
    flags[idx] = 1
    return flags.tolist()

def calculate_moving_average(data, window):
    cumsum = np.cumsum(data, axis=0)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    result = cumsum[window - 1:] / window
    pad = np.full((window - 1,) + data.shape[1:], np.nan)
    return np.concatenate([pad, result], axis=0)

def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    def ema(arr, span):
        alpha = 2 / (span + 1)
        out = np.empty_like(arr, dtype=float)
        out[0] = arr[0]
        for i in range(1, len(arr)):
            out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
        return out
    short_ema = ema(data, short_window)
    long_ema = ema(data, long_window)
    macd_line = short_ema - long_ema
    signal_line = ema(macd_line, signal_window)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_rsi(data, window=14):
    delta = np.diff(data, axis=0)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.empty_like(data, dtype=float)
    avg_loss = np.empty_like(data, dtype=float)
    avg_gain[:window] = np.nan
    avg_loss[:window] = np.nan
    avg_gain[window] = np.mean(gain[:window], axis=0)
    avg_loss[window] = np.mean(loss[:window], axis=0)
    for i in range(window + 1, len(data)):
        avg_gain[i] = (avg_gain[i-1] * (window - 1) + gain[i-1]) / window
        avg_loss[i] = (avg_loss[i-1] * (window - 1) + loss[i-1]) / window
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi = 100 - 100 / (1 + rs)
    rsi[:window] = np.nan
    return rsi

def calculate_bollinger_bands(data, window=20, num_std_dev=2):
    middle = calculate_moving_average(data, window)
    pad = np.full((window - 1,) + data.shape[1:], np.nan)
    std = np.array([np.std(data[i:i+window], axis=0) for i in range(len(data) - window + 1)])
    std = np.concatenate([pad, std], axis=0)
    upper = middle + num_std_dev * std
    lower = middle - num_std_dev * std
    return upper, middle, lower

def calculate_kdj(high, low, close, n=9, m1=3, m2=3):
    # high/low/close: (N, 1)
    N = len(close)
    rsv = np.full((N, 1), np.nan)
    for i in range(n - 1, N):
        hh = np.max(high[i - n + 1:i + 1])
        ll = np.min(low[i - n + 1:i + 1])
        rsv[i, 0] = (close[i, 0] - ll) / (hh - ll) * 100 if hh != ll else 50
    k = np.full((N, 1), np.nan)
    d = np.full((N, 1), np.nan)
    k[n - 1] = 50
    d[n - 1] = 50
    for i in range(n, N):
        k[i] = (m1 - 1) / m1 * k[i - 1] + 1 / m1 * rsv[i]
        d[i] = (m2 - 1) / m2 * d[i - 1] + 1 / m2 * k[i]
    j = 3 * k - 2 * d
    return k, d, j