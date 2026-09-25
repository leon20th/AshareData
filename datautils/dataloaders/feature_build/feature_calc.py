import os
import pandas as pd
from AshareData.paths import BASE_FEATURE_DIR, DAILY_KLINE_V2_DIR, M15_KLINE_DIR
from AshareData.datautils.dataloaders.feature_build.feature_utils import *
from AshareData.utils.kline_data_utils.kline_v2_reader import read_daily

# daily 已切 v2（读时复权 adapter）；KLINE_DIRS['daily'] 仅用于存在性检查/目录遍历
KLINE_DIRS = {
    'daily': DAILY_KLINE_V2_DIR,
    'm15':   M15_KLINE_DIR,
}
FeatDIR = BASE_FEATURE_DIR
_PRICE_COLS = ('open', 'high', 'low', 'close', 'preclose')

# 固定 parquet schema：所有数值列统一为 float64，避免 NaN 导致 int/float 不一致
PARQUET_NUM_SCHEMA = {
    'volume': 'float64', 'tradestatus': 'float64', 'isST': 'float64',
    'trade_date_idx': 'float64', 'code_idx': 'float64', 'isNew': 'float64',
}

def read_kline(kline_type='daily', code=None, end_date=None, **kwargs):
    """读取 kline。daily=v2 读时复权（默认 qfq，价格列随除权连续；usecols 不含价格列时直读原值加速）；
    m15=直读目录 CSV（kwargs 透传给 read_csv）。指定 code 读单只，否则读全目录。end_date 截断到该日期。"""
    usecols = kwargs.get('usecols')
    need_price = not usecols or any(c in usecols for c in _PRICE_COLS)

    def _read(code_):
        if kline_type == 'daily':
            df = read_daily(code_, 'qfq' if need_price else 'raw')
            return df[[c for c in usecols if c in df.columns]] if usecols else df
        df = pd.read_csv(f'{KLINE_DIRS[kline_type]}/{code_}.csv', **kwargs)
        # 上游脏数据：time 列可能是 YYYYMMDDHHMMSS 格式（如 20260624094500000），修正为 HHMM
        time_str = df['time'].astype(str)
        bad = time_str.str.len() > 4
        if bad.any():
            df.loc[bad, 'time'] = time_str[bad].str[8:12].astype(int)
        return df

    def _prep(df):
        df['date'] = df['date'].str.replace('-', '')
        if end_date:
            df = df[df.date <= end_date]
        dedup_cols = ['date', 'code'] if kline_type == 'daily' else ['date', 'code', 'time']
        return df.drop_duplicates(dedup_cols, keep='last')

    if code:
        return _prep(_read(code))
    d = KLINE_DIRS[kline_type]
    files = [f for f in sorted(os.listdir(d)) if f.endswith('.csv') and not f.startswith('_')]
    return pd.concat(_prep(_read(f[:-4])) for f in files)

def build_daily_kline_feature(code, data, latest_date=None):
    """计算 daily kline 特征。latest_date 之后的行才算新数据，内部自动留100行lookback给指标热身。"""
    if latest_date is not None:
        calc_mask = data['date'].values > latest_date
        if not calc_mask.any():
            return data.iloc[0:0]
        calc_start = int(np.argmax(calc_mask))
        lookback = 100
        win_start = max(0, calc_start - lookback)
    else:
        calc_start = 0
        win_start = 0

    window = data.iloc[win_start:].copy()
    close = window['close'].values.astype(float).reshape(-1, 1)

    # 索引特征
    window['trade_date_idx'] = get_trade_date_idx(window['date'].tolist())
    window['code_idx'] = get_code_idx([code])[0]
    window['isNew'] = isNew(window['trade_date_idx'].values.tolist())
    window['price_limit'] = get_code_price_limit(
        [code] * len(window), window['isST'].values.astype(int).tolist(), window['isNew'].tolist()
    )

    # MA
    for w in [5, 10, 20, 30, 60]:
        window[f'ma{w}'] = calculate_moving_average(close, w)[:, 0]

    # MACD
    macd, signal, hist = calculate_macd(close)
    window['macd'] = macd[:, 0]
    window['macd_signal'] = signal[:, 0]
    window['macd_hist'] = hist[:, 0]

    # RSI
    window['rsi14'] = calculate_rsi(close, 14)[:, 0]

    # Bollinger
    upper, mid, lower = calculate_bollinger_bands(close, 20, 2)
    window['boll_upper'] = upper[:, 0]
    window['boll_mid'] = mid[:, 0]
    window['boll_lower'] = lower[:, 0]

    # KDJ
    high = window['high'].values.astype(float).reshape(-1, 1)
    low = window['low'].values.astype(float).reshape(-1, 1)
    k, d, j = calculate_kdj(high, low, close)
    window['kdj_k'] = k[:, 0]
    window['kdj_d'] = d[:, 0]
    window['kdj_j'] = j[:, 0]

    # 只返回 latest_date 之后的行
    if latest_date is not None:
        return window.iloc[calc_start - win_start:]
    return window


def build_m15_kline_feature(code, data, latest_date=None):
    """计算 m15 kline 特征。latest_date 之后的行才算新数据，内部留100行lookback。"""
    if latest_date is not None:
        calc_mask = data['date'].values > latest_date
        if not calc_mask.any():
            return data.iloc[0:0]
        calc_start = int(np.argmax(calc_mask))
        lookback = 100
        win_start = max(0, calc_start - lookback)
    else:
        calc_start = 0
        win_start = 0

    window = data.iloc[win_start:].copy()
    close = window['close'].values.astype(float).reshape(-1, 1)
    window['rsi14'] = calculate_rsi(close, 14)[:, 0]
    high = window['high'].values.astype(float).reshape(-1, 1)
    low = window['low'].values.astype(float).reshape(-1, 1)
    k, d, j = calculate_kdj(high, low, close)
    window['kdj_k'] = k[:, 0]
    window['kdj_d'] = d[:, 0]
    window['kdj_j'] = j[:, 0]

    if latest_date is not None:
        return window.iloc[calc_start - win_start:]
    return window


def build_merged_feature(code, rebuild=False):
    """合并 daily + m15 为单个 DataFrame，每天一行，m15 flatten 到列。支持增量更新。"""
    # 读已有 parquet，取 pq_max_date 作为增量起点
    pq_path = f'{FeatDIR}/{code}.parquet'
    old_df = None
    pq_max_date = None
    if os.path.exists(pq_path) and not rebuild:
        old_df = pd.read_parquet(pq_path)
        pq_max_date = old_df['date'].max()

    # 读取全量 kline，feature 函数根据 pq_max_date 内部算窗口
    daily = read_kline('daily', code)
    if daily.empty:
        return old_df
    daily = build_daily_kline_feature(code, daily, latest_date=pq_max_date)

    m15 = read_kline('m15', code)
    if not m15.empty:
        m15 = build_m15_kline_feature(code, m15, latest_date=pq_max_date)
        m15_value_cols = [c for c in m15.columns if c not in ('date', 'time', 'code')]
        pivoted = m15.pivot_table(index='date', columns='time', values=m15_value_cols, aggfunc='last')
        pivoted.columns = [f'm15_{int(t):04d}_{f}' for f, t in pivoted.columns]
        daily = daily.merge(pivoted.reset_index(), on='date', how='left')

    # 无新数据直接返回老数据
    if daily.empty:
        return old_df
    # 增量追加：去掉全 NA 列避免 FutureWarning
    if old_df is not None:
        return pd.concat([old_df, daily.dropna(axis=1, how='all')], ignore_index=True)
    return daily
    