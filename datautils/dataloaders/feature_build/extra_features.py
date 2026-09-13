
# ======================== 涨跌停特征构建 ========================

import os, re, json
import pandas as pd
import numpy as np
from datetime import datetime
from env_setting import ROOT

from AshareData.datautils.dataloaders.feature_build.feature_utils import get_code_idx, get_trade_date_idx

UPDOWN_COLUMNS = [
    'date',              # 交易日 YYYYMMDD
    'trade_date_idx',    # 交易日索引
    'code',              # 股票代码 (6位)
    'code_idx',          # 代码索引
    'is_limit_up',       # 1=涨停 -1=跌停
    'first_limit_min',   # 首次涨/跌停 开盘后分钟(不含午休)
    'last_limit_min',    # 最终涨/跌停 开盘后分钟(不含午休)
    'consecutive_days',  # 连续涨/跌停天数
    'seal_volume_wan',   # 封单量(万手)
    'seal_value_wan',    # 封单额(万元)
    'seal_pct',          # 封成比(%)
    'seal_flow_pct',     # 封流比(%)
    'open_count',        # 开板次数
    'days',              # 几天几板 天数 (跌停板填 0)
    'boards',            # 几天几板 板数 (跌停板填 0)
    'reason',            # 涨停原因类别 / 跌停原因类型
]

UPLIMIT_DIR = f'{ROOT}/AshareData/dataset/scrap_data/zhangtingban'
DOWNLIMIT_DIR = f'{ROOT}/AshareData/dataset/scrap_data/dietingban'

UPDOWN_PARQUET = f'{ROOT}/AshareData/dataset/built_data/updown_limit.parquet'
os.makedirs(os.path.dirname(UPDOWN_PARQUET), exist_ok=True)


def _read_single_updown_file(filepath: str, is_limit_up: int) -> pd.DataFrame:
    """读取单个涨/跌停 xlsx，返回标准化 DataFrame。is_limit_up: 1=涨停, -1=跌停"""
    df = pd.read_excel(filepath, dtype=str)

    col_map = {
        '股票代码': 'code',
        '首次涨停时间': 'first_limit_time', '最终涨停时间': 'last_limit_time',
        '首次跌停时间': 'first_limit_time', '最终跌停时间': 'last_limit_time',
        '连续涨停天数(天)': 'consecutive_days', '连续跌停天数(天)': 'consecutive_days',
        '涨停封单量(股)': 'seal_volume', '跌停封单量(股)': 'seal_volume',
        '涨停封单额(元)': 'seal_value',  '跌停封单额(元)': 'seal_value',
        '涨停封成比(%)': 'seal_pct',     '跌停封成比(%)': 'seal_pct',
        '涨停封流比(%)': 'seal_flow_pct','跌停封流比(%)': 'seal_flow_pct',
        '涨停开板次数(次)': 'open_count', '跌停开板次数(次)': 'open_count',
        '几天几板': 'days_boards',
        '涨停原因类别': 'reason', '跌停原因类型': 'reason',
    }
    df = df.rename(columns=col_map)

    # HH:MM:SS → 开盘后分钟(不含午休 11:30~13:00)
    def _t2min(t):
        try:
            dt = datetime.strptime(str(t).strip(), '%H:%M:%S')
        except ValueError:
            return np.nan
        m = dt.hour * 60 + dt.minute + dt.second / 60 - 570
        if   m < 0:    return np.nan   # 9:30 前
        if   m <= 120: return m        # 早盘 9:30~11:30 → 0~120
        if   m < 210:  return np.nan   # 午休 11:30~13:00
        if   m <= 330: return m - 90   # 午盘 13:00~15:00 → 120~240
        return np.nan                  # 15:00 后

    # '6,832.12万'→68321200, '3.27亿'→327000000, '218'→218
    _n = lambda s: float(s.replace(',', '').replace('亿', 'e8').replace('万', 'e4')) if isinstance(s, str) and s.strip() not in ('', '--') else np.nan

    out = pd.DataFrame()
    out['code'] = df['code'].apply(lambda v: str(v).strip().split('.')[0].zfill(6)) if 'code' in df.columns else ''
    out['is_limit_up'] = int(is_limit_up)
    out['first_limit_min'] = df['first_limit_time'].apply(_t2min) if 'first_limit_time' in df.columns else np.nan
    out['last_limit_min'] = df['last_limit_time'].apply(_t2min) if 'last_limit_time' in df.columns else np.nan
    out['consecutive_days'] = pd.to_numeric(df.get('consecutive_days'), errors='coerce').fillna(0).astype(int)
    out['seal_volume_wan'] = df['seal_volume'].apply(lambda v: _n(v) / 1e6) if 'seal_volume' in df.columns else np.nan  # 股→万手
    out['seal_value_wan'] = df['seal_value'].apply(lambda v: _n(v) / 1e4) if 'seal_value' in df.columns else np.nan    # 元→万元
    out['seal_pct'] = pd.to_numeric(df.get('seal_pct'), errors='coerce')
    out['seal_flow_pct'] = pd.to_numeric(df.get('seal_flow_pct'), errors='coerce')
    out['open_count'] = pd.to_numeric(df.get('open_count'), errors='coerce').fillna(0).astype(int)

    # 几天几板: '3天3板'→(3,3), '首板涨停'→(1,1), 缺失→(0,0)
    def _db(v):
        s = str(v).strip()
        if '首板' in s: return 1, 1
        m = re.match(r'(\d+)\D+(\d+)', s)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    if 'days_boards' in df.columns:
        out['days'] = df['days_boards'].apply(lambda v: _db(v)[0])
        out['boards'] = df['days_boards'].apply(lambda v: _db(v)[1])
    else:
        out['days'], out['boards'] = 0, 0

    out['reason'] = df['reason'].fillna('') if 'reason' in df.columns else ''
    return out


def get_updown_limit_feature(update: bool = False, force_rebuild: bool = False) -> pd.DataFrame:
    """构建涨跌停历史特征(增量)。已有 parquet 只处理新文件，force_rebuild 全量重建。"""
    existing = pd.read_parquet(UPDOWN_PARQUET) if not force_rebuild and os.path.exists(UPDOWN_PARQUET) else None
    if not update:
        return existing
    latest_up = existing.loc[existing.is_limit_up == 1, 'date'].max() if existing is not None and len(existing) else '00000000'
    latest_dn = existing.loc[existing.is_limit_up == -1, 'date'].max() if existing is not None and len(existing) else '00000000'
    latest_up = latest_up if pd.notna(latest_up) else '00000000'
    latest_dn = latest_dn if pd.notna(latest_dn) else '00000000'

    new_up = [f for f in sorted(os.listdir(UPLIMIT_DIR)) if f.endswith('.xlsx') and f[:8] > latest_up]
    new_dn = [f for f in sorted(os.listdir(DOWNLIMIT_DIR)) if f.endswith('.xlsx') and f[:8] > latest_dn]
    if not new_up and not new_dn:
        return existing

    new_frames = [_read_single_updown_file(f'{UPLIMIT_DIR}/{f}', 1).assign(date=f[:8]) for f in new_up] + \
                 [_read_single_updown_file(f'{DOWNLIMIT_DIR}/{f}', -1).assign(date=f[:8]) for f in new_dn]

    combined = pd.concat([existing, *new_frames], ignore_index=True) if existing is not None else pd.concat(new_frames, ignore_index=True)
    combined = (combined.drop_duplicates(['date', 'code', 'is_limit_up'], keep='last')
                .sort_values(['date', 'code', 'is_limit_up']).reset_index(drop=True))
    if 'trade_date_idx' not in combined.columns or combined['trade_date_idx'].isna().any():
        combined['trade_date_idx'] = get_trade_date_idx(combined['date'].astype(str))
    if 'code_idx' not in combined.columns or combined['code_idx'].isna().any():
        combined['code_idx'] = get_code_idx(combined['code'].astype(str))
    combined = combined[UPDOWN_COLUMNS]
    combined.to_parquet(UPDOWN_PARQUET, index=False)
    print(f'[updown] +{len(new_up)}涨+{len(new_dn)}跌, {len(combined)}行, 最新涨{combined.loc[combined.is_limit_up==1,"date"].max()} 跌{combined.loc[combined.is_limit_up==-1,"date"].max()}')
    return combined


MARKET_PARQUET = f'{ROOT}/AshareData/dataset/built_data/market_daily_stats.parquet'
MARKET_COLUMNS = ['date', 'trade_date_idx', 'avg_pctChg', 'amount_sum', 'zhangting_count', 'dieting_count',
                  'up_count', 'down_count', 'flat_count', 'stock_count']
os.makedirs(os.path.dirname(MARKET_PARQUET), exist_ok=True)


def build_market_features(update: bool = False, force_rebuild: bool = False) -> pd.DataFrame:
    """每天全市场统计: 平均涨幅、成交额、涨跌平家数、涨跌停家数、股票总数。"""
    existing = pd.read_parquet(MARKET_PARQUET) if not force_rebuild and os.path.exists(MARKET_PARQUET) else None
    if not update and not force_rebuild:
        return existing
    latest = existing['date'].max() if existing is not None and len(existing) else '00000000'

    from AshareData.datautils.dataloaders.feature_build.feature_calc import read_kline
    all_kline = read_kline('daily', usecols=['date', 'code', 'pctChg', 'amount'])
    all_kline = all_kline[all_kline.date > latest]

    if all_kline.empty:
        return existing

    all_new = all_kline.drop_duplicates(['date', 'code'], keep='last')

    daily = all_new.groupby('date').agg(
        avg_pctChg=('pctChg', 'mean'),
        amount_sum=('amount', 'sum'),
        up_count=('pctChg', lambda x: (x > 0).sum()),
        down_count=('pctChg', lambda x: (x < 0).sum()),
        flat_count=('pctChg', lambda x: (x == 0).sum()),
        stock_count=('pctChg', 'count'),
    ).reset_index()

    updown = get_updown_limit_feature(update=(update or force_rebuild))
    if updown is not None:
        zt = updown[updown.is_limit_up == 1].groupby('date').size().rename('zhangting_count')
        dt = updown[updown.is_limit_up == -1].groupby('date').size().rename('dieting_count')
        daily = daily.join(zt, on='date').join(dt, on='date')
        daily[['zhangting_count', 'dieting_count']] = daily[['zhangting_count', 'dieting_count']].fillna(0).astype(int)

    combined = pd.concat([existing, daily], ignore_index=True) if existing is not None else daily
    combined = (combined.sort_values('date').drop_duplicates('date', keep='last')
                .reset_index(drop=True))
    if 'trade_date_idx' not in combined.columns or combined['trade_date_idx'].isna().any():
        combined['trade_date_idx'] = get_trade_date_idx(combined['date'].astype(str))
    combined = combined[MARKET_COLUMNS]
    combined.to_parquet(MARKET_PARQUET, index=False)
    print(f'[market] +{len(daily)}天, {len(combined)}行, 最新{combined.date.max()}')
    return combined


PCT_RANK_PARQUET = f'{ROOT}/AshareData/dataset/built_data/pct_cross_rank.parquet'
PCT_RANK_COLUMNS = ['trade_date_idx', 'code_idx', 'pct_rank']


def build_pct_cross_rank(update: bool = False, force_rebuild: bool = False) -> pd.DataFrame:
    """个股当日涨幅的横截面分位排名（剔除 ST/新股，按 price_limit 归一后排名）。

    rel = pctChg / price_limit —— 源 base_feature 的 price_limit 在 isST=0 & isNew=0
    行即纯板幅（主板0.1 / 创业·科创0.2 / 北交0.3），故 rel 跨板块可比。当日全体可比
    股票 rel 升序分位 → [0,1]，天生逐日无基准率漂移（与温度缩放失败结论同构）。
    剔除 ST/新股后剩余样本再排名；被剔除者不进表（loader lookup 落空 → 0）。

    横截面 rank 按天独立，可增量（只算新 trade_date_idx）。
    """
    existing = pd.read_parquet(PCT_RANK_PARQUET) if not force_rebuild and os.path.exists(PCT_RANK_PARQUET) else None
    if not update and not force_rebuild:
        return existing
    latest = int(existing['trade_date_idx'].max()) if existing is not None and len(existing) else -1

    import glob as _g
    import polars as pl
    files = sorted(_g.glob(f'{ROOT}/AshareData/dataset/built_data/base_feature/*.parquet'))
    cols = ['trade_date_idx', 'code_idx', 'pctChg', 'price_limit', 'isST', 'isNew']
    lazy = pl.scan_parquet(files).select(cols).filter(
        (pl.col('isST') == 0) & (pl.col('isNew') == 0)
        & (pl.col('price_limit') > 0) & pl.col('pctChg').is_not_nan()
    )
    if latest >= 0:
        lazy = lazy.filter(pl.col('trade_date_idx') > latest)
    d = lazy.with_columns(rel=(pl.col('pctChg') / pl.col('price_limit'))).collect().to_pandas()
    if d.empty:
        return existing

    d['pct_rank'] = d.groupby('trade_date_idx')['rel'].rank(pct=True).astype('float32')
    daily = d[['trade_date_idx', 'code_idx', 'pct_rank']].copy()
    combined = pd.concat([existing, daily], ignore_index=True) if existing is not None else daily
    combined = (combined.sort_values(['trade_date_idx', 'code_idx'])
                .drop_duplicates(['trade_date_idx', 'code_idx'], keep='last').reset_index(drop=True))
    combined['trade_date_idx'] = combined['trade_date_idx'].astype('int64')
    combined['code_idx'] = combined['code_idx'].astype('int64')
    combined = combined[PCT_RANK_COLUMNS]
    combined.to_parquet(PCT_RANK_PARQUET, index=False)
    print(f'[pct_rank] +{int(d.trade_date_idx.nunique())}天, {len(combined)}行, 最新tdi{int(combined.trade_date_idx.max())}')
    return combined


# ======================== 龙虎榜特征构建 ========================
# 独立重实现：直接解析 PPO 本地 scrap_data/longhu/*.json，不依赖 RegimeFramework 老管线。
# 列 = 探针定案 8 列（tests/probe_longhu_next_ret.py，联合控制 ret/turn/uplimit 后独立存活），
# 每族 1 根：方向 net_buy_ratio；结构 buy/sell_hhi；席位 buy_lhasa_ratio/famous_net_bias；
# 原因 reason_turnover/reason_amplitude；存在位 is_longhu。
# 老 builder 死列不进（change/turnover_percent_norm 解析 bug、top5_ratio≡1、恒等共线族、
# north/inst 席位 t<3.7 且事件内非零率≤44%）。
# ⚠️ 同 (code,date) 多条 reason 记录 → 保留文件序最后一条（与老管线 dict 覆盖语义一致，
#    保证与已探针验证的事件表逐值可比）。

LONGHU_DIR = f'{ROOT}/AshareData/dataset/scrap_data/longhu'
LONGHU_PARQUET = f'{ROOT}/AshareData/dataset/built_data/longhu_features.parquet'
os.makedirs(os.path.dirname(LONGHU_PARQUET), exist_ok=True)

LONGHU_COLUMNS = ['date', 'trade_date_idx', 'code', 'code_idx', 'is_longhu',
                  'net_buy_ratio', 'buy_hhi', 'sell_hhi', 'buy_lhasa_ratio',
                  'famous_net_bias', 'reason_turnover', 'reason_amplitude']

def _parse_wan(value) -> float:
    """金额 → 万元。兼容 '4834.64万元' / '4.12亿' / '8462.83万' / 纯数字元 / 带负号。"""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(',', '')
    if not text:
        return 0.0
    sign = -1.0 if text.startswith('-') else 1.0
    m = re.search(r'(\d+(?:\.\d+)?)', text)
    if not m:
        return 0.0
    num = float(m.group(1))
    if '亿' in text:
        return sign * num * 1e4
    if '万' in text:
        return sign * num
    return sign * num / 1e4 if '元' in text else sign * num


def _seat_profile(rows: list, amount_key: str) -> dict:
    """席位列表 → 集中度(hhi) + 席位占比(lhasa=名称含拉萨 / famous=tags 含知名·一线游资)。"""
    n = len(rows)
    vals = [_parse_wan(r.get(amount_key)) for r in rows]
    total = max(sum(vals), 1e-6)
    ratios = [v / total for v in vals]
    tags = [set(map(str, r.get('tags') or [])) for r in rows]
    names = [str(r.get('name', '')) for r in rows]
    d = max(1.0, float(n))
    return dict(
        hhi=sum(r * r for r in ratios),
        lhasa=sum('\u62c9\u8428' in nm for nm in names) / d,
        famous=sum(bool(t & {'\u77e5\u540d\u6e38\u8d44', '\u4e00\u7ebf\u6e38\u8d44'}) for t in tags) / d,
    )


def _build_longhu_event(stock: dict) -> list:
    """单条龙虎榜记录 → LONGHU_COLUMNS 数值行（与 LONGHU_FEATURES 顺序一致）。"""
    det = stock.get('detail') or {}
    reason = str(det.get('reason', ''))
    ts = max(_parse_wan(det.get('turnover_amount') or stock.get('turnover')), 1e-6)
    net = _parse_wan(det.get('net_amount') or stock.get('net_buy'))
    br = ((det.get('buy_top5_departments') or {}).get('rows') or [])
    sr = ((det.get('sell_top5_departments') or {}).get('rows') or [])
    bp, sp = _seat_profile(br, 'buy_amount_wan'), _seat_profile(sr, 'sell_amount_wan')
    return [1.0,
            float(np.clip(net / ts, -2, 2)),
            float(np.clip(bp['hhi'], 0, 1)), float(np.clip(sp['hhi'], 0, 1)),
            float(np.clip(bp['lhasa'], 0, 1)),
            float(np.clip(bp['famous'] - sp['famous'], -1, 1)),
            1.0 if '\u6362\u624b\u7387' in reason else 0.0,
            1.0 if '\u632f\u5e45' in reason else 0.0]


def _read_longhu_file(filepath: str) -> pd.DataFrame:
    payload = json.load(open(filepath, 'r', encoding='utf-8'))
    date = str(payload.get('date', os.path.basename(filepath)[:-5])).replace('-', '')[:8]
    stocks = payload.get('stocks', [])
    rows = [{'code': str(s.get('code', '')).strip().zfill(6), 'date': date,
             **dict(zip(LONGHU_COLUMNS[4:], _build_longhu_event(s)))}
            for s in stocks if str(s.get('code', '')).strip()]
    out = pd.DataFrame(rows)
    # 同 (code,date) 多 reason → 保留文件序最后一条（老管线覆盖语义）
    return out.drop_duplicates(['date', 'code'], keep='last')


def get_longhu_feature(update: bool = False, force_rebuild: bool = False) -> pd.DataFrame:
    """构建龙虎榜历史特征(增量)。已有 parquet 只处理新文件，force_rebuild 全量重建。"""
    existing = pd.read_parquet(LONGHU_PARQUET) if not force_rebuild and os.path.exists(LONGHU_PARQUET) else None
    if not update:
        return existing
    latest = existing['date'].max() if existing is not None and len(existing) else '00000000'
    new_files = [f for f in sorted(os.listdir(LONGHU_DIR)) if f.endswith('.json') and f[:10].replace('-', '') > latest]
    if not new_files:
        return existing

    new_frames = [_read_longhu_file(f'{LONGHU_DIR}/{f}') for f in new_files]
    combined = pd.concat([existing, *new_frames], ignore_index=True) if existing is not None else pd.concat(new_frames, ignore_index=True)
    combined = (combined.drop_duplicates(['date', 'code'], keep='last')
                .sort_values(['date', 'code']).reset_index(drop=True))
    if 'trade_date_idx' not in combined.columns or combined['trade_date_idx'].isna().any():
        combined['trade_date_idx'] = get_trade_date_idx(combined['date'].astype(str))
    if 'code_idx' not in combined.columns or combined['code_idx'].isna().any():
        combined['code_idx'] = get_code_idx(combined['code'].astype(str))
    bad = int((combined['code_idx'] < 0).sum())
    if bad:
        print(f'[longhu] 丢弃无法定位 code_idx 的 {bad} 行（B股/退市等）')
        combined = combined[combined['code_idx'] >= 0]
    combined = combined[LONGHU_COLUMNS]
    combined.to_parquet(LONGHU_PARQUET, index=False)
    print(f'[longhu] +{len(new_files)}天, {len(combined)}行, 最新{combined.date.max()}')
    return combined


if __name__ == '__main__':
    _ = get_updown_limit_feature(update=True, force_rebuild=True)
    data = build_market_features(force_rebuild=True)
    print(data)
    rank = build_pct_cross_rank(force_rebuild=True)
    print(rank)
    lh = get_longhu_feature(update=True, force_rebuild=True)
    print(lh)