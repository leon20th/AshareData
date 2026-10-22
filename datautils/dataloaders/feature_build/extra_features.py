
# ======================== 涨跌停特征构建 ========================

import os, re, json
import pandas as pd
import numpy as np
from datetime import datetime
from AshareData.paths import BUILT_DATA_DIR, SCRAP_DATA_DIR

from AshareData.datautils.dataloaders.feature_build.feature_utils import get_code_idx, get_trade_date_idx
from AshareData.datautils.dataloaders.feature_build.loader_tables import (
    save_key_table, read_key_table_df, read_meta)
from AshareData.datautils.dataloaders.feature_build.model_ready import (
    MARKET_SUB_COLUMNS, LONGHU_SUB_COLUMNS, UPDOWN_SUB_COLUMNS)

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

UPLIMIT_DIR = f'{SCRAP_DATA_DIR}/zhangtingban'
DOWNLIMIT_DIR = f'{SCRAP_DATA_DIR}/dietingban'

# ---- 源表列名归一化（2026-10-22 起替代手工 col_map）----
# 站点历次改版只做两类动作：改方向前缀（涨停/跌停）、改单位/说明括号（(股)(元)(%)(次)(天)）。
# 归一化后落到**语义词**再查字段表 ⇒ 前缀/单位变化无需改表；语义词变化（罕见）会 miss，
# 而 miss 现在由下面的必需列校验**显式报错**，不再静默产出 NaN。
_RE_UNIT = re.compile(r'[（(][^）)]*[)）]')
_DIR_WORDS = ('涨停', '跌停')

_CANON_FIELD = {
    '股票代码': 'code', '代码': 'code',
    '首次时间': 'first_limit_time', '最终时间': 'last_limit_time',
    '连续天数': 'consecutive_days',
    '封单量': 'seal_volume', '封单额': 'seal_value',
    '封成比': 'seal_pct', '封流比': 'seal_flow_pct',
    '开板次数': 'open_count', '几天几板': 'days_boards',
}
# 已知但**不入模**的列（改版新增此类列不该触发报错）
_CANON_IGNORE = {'', '股票简称', '简称', '名称', '现价', '最新价', '涨跌幅',
                 '明细数据', '类型', '封耗比', '市值', 'a股流通市值'}


def _strip_dir_units(name) -> str:
    """'涨停封成比(%)' / '跌停封成比(%)' / '封成比(%)' → '封成比'。"""
    s = _RE_UNIT.sub('', str(name)).strip()
    for d in _DIR_WORDS:
        s = s.replace(d, '')
    return s


def _canon_field(canon: str):
    """归一列名 → 输出字段；'原因*' 统一走 reason（原因/原因类别/原因类型/原因说明）。"""
    if canon in _CANON_FIELD:
        return _CANON_FIELD[canon]
    if canon.startswith('原因'):
        return 'reason'
    return None

UPDOWN_PARQUET = f'{BUILT_DATA_DIR}/misc/updown_limit.parquet'
os.makedirs(os.path.dirname(UPDOWN_PARQUET), exist_ok=True)


def _read_single_updown_file(filepath: str, is_limit_up: int) -> pd.DataFrame:
    """读取单个涨/跌停 xlsx，返回标准化 DataFrame。is_limit_up: 1=涨停, -1=跌停"""
    df = pd.read_excel(filepath, dtype=str)
    if df.shape[1] == 0:
        # 空表（历史上确有此类文件，如 dietingban/20200212跌停板.xlsx 无任何列）
        # ⇒ 当天该方向无标的，返回空帧（不是改版，不该报错）
        return pd.DataFrame({c: pd.Series(dtype="object") for c in (
            "code", "is_limit_up", "first_limit_min", "last_limit_min",
            "consecutive_days", "seal_volume_wan", "seal_value_wan", "seal_pct",
            "seal_flow_pct", "open_count", "days", "boards", "reason")})

    # ---------- 列名归一化（2026-10-22）----------
    # 站点改版只做两类动作：改方向前缀（涨停/跌停）、改单位/说明括号（(股)/(%)/无）。
    # 2026-09-14 事故：`涨停封成比(%)`→`封成比(%)` 靠手工 col_map 维护，漏了就静默
    # 变 NaN（历史 3.5 万行跌停数据一度被抹掉）。改为**归一化后查字典**：
    #   '涨停封成比(%)' / '跌停封成比(%)' / '封成比(%)' → 归一 '封成比' → seal_pct
    # 未来再改前缀/单位无需动表；改**语义词**（如 封成比→封成率）才会 miss，
    # 而 miss 现在会**显式报错**（见下方校验），不再静默。
    _canon = lambda name: _strip_dir_units(name)
    rename, unknown = {}, []
    for c in df.columns:
        fld = _canon_field(_canon(c))
        if fld:
            rename[c] = fld
        elif _canon(c) not in _CANON_IGNORE:
            unknown.append(c)
    df = df.rename(columns=rename)

    # 必需列校验：涨停文件必须有代码 + 首次涨停时间；跌停文件历史上只有 5 列（无时间），
    # 故对跌停只校验代码。缺列 = 站点又改版 ⇒ 立刻炸掉，不要产出静默 NaN。
    if 'code' not in df.columns:
        raise ValueError(f"[updown] {filepath} 找不到『股票代码』列，实际列={list(df.columns)}")
    if is_limit_up == 1 and 'first_limit_time' not in df.columns:
        raise ValueError(
            f"[updown] {filepath} 涨停文件缺『首次涨停时间』（站点改版？）"
            f"实际列={list(df.columns)} 未识别列={unknown}")

    # 严格模式：**每一列都必须被归类**（映射 或 显式忽略），否则报错。
    # 历史 3,249 个文件实测 0 个未识别列 ⇒ 只有"站点再改版"才会触发。
    # 这是把"静默变 NaN"换成"构建期就炸"的那道闸门（2026-09-14 曾因静默丢 22 天数据）。
    # ★ 已确认取舍（2026-10-22，用户决策）：**连"新增无关列"也允许阻断日构建**。
    #   宁可当天构建失败、人工看一眼加进 _CANON_IGNORE，也不要静默降级。
    #   不要改成 warnings.warn 后继续跑。
    if unknown:
        raise ValueError(
            f"[updown] {filepath} 出现无法识别的列 {unknown}（站点改版？）\n"
            f"  · 有用 → 在 extra_features._CANON_FIELD 加映射（键=归一化后的语义词）\n"
            f"  · 无关 → 加入 _CANON_IGNORE\n"
            f"  · 归一化会自动剥掉『涨停/跌停』前缀与『(单位)』括号")

    # HH:MM:SS → 开盘后分钟(不含午休 11:30~13:00)
    def _t2min(t):
        """兼容两种口径：旧 `HH:MM:SS`；2026-09-14 起站点改版为 `YYYY-MM-DD HH:MM:SS`。

        用正则取尾部时刻而不是 strptime：改版后整串带日期，strptime('%H:%M:%S') 必
        挂 ⇒ 2026-09-14 起 first_limit_min/last_limit_min 全 NaN（已发生过一次）。
        """
        s = str(t).strip()
        mm_ = re.search(r'(\d{1,2}):(\d{2})(?::(\d{2}(?:\.\d+)?))?$', s)
        if s in ('', 'nan', 'None', '--') or mm_ is None:
            return np.nan
        hh, mi = int(mm_.group(1)), int(mm_.group(2))
        sec = float(mm_.group(3) or 0.0)
        m = hh * 60 + mi + sec / 60 - 570
        if   m < 0:    return np.nan   # 9:30 前
        if   m <= 120: return m        # 早盘 9:30~11:30 → 0~120
        if   m < 210:  return np.nan   # 午休 11:30~13:00
        if   m <= 330: return m - 90   # 午盘 13:00~15:00 → 120~240
        return np.nan                  # 15:00 后

    # '6,832.12万'→68321200, '3.27亿'→327000000, '218'→218
    _n = lambda s: float(s.replace(',', '').replace('亿', 'e8').replace('万', 'e4')) if isinstance(s, str) and s.strip() not in ('', '--') else np.nan

    def _pick(name: str):
        """rename 后可能重名（新旧列名并存）→ 取首个非空列；缺列→None。"""
        cols = df.loc[:, df.columns == name]
        if cols.shape[1] == 0:
            return None
        return cols.bfill(axis=1).iloc[:, 0]

    out = pd.DataFrame()
    out['code'] = df['code'].apply(lambda v: str(v).strip().split('.')[0].zfill(6)) if 'code' in df.columns else ''
    out['is_limit_up'] = int(is_limit_up)
    out['first_limit_min'] = df['first_limit_time'].apply(_t2min) if 'first_limit_time' in df.columns else np.nan
    out['last_limit_min'] = df['last_limit_time'].apply(_t2min) if 'last_limit_time' in df.columns else np.nan
    out['consecutive_days'] = pd.to_numeric(df['consecutive_days'], errors='coerce').fillna(0).astype(int) if 'consecutive_days' in df.columns else 0
    out['seal_volume_wan'] = df['seal_volume'].apply(lambda v: _n(v) / 1e6) if 'seal_volume' in df.columns else np.nan  # 股→万手
    out['seal_value_wan'] = df['seal_value'].apply(lambda v: _n(v) / 1e4) if 'seal_value' in df.columns else np.nan    # 元→万元
    out['seal_pct'] = pd.to_numeric(_pick('seal_pct'), errors='coerce')
    out['seal_flow_pct'] = pd.to_numeric(_pick('seal_flow_pct'), errors='coerce')
    out['open_count'] = pd.to_numeric(df['open_count'], errors='coerce').fillna(0).astype(int) if 'open_count' in df.columns else 0

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

    def _reason_col(frame: pd.DataFrame) -> pd.Series:
        """原因列：rename 后可能重名（新旧列名同时存在）；缺列→空串。"""
        cols = frame.loc[:, frame.columns == 'reason']
        if cols.shape[1] == 0:
            return pd.Series([''] * len(frame), index=frame.index, dtype=object)
        if cols.shape[1] == 1:
            return cols.iloc[:, 0].fillna('')
        return cols.bfill(axis=1).iloc[:, 0].fillna('')   # 先出现的非空值优先

    out['reason'] = _reason_col(df)
    return out


def build_updown_limit_feature(update: bool = False, force_rebuild: bool = False) -> pd.DataFrame:
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


def publish_updown_npy() -> int:
    """updown_limit.parquet（UI/决策共享源）→ loader_npy/updown 键表。"""
    ud = build_updown_limit_feature()
    if ud is None:
        return 0
    return save_key_table('updown', ud, UPDOWN_SUB_COLUMNS, 0.0,
                          source='updown_limit.parquet')


def build_market_features(update: bool = False, force_rebuild: bool = False) -> pd.DataFrame:
    """每天全市场统计: 平均涨幅、成交额、涨跌平家数、涨跌停家数、股票总数。

    状态与产物 = loader_npy/market 键表（key = tdi，无 code 轴）；meta.latest_date 记录增量水位。
    """
    existing = read_key_table_df('market') if not force_rebuild else None
    if not update and not force_rebuild:
        return existing
    latest = read_meta('market').get('latest_date', '00000000') if existing is not None else '00000000'

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

    updown = build_updown_limit_feature(update=(update or force_rebuild))
    if updown is not None:
        zt = updown[updown.is_limit_up == 1].groupby('date').size().rename('zhangting_count')
        dt = updown[updown.is_limit_up == -1].groupby('date').size().rename('dieting_count')
        daily = daily.join(zt, on='date').join(dt, on='date')
        daily[['zhangting_count', 'dieting_count']] = daily[['zhangting_count', 'dieting_count']].fillna(0).astype(int)

    daily['trade_date_idx'] = get_trade_date_idx(daily['date'].astype(str))
    combined = pd.concat([existing, daily], ignore_index=True) if existing is not None else daily
    combined = (combined.sort_values('trade_date_idx').drop_duplicates('trade_date_idx', keep='last')
                .reset_index(drop=True))
    combined = combined[MARKET_SUB_COLUMNS]
    save_key_table('market', combined, MARKET_SUB_COLUMNS, 0.0,
                   source='build_market_features',
                   extra={'latest_date': str(daily.date.max())})
    print(f'[market] +{len(daily)}天, {len(combined)}行, 最新{daily.date.max()}')
    return combined


PCT_RANK_COLUMNS = ['trade_date_idx', 'code_idx', 'pct_rank']


def build_pct_cross_rank(update: bool = False, force_rebuild: bool = False) -> pd.DataFrame:
    """个股当日涨幅的横截面分位排名（剔除 ST/新股，按 price_limit 归一后排名）。

    rel = pctChg / price_limit —— 源 base_feature 的 price_limit 在 isST=0 & isNew=0
    行即纯板幅（主板0.1 / 创业·科创0.2 / 北交0.3），故 rel 跨板块可比。当日全体可比
    股票 rel 升序分位 → [0,1]，天生逐日无基准率漂移。
    剔除 ST/新股后剩余样本再排名；被剔除者不进表（loader lookup 落空 → 0）。

    横截面 rank 按天独立，可增量（只算新 trade_date_idx）。
    """
    existing = read_key_table_df('pct_rank') if not force_rebuild else None
    if not update and not force_rebuild:
        return existing
    latest = int(existing['trade_date_idx'].max()) if existing is not None and len(existing) else -1

    import glob as _g
    from AshareData.datautils.dataloaders.feature_build.model_ready import (
        KLINE_NPY_DIR, part_glob, full_cols as npy_full_cols)
    cols = ['trade_date_idx', 'code_idx', 'pctChg', 'price_limit', 'isST', 'isNew']
    fc = npy_full_cols()
    jj = [fc.index(c) for c in cols]
    codes = sorted(d for d in os.listdir(KLINE_NPY_DIR)
                   if os.path.isdir(os.path.join(KLINE_NPY_DIR, d)))
    mats = []
    for code in codes:
        parts = sorted(_g.glob(part_glob(code)))
        if not parts:
            continue
        for p in (parts if force_rebuild else parts[-1:]):   # 增量只读尾页（新行必在尾页）
            m = np.load(p, mmap_mode="r", allow_pickle=False)
            mats.append(m[:, jj])
    d = (pd.DataFrame(np.concatenate(mats, axis=0), columns=cols)
         if mats else pd.DataFrame(columns=cols))
    d = d[(d.isST == 0) & (d.isNew == 0) & (d.price_limit > 0) & d.pctChg.notna()]
    if latest >= 0:
        d = d[d.trade_date_idx > latest]
    # 与旧 parquet(float64) 同口径：rank 前转回 float64（页存 float32）
    d = d.copy()
    d['pctChg'] = d['pctChg'].astype('float64')
    d['price_limit'] = d['price_limit'].astype('float64')
    d['rel'] = d['pctChg'] / d['price_limit']
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
    save_key_table('pct_rank', combined, PCT_RANK_COLUMNS, 0.0,
                   source='build_pct_cross_rank')
    print(f'[pct_rank] +{int(d.trade_date_idx.nunique())}天, {len(combined)}行, 最新tdi{int(combined.trade_date_idx.max())}')
    return combined


# ======================== 龙虎榜特征构建 ========================
# 解析 scrap_data/longhu/*.json → 8 特征列，每族 1 根：
# 方向 net_buy_ratio；结构 buy/sell_hhi；席位 buy_lhasa_ratio/famous_net_bias；
# 原因 reason_turnover/reason_amplitude；存在位 is_longhu。
# 未采用列：change/turnover_percent_norm（解析 bug）、top5_ratio（恒等共线）、
# north/inst 席位（t<3.7 且事件内非零率≤44%）。
# 同 (code,date) 多条 reason 记录 → 保留文件序最后一条。

LONGHU_DIR = f'{SCRAP_DATA_DIR}/longhu'

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


def build_longhu_feature(update: bool = False, force_rebuild: bool = False) -> pd.DataFrame:
    """构建龙虎榜历史特征(增量)。状态与产物 = loader_npy/longhu 键表；meta.latest_date 增量水位。"""
    existing = read_key_table_df('longhu') if not force_rebuild else None
    if not update:
        return existing
    latest = read_meta('longhu').get('latest_date', '00000000') if existing is not None else '00000000'
    new_files = [f for f in sorted(os.listdir(LONGHU_DIR)) if f.endswith('.json') and f[:10].replace('-', '') > latest]
    if not new_files:
        return existing

    new_frames = [_read_longhu_file(f'{LONGHU_DIR}/{f}') for f in new_files]
    newd = pd.concat(new_frames, ignore_index=True).drop_duplicates(['date', 'code'], keep='last')
    newd['trade_date_idx'] = get_trade_date_idx(newd['date'].astype(str))
    newd['code_idx'] = get_code_idx(newd['code'].astype(str))
    combined = pd.concat([existing, newd], ignore_index=True) if existing is not None else newd
    combined = (combined.sort_values('trade_date_idx')
                .drop_duplicates(['trade_date_idx', 'code_idx'], keep='last')
                .reset_index(drop=True))
    bad = int((combined['code_idx'] < 0).sum())
    if bad:
        print(f'[longhu] 丢弃无法定位 code_idx 的 {bad} 行（B股/退市等）')
        combined = combined[combined['code_idx'] >= 0]
    combined['trade_date_idx'] = combined['trade_date_idx'].astype('int64')
    combined['code_idx'] = combined['code_idx'].astype('int64')
    save_key_table('longhu', combined, LONGHU_SUB_COLUMNS, 0.0,
                   source='build_longhu_feature',
                   extra={'latest_date': max(f[:10].replace('-', '') for f in new_files)})
    print(f'[longhu] +{len(new_files)}天, {len(combined)}行, 最新{newd.date.max()}')
    return combined


if __name__ == '__main__':
    _ = build_updown_limit_feature(update=True, force_rebuild=True)
    data = build_market_features(force_rebuild=True)
    print(data)
    rank = build_pct_cross_rank(force_rebuild=True)
    print(rank)
    lh = build_longhu_feature(update=True, force_rebuild=True)
    print(lh)