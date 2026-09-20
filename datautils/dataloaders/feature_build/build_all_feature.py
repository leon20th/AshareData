from AshareData.paths import BASE_FEATURE_DIR
from AshareData.utils.exchanges_utils.stock_utils import get_code_list

import json
import os
import tqdm
from datetime import datetime

from AshareData.datautils.dataloaders.feature_build.feature_calc import (
    build_merged_feature,
    KLINE_DIRS,
    PARQUET_NUM_SCHEMA,
)
from AshareData.datautils.dataloaders.feature_build.extra_features import (
    build_updown_limit_feature,
    build_market_features,
    build_pct_cross_rank,
    build_longhu_feature
)
from AshareData.datautils.dataloaders.feature_build.kpl_features import (
    build_kpl_features
)
from AshareData.datautils.dataloaders.feature_build.stock_traits import (
    build as build_stock_traits
)

FeatDIR = BASE_FEATURE_DIR
os.makedirs(FeatDIR, exist_ok=True)

def process_feature(codes, rebuild=False):
    """构建特征。codes 可以是单个 code 字符串或列表。"""
    if isinstance(codes, str):
        codes = [codes]
    latest_date, built = '00000000', 0
    for code in codes:
        if not os.path.exists(f'{KLINE_DIRS["daily"]}/{code}.csv'):
            continue
        df = build_merged_feature(code, rebuild)
        if df is None:
            continue
        # 按固定 schema 统一数值类型，避免 NaN 导致 int/float 不一致
        for col, dtype in PARQUET_NUM_SCHEMA.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
        # schema 未覆盖的 int 列也统一转 float64
        for col in df.columns:
            if col not in PARQUET_NUM_SCHEMA and df[col].dtype in ('int64', 'Int64'):
                df[col] = df[col].astype('float64')
        df.to_parquet(f'{FeatDIR}/{code}.parquet', index=False)
        d = df['date'].max()
        if d > latest_date:
            latest_date = d
        built += 1
    return dict(
        stock_count=built,
        latest_date=latest_date
    )


def build_base_feature(codes=None, workers=16, rebuild=False):
    if codes is None:
        codes = get_code_list()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    latest_date, built = '00000000', 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(process_feature, c, rebuild): c for c in codes}
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs), desc='build_all'):
            r = fut.result()
            built += r['stock_count']
            if r['latest_date'] > latest_date:
                latest_date = r['latest_date']
    # 写meta（含 schema）
    schema = {}
    import pyarrow.parquet as pq
    for f in os.listdir(FeatDIR):
        if f.endswith('.parquet'):
            pf = pq.read_schema(f'{FeatDIR}/{f}')
            schema = {field.name: str(field.type) for field in pf}
            break
    meta = {'build_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stock_count': built, 'latest_date': latest_date,
            'schema': schema}
    with open(f'{FeatDIR}/meta.json', 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


if __name__ == '__main__':
    build_base_feature(rebuild=False)
    build_updown_limit_feature(update=True)
    build_market_features(update=True)
    build_pct_cross_rank(update=True)  # 依赖 base_feature，须在其后
    build_longhu_feature(update=True)
    build_stock_traits()                # 股性 traits 14 列（全量重建 ~10s；依赖 base_feature/updown/market）
    build_kpl_features(update=True)     # 开盘啦事件结构（依赖 scrap_kaipanla 产物）
