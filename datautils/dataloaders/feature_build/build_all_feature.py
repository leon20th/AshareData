from AshareData.paths import BASE_FEATURE_DIR
from AshareData.utils.exchanges_utils.stock_utils import get_code_list
from AshareData.datautils.dataloaders.feature_build.model_ready import (
    PAGE_SIZE, part_path, part_glob, code_dir, full_cols, save_manifest)

import glob
import json
import os
import numpy as np
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
from AshareData.datautils.dataloaders.feature_build.kpl_entities import (
    build as build_kpl_entities
)
from AshareData.datautils.dataloaders.feature_build.stock_traits import (
    build as build_stock_traits
)

FeatDIR = BASE_FEATURE_DIR
os.makedirs(FeatDIR, exist_ok=True)


def _write_pages(code, df, rebuild=False):
    """df → 分页 npy（全字段数值列、按交易日升序、float32，NaN 原样保留）。

    封存页只读不动（rebuild 或缺失才写）；尾页恒写（每日增量/回卷都在尾页内）；
    多余尾页删除。返回本票全字段列序。
    """
    full_cols_now = [c for c in sorted(df.columns) if c not in ("date", "code")]
    x = df.sort_values("trade_date_idx", kind="stable")[full_cols_now].to_numpy(dtype=np.float32)
    n = x.shape[0]
    n_pages = (n + PAGE_SIZE - 1) // PAGE_SIZE if n else 0
    os.makedirs(code_dir(code), exist_ok=True)
    for k in range(max(0, n_pages - 1)):
        target = part_path(code, k)
        if rebuild or not os.path.exists(target):
            tmp = f"{target}.tmp{os.getpid()}.npy"
            np.save(tmp, x[k * PAGE_SIZE:(k + 1) * PAGE_SIZE])
            os.replace(tmp, target)
    if n_pages:
        target = part_path(code, n_pages - 1)
        tmp = f"{target}.tmp{os.getpid()}.npy"
        np.save(tmp, x[(n_pages - 1) * PAGE_SIZE:])
        os.replace(tmp, target)
    for p in glob.glob(part_glob(code)):
        try:
            k = int(os.path.splitext(os.path.basename(p))[0].split("_")[1])
        except (ValueError, IndexError):
            continue
        if k >= n_pages:
            os.remove(p)
    return full_cols_now


def process_feature(codes, rebuild=False):
    """构建特征。codes 可以是单个 code 字符串或列表。"""
    if isinstance(codes, str):
        codes = [codes]
    latest_date, built, first_full_cols = '00000000', 0, None
    drift_n = 0
    for code in codes:
        if not os.path.exists(f'{KLINE_DIRS["daily"]}/{code}.csv'):
            continue
        df, force = build_merged_feature(code, rebuild)
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
        # 分页写 npy（封存页跳过、尾页重写、多余页删除；漂移票全页重写）
        fc = _write_pages(code, df, rebuild or force)
        if force:
            drift_n += 1
        if first_full_cols is None:
            first_full_cols = fc
        d = df['date'].max()
        if d > latest_date:
            latest_date = d
        built += 1
    return dict(
        stock_count=built,
        latest_date=latest_date,
        full_cols=first_full_cols,
        drift=drift_n,
    )


def build_base_feature(codes=None, workers=16, rebuild=False):
    if codes is None:
        codes = get_code_list()
    from concurrent.futures import ProcessPoolExecutor, as_completed
    latest_date, built, full_cols_built, drift_total = '00000000', 0, None, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(process_feature, c, rebuild): c for c in codes}
        for fut in tqdm.tqdm(as_completed(futs), total=len(futs), desc='build_all'):
            r = fut.result()
            built += r['stock_count']
            if r['latest_date'] > latest_date:
                latest_date = r['latest_date']
            if full_cols_built is None and r['full_cols']:
                full_cols_built = r['full_cols']
            drift_total += r['drift']
    print(f'[build_all] 复权锚点漂移重算: {drift_total} 票')
    # manifest：全字段列序（列集变化自动覆盖，dataloader 按名映射）
    if full_cols_built:
        try:
            _old = full_cols()
            if _old != full_cols_built:
                print(f'[build_all] 全字段列集变化: {len(_old)} → {len(full_cols_built)} 列')
        except FileNotFoundError:
            pass
        save_manifest(full_cols_built)
    schema = {c: 'float32' for c in (full_cols_built or [])}
    schema['date'] = 'string'
    schema['code'] = 'string'
    meta = {'build_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stock_count': built, 'latest_date': latest_date,
            'schema': schema}
    with open(f'{FeatDIR}/meta.json', 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return meta


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='特征构建（base_feature 增量；--rebuild 全史重建）')
    ap.add_argument('--rebuild', action='store_true', help='base_feature 全量重建（读时复权源，已存在 parquet 一并重算）')
    ap.add_argument('--daily-source', default='', choices=['', 'tushare', 'mix'],
                    help='日K读口管线（默认按环境变量 ASHARE_DAILY_SOURCE / tushare）')
    a = ap.parse_args()
    if a.daily_source:
        os.environ['ASHARE_DAILY_SOURCE'] = a.daily_source
    build_base_feature(rebuild=a.rebuild)
    build_updown_limit_feature(update=True)
    build_market_features(update=True)
    build_pct_cross_rank(update=True)  # 依赖 base_feature，须在其后
    build_longhu_feature(update=True)
    build_stock_traits()                # 股性 traits 14 列（全量重建 ~10s；依赖 base_feature/updown/market）
    build_kpl_features(update=True)     # 开盘啦事件结构（依赖 scrap_kaipanla 产物）
    build_kpl_entities()                # KPL 题材实体化（实体/别名/成员/复核 四件套，秒级）
    # KPL 细分/行级两阶段 AI 映射增量（无新数据秒退；API 不可用自动跳过不阻断；
    # 有新映射才重建实体）
    try:
        from AshareData.datautils.dataloaders.feature_build.kpl_segment_map_ai import (
            update as kpl_segmap_update)
        if kpl_segmap_update():
            build_kpl_entities()
    except Exception as exc:
        print(f'[kpl] 细分/行级映射更新跳过: {exc}')
    # KPL 涨停原因嵌入 + 冻结 PCA16 → event_feat 列 kpe_*（增量；无新事件秒退；
    # GPU 忙自动退 CPU；失败不阻断）
    try:
        from AshareData.datautils.dataloaders.feature_build.kpl_reason_emb import (
            update as kpl_reason_emb_update)
        kpl_reason_emb_update()
    except Exception as exc:
        print(f'[kpl] 原因嵌入更新跳过: {exc}')
    # 巨潮公告标题嵌入 → 格子级 768 维表（ann_index.npy + ann_emb.f16.npy）：
    # 增量（标题级嵌入缓存命中则不重编码）；失败不阻断
    try:
        from AshareData.datautils.dataloaders.feature_build.ann_emb import (
            update as ann_emb_update)
        ann_emb_update()
    except Exception as exc:
        print(f'[ann] 公告嵌入更新跳过: {exc}')
    # 新闻快讯标签计数特征 → news_features.parquet（东财 7x24 + 财联社；全量重建 ~8s；
    # 原始快讯缺失则跳过，不阻断）
    try:
        from AshareData.datautils.dataloaders.feature_build.news_features import (
            build_news_features)
        build_news_features(force_rebuild=True)
    except Exception as exc:
        print(f'[news] 新闻特征更新跳过: {exc}')
