"""KPL 涨停原因 Fin-Retriever 嵌入 + 冻结 PCA16（2026-09-25 生产化）。

每日链增量（无新事件秒退）：
  kpl_events.reason_txt → 增量嵌入（GPU 空闲用 GPU / 忙或无卡自动退 CPU）
  → built_data/kpl_reason_emb.parquet    事件级嵌入（date, code, emb[768]，L2 归一化）
  → 冻结 PCA16（kpl_reason_pca16.npz）  → built_data/kpl_reason_emb16.parquet（kpe_0..15）
  → 并入 event_feat.parquet（outer merge，与 kp_* 同法；无事件行补 0）

规则：
- PCA 冻结：新增事件只 transform 不 refit（保证特征跨期一致）；重拟合需显式
  build_pca16(refit=True)（或删 npz）。
- event_feat 为训练关键文件：仅在新嵌入/缺列时改写（本模块幂等）。
- 训练侧启用（loader EVENT_SUB_COLUMNS / EVENT_SETS 加一组）不在本模块，需要时另行显式接线。
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from AshareData.paths import BUILT_DATA_DIR
from AshareData.datautils.dataloaders.feature_build.feature_utils import (
    get_code_idx,
    get_trade_date_idx,
)

EVENTS = f'{BUILT_DATA_DIR}/kpl_events.parquet'
EMB = f'{BUILT_DATA_DIR}/kpl_reason_emb.parquet'
PCA_NPZ = f'{BUILT_DATA_DIR}/kpl_reason_pca16.npz'
EMB16 = f'{BUILT_DATA_DIR}/kpl_reason_emb16.parquet'
EVENT_FEAT_PARQUET = f'{BUILT_DATA_DIR}/event_feat.parquet'
MODEL_DIR = 'business_models/external_models/Fin-Retriever-base'
N_PCA = 16
KPE_COLUMNS = [f'kpe_{i}' for i in range(N_PCA)]


def _clean(s: pd.Series) -> pd.Series:
    return s.fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()


def _device() -> str:
    import torch
    if not torch.cuda.is_available():
        return 'cpu'
    free, _ = torch.cuda.mem_get_info()
    return 'cuda' if free > 2.5e9 else 'cpu'


def build_reason_emb(force_rebuild: bool = False, batch: int = 256) -> int:
    """增量嵌入 kpl_events.reason_txt → kpl_reason_emb.parquet。返回新增行数。"""
    ev = pd.read_parquet(EVENTS, columns=['date', 'code', 'reason_txt'])
    ev['date'] = ev.date.astype(str)
    ev['code'] = ev.code.astype(str).str.zfill(6)
    ev['text'] = _clean(ev.reason_txt)
    ev = ev[ev.text != ''].drop_duplicates(['date', 'code'], keep='last').reset_index(drop=True)
    if not force_rebuild and os.path.exists(EMB):
        ok = pq.read_table(EMB, columns=['date', 'code'])
        have = set(zip(ok['date'].to_pylist(), ok['code'].to_pylist()))
        ev = ev[[k not in have for k in zip(ev.date, ev.code)]].reset_index(drop=True)
    if not len(ev):
        print('[kpl-emb] 无新事件')
        return 0

    from sentence_transformers import SentenceTransformer
    dev = _device()
    t0 = time.time()
    model = SentenceTransformer(MODEL_DIR, device=dev)
    model.max_seq_length = 512
    if dev == 'cuda':
        model.half()
    v = model.encode(ev.text.tolist(), batch_size=batch, show_progress_bar=False,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    arr = pa.FixedSizeListArray.from_arrays(pa.array(v.reshape(-1)), 768)
    new = pa.table({'date': pa.array(ev.date.values),
                    'code': pa.array(ev.code.values), 'emb': arr})
    tbl = (new if (force_rebuild or not os.path.exists(EMB))
           else pa.concat_tables([pq.read_table(EMB), new]))
    pq.write_table(tbl, EMB, compression='zstd')
    print(f'[kpl-emb] 新增嵌入 {len(ev)} 行（{dev} {time.time() - t0:.0f}s）'
          f' → kpl_reason_emb.parquet（共 {len(tbl)}）')
    return len(ev)


def build_pca16(refit: bool = False) -> pd.DataFrame:
    """应用（必要时先拟合）冻结 PCA16 → kpl_reason_emb16.parquet。"""
    tbl = pq.read_table(EMB)
    X = tbl['emb'].combine_chunks().flatten().to_numpy().reshape(-1, 768).astype(np.float32)
    if refit or not os.path.exists(PCA_NPZ):
        from sklearn.decomposition import PCA
        p = PCA(n_components=N_PCA, random_state=0).fit(X)
        np.savez(PCA_NPZ, mean=p.mean_.astype(np.float32),
                 comp=p.components_.astype(np.float32),
                 evr=p.explained_variance_ratio_.astype(np.float32))
        print(f'[kpl-emb] PCA{N_PCA} 拟合完成（解释方差 {p.explained_variance_ratio_.sum():.1%}）')
    z = np.load(PCA_NPZ)
    Z = ((X - z['mean']) @ z['comp'].T).astype(np.float32)
    out = pd.DataFrame({'date': tbl['date'].to_pandas().astype(str),
                        'code': tbl['code'].to_pandas().astype(str)})
    for i in range(N_PCA):
        out[f'kpe_{i}'] = Z[:, i]
    out['trade_date_idx'] = get_trade_date_idx(out.date.tolist())
    out['code_idx'] = get_code_idx(out.code.tolist())
    out = out[(out.trade_date_idx >= 0) & (out.code_idx >= 0)]
    out = out.drop_duplicates(['trade_date_idx', 'code_idx'], keep='last').copy()
    out['trade_date_idx'] = out.trade_date_idx.astype('int64')
    out['code_idx'] = out.code_idx.astype('int64')
    out = out[['date', 'trade_date_idx', 'code_idx'] + KPE_COLUMNS]
    out.to_parquet(EMB16, index=False)
    print(f'[kpl-emb] emb16 {len(out)} 行（原始 {len(Z)}）→ kpl_reason_emb16.parquet')
    return out


def merge_event_feat(force: bool = False) -> int:
    """kpe_* 并入 event_feat.parquet（outer merge，ev_*/kp_* 原样保留）。返回行数。"""
    if not os.path.exists(EVENT_FEAT_PARQUET) or not os.path.exists(EMB16):
        return 0
    feat = pd.read_parquet(EVENT_FEAT_PARQUET)
    if all(c in feat.columns for c in KPE_COLUMNS) and not force:
        return 0
    b16 = pd.read_parquet(EMB16, columns=['trade_date_idx', 'code_idx'] + KPE_COLUMNS)
    base = feat.drop(columns=[c for c in KPE_COLUMNS if c in feat.columns]).copy()
    base['trade_date_idx'] = base.trade_date_idx.astype('int64')
    base['code_idx'] = base.code_idx.astype('int64')
    out = base.merge(b16, on=['trade_date_idx', 'code_idx'], how='outer')
    ev_cols = [c for c in out.columns if c.startswith('ev_')]
    kp_cols = [c for c in out.columns if c.startswith('kp_')]
    for c in ev_cols + kp_cols + KPE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0).astype(np.float32)
    out = out[['trade_date_idx', 'code_idx'] + ev_cols + kp_cols + KPE_COLUMNS]
    out = out.sort_values(['trade_date_idx', 'code_idx']).reset_index(drop=True)
    out.to_parquet(EVENT_FEAT_PARQUET, index=False)
    print(f'[kpl-emb] event_feat {feat.shape} → {out.shape}（+{len(KPE_COLUMNS)} 列 kpe_*）')
    return len(out)


def update() -> int:
    """每日链入口：增量嵌入 → PCA16 → 并入 event_feat。返回新增嵌入行数。"""
    n = build_reason_emb()
    stale16 = (not os.path.exists(EMB16)) or os.path.getmtime(EMB16) < os.path.getmtime(EMB)
    if stale16:
        build_pca16()
    merge_event_feat(force=n > 0)
    return n


if __name__ == '__main__':
    update()
