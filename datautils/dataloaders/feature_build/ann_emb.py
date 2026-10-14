"""巨潮公告标题 Fin-Retriever 嵌入 + 冻结 PCA16（2026-10-14 生产化）。

每日链增量（无新公告秒退）：
  scrap_data/cninfo/stock/*.json 的「归零」条目（announcementTime 时刻恰为 00:00:00）
  → 唯一标题增量嵌入（Fin-Retriever；CUDA 空闲用 CUDA / 忙则退 CPU；macOS 用 MPS；都无则 CPU）
  → built_data/ann_title_emb.parquet   标题级嵌入（title, emb[768]，L2 归一化）
  → 按格子 (date, code) 取标题嵌入均值 + 按列标准化
  → built_data/ann_cell_emb.parquet    格子级嵌入（date, code, emb[768]，fp16）
  → 冻结 PCA16（ann_cell_pca16.npz）   → built_data/ann_cell_emb16.parquet（ane_0..15）
  → 并入 event_feat.parquet（outer merge，与 kpe_* 同法；无公告行补 0）

口径（与实验线 business_tmp_files/cninfo_*/build_cell_text768.py 的 cell_v3 一致，勿改）
- 只保留 announcementTime 时刻**恰为 00:00:00** 的条目；带时刻的那群已证为不同分布。
- 标签日锚定：k = bisect_left(TD, 归零日) − 1，键 = k * 100000 + code_idx。
  归零 = announcementTime 的时刻恰为 00:00 ⇒ 该公告在**样本日（= k = 归零日的前一交易日）**
  收盘后、次日开盘前发布，所以用 k 作样本日**无前视**（与模型 batch 的 trade_date_idx 同一坐标系）。
- 格子值 = 该格内归零标题嵌入的**均值**；随后**按列标准化**（std 由格子分布得出，冻结）。
- key 约定 = trade_date_idx * 100000 + code_idx（与 loader / dataloader 一致）。

内存策略（本机 swap 常年吃紧，勿改成一次性大矩阵）
- 格子均值走 fp16 memmap 中转（989k×768 ≈ 1.5GB 磁盘，峰值 heap ≈ 2×分块）。
- PCA 拟合用**固定种子子样本**（上限 200k 格）⇒ 跨机可复现，且不吃 3GB（全量 989k×768 fp32 = 3.0GB）。
- 变换/标准化一律分块流式（`ParquetFile.iter_batches`），输出只留 (n, N_PCA)。

规则：
- 嵌入缓存按标题增量（只对新标题编码）；标准化与 PCA 冻结（ann_cell_norm.npz / ann_cell_pca16.npz）：
  新增只 transform 不 refit，重拟合需显式 build_pca(refit=True)（或删 npz）。保证特征跨期一致。
- event_feat 为训练关键文件：仅在新嵌入/缺列时改写（本模块幂等）。
- 训练侧启用：PPOFramework/model_framework/general_datatransform.py 的 EVENT_SETS['annemb']
  （loader 的 EVENT_SUB_COLUMNS 已含 ane_*，列序即 ane_0..ane_{N_PCA-1}）。
"""
from __future__ import annotations

import bisect
import glob
import json
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from AshareData.paths import BUILT_DATA_DIR, FIN_RETRIEVER_DIR, SCRAP_DATA_DIR
from AshareData.utils.exchanges_utils.a_open import get_trade_date_list
from AshareData.utils.exchanges_utils.stock_utils import get_code_list

DIM = 768
N_PCA = 16
KEY_MUL = 100000
PCA_FIT_MAX = 200_000
CHUNK = 20_000
TZ = timezone(timedelta(hours=8))

STOCK_DIR = f'{SCRAP_DATA_DIR}/cninfo/stock'
TITLE_EMB = f'{BUILT_DATA_DIR}/ann_title_emb.parquet'
CELL_EMB = f'{BUILT_DATA_DIR}/ann_cell_emb.parquet'
CELL_NORM = f'{BUILT_DATA_DIR}/ann_cell_norm.npz'
PCA_NPZ = f'{BUILT_DATA_DIR}/ann_cell_pca{N_PCA}.npz'
EMB_N = f'{BUILT_DATA_DIR}/ann_cell_emb{N_PCA}.parquet'
EVENT_FEAT_PARQUET = f'{BUILT_DATA_DIR}/event_feat.parquet'
MEAN_TMP = f'{BUILT_DATA_DIR}/_ann_cell_mean.f16.npy'
ANE_COLUMNS = [f'ane_{i}' for i in range(N_PCA)]
MODEL_DIR = FIN_RETRIEVER_DIR


def log(m: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def _device() -> str:
    import torch
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        return 'cuda' if free > 2.5e9 else 'cpu'
    if torch.backends.mps.is_available():      # macOS 无 CUDA：走 MPS（实测 ~2~3x）
        return 'mps'
    return 'cpu'


# --------------------------------------------------------------------------- #
# 1) 扫原始公告 → 归零条目
# --------------------------------------------------------------------------- #
def scan_zero_anchors(limit_files: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """扫原始 JSON 取归零条目 → (keys int64, titles object)。

    limit_files>0 时只扫文件名排序后的前 N 个（冒烟用）。
    """
    ci_map = {c.split('.')[-1]: i for i, c in enumerate(get_code_list())}
    td = get_trade_date_list()
    keys, titles = [], []
    files = sorted(glob.glob(f'{STOCK_DIR}/*.json'))
    if limit_files > 0:
        files = files[:limit_files]
    n_all = n_timed = n_empty = n_bad = n_skip = 0
    for nf, f in enumerate(files, 1):
        ci = ci_map.get(os.path.basename(f)[:-5])
        if ci is None:
            n_skip += 1
            continue
        with open(f, encoding='utf-8') as fh:
            rows = json.load(fh)
        for r in rows:
            ts = r.get('announcementTime')
            if not ts:
                continue
            n_all += 1
            dt = datetime.fromtimestamp(int(ts) / 1000, TZ)
            if not (dt.hour == 0 and dt.minute == 0 and dt.second == 0):
                n_timed += 1
                continue
            k = bisect.bisect_left(td, dt.strftime('%Y%m%d'))
            if k <= 0 or k >= len(td):
                n_bad += 1
                continue
            k -= 1                      # 键里的 trade_date_idx = 样本日（见模块头「标签日锚定」）
            # 标题**不 strip**：与实验线 build_cell_text768.py / build_title_emb.py 逐位对齐
            t = r.get('announcementTitle') or ''
            if not t:
                n_empty += 1
                continue
            keys.append(k * KEY_MUL + ci)
            titles.append(t)
        if nf % 1000 == 0:
            log(f'  {nf}/{len(files)}  保留 {len(keys):,d}')
    log(f'扫描 {len(files)} 个文件：条目 {n_all:,d} 保留 {len(keys):,d} '
        f'剔除(带时刻) {n_timed:,d} 空标题 {n_empty:,d} 越界 {n_bad} 代码不在网格 {n_skip}')
    return np.asarray(keys, np.int64), np.asarray(titles, object)


# --------------------------------------------------------------------------- #
# 2) 标题级嵌入（增量缓存）
# --------------------------------------------------------------------------- #
def _title_index() -> dict:
    if not os.path.exists(TITLE_EMB):
        return {}
    t = pq.read_table(TITLE_EMB, columns=['title'])
    return {s: i for i, s in enumerate(t['title'].to_pylist())}


def build_title_emb(titles: np.ndarray, batch: int = 512) -> dict:
    """只对新标题编码 → ann_title_emb.parquet；返回 {title: 行号}。"""
    have = _title_index()
    todo = sorted({t for t in titles.tolist() if t not in have})
    if todo:
        from sentence_transformers import SentenceTransformer
        dev = _device()
        t0 = time.time()
        model = SentenceTransformer(MODEL_DIR, device=dev)
        if dev == 'cuda':
            model.half()
        v = model.encode(todo, batch_size=batch, show_progress_bar=False,
                         normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
        arr = pa.FixedSizeListArray.from_arrays(pa.array(v.reshape(-1)), DIM)
        new = pa.table({'title': pa.array(todo), 'emb': arr})
        tbl = (new if not os.path.exists(TITLE_EMB)
               else pa.concat_tables([pq.read_table(TITLE_EMB), new]))
        pq.write_table(tbl, TITLE_EMB, compression='zstd')
        log(f'新编码标题 {len(todo):,d}（{dev} {time.time() - t0:.0f}s）'
            f' → ann_title_emb.parquet（共 {len(tbl):,d}）')
        have = {s: i for i, s in enumerate(tbl['title'].to_pylist())}
    if not have:
        raise RuntimeError('标题嵌入为空：先跑 build_title_emb')
    return have


def _flat_emb(col) -> np.ndarray:
    """FixedSizeList 列 → (n, DIM) numpy（分块调用，勿对全表调）。

    col 可能是 ChunkedArray（表）或 Array（RecordBatch）——两者都要能接。
    """
    if hasattr(col, 'combine_chunks'):
        col = col.combine_chunks()
    return col.flatten().to_numpy().reshape(-1, DIM)


# --------------------------------------------------------------------------- #
# 3) 格子均值 + 按列标准化
# --------------------------------------------------------------------------- #
def build_cell_emb(t2i: dict, keys: np.ndarray, titles: np.ndarray) -> None:
    """格子均值（fp16 memmap 中转）+ 按列标准化 → ann_cell_emb.parquet + ann_cell_norm.npz。"""
    t0 = time.time()
    E = _flat_emb(pq.read_table(TITLE_EMB, columns=['emb']).column(0))
    log(f'标题嵌入矩阵 {E.shape}  {E.nbytes / 1e6:.0f}MB')

    uk, inv = np.unique(keys, return_inverse=True)
    n_cell = len(uk)
    per_cell = np.bincount(inv, minlength=n_cell).astype(np.int64)
    assert (per_cell > 0).all()
    order = np.argsort(inv, kind='stable')
    starts = np.concatenate([[0], np.cumsum(per_cell)[:-1]])
    trow = np.fromiter((t2i[t] for t in titles[order].tolist()), np.int64, len(keys))
    log(f'格子 {n_cell:,d}  条目 {len(keys):,d}（{len(keys) / n_cell:.2f} 条/格）')
    del keys, titles, inv, order

    W = np.lib.format.open_memmap(MEAN_TMP, mode='w+', dtype=np.float16, shape=(n_cell, DIM))
    colsum = np.zeros(DIM, np.float64)
    colsq = np.zeros(DIM, np.float64)
    for a in range(0, n_cell, CHUNK):
        b = min(a + CHUNK, n_cell)
        cnt = per_cell[a:b]
        off = np.zeros(b - a, np.int64)
        np.cumsum(cnt[:-1], out=off[1:])
        rows = np.concatenate([np.arange(starts[i], starts[i] + per_cell[i])
                               for i in range(a, b)])
        g = E[trow[rows]].astype(np.float32)
        mu_c = np.add.reduceat(g, off, axis=0) / cnt[:, None]
        colsum += mu_c.sum(0, dtype=np.float64)
        colsq += (mu_c.astype(np.float64) ** 2).sum(0)
        W[a:b] = mu_c.astype(np.float16)
        if a % (CHUNK * 10) == 0:
            log(f'  池化 {b:,d}/{n_cell:,d}  ({time.time() - t0:.0f}s)')
    W.flush()
    del E, trow, per_cell, starts

    mu = colsum / n_cell
    sd = np.sqrt(np.maximum(colsq / n_cell - mu ** 2, 0.0))
    sd[sd < 1e-6] = 1.0
    np.savez(CELL_NORM, mean=mu.astype(np.float32), std=sd.astype(np.float32))
    log(f'按列标准化参数冻结 → ann_cell_norm.npz  std 中位 {np.median(sd):.4f} '
        f'范围 [{sd.min():.4f}, {sd.max():.4f}]')

    td = get_trade_date_list()
    codes = [c.split('.')[-1] for c in get_code_list()]
    # date = 键里的样本日（td[k]）——使 feature_utils.get_trade_date_idx(date) 回到同一个 k
    dates = [td[int(k) // KEY_MUL] for k in uk]
    clist = [codes[int(k) % KEY_MUL] for k in uk]
    del uk

    Wr = np.load(MEAN_TMP, mmap_mode='r')
    schema = pa.schema([('date', pa.string()), ('code', pa.string()),
                        ('emb', pa.list_(pa.float16(), DIM))])
    with pq.ParquetWriter(CELL_EMB, schema, compression='zstd') as w:
        for a in range(0, n_cell, CHUNK):
            b = min(a + CHUNK, n_cell)
            z = ((Wr[a:b].astype(np.float32) - mu) / sd).astype(np.float16)
            arr = pa.FixedSizeListArray.from_arrays(pa.array(z.reshape(-1)), DIM)
            w.write_table(pa.table({'date': pa.array(dates[a:b]),
                                    'code': pa.array(clist[a:b]), 'emb': arr}))
    del Wr, W
    os.remove(MEAN_TMP)
    log(f'格子嵌入 {n_cell:,d} 行 → ann_cell_emb.parquet（{time.time() - t0:.0f}s）')


# --------------------------------------------------------------------------- #
# 4) 冻结 PCA → ane_* 列
# --------------------------------------------------------------------------- #
def build_pca(refit: bool = False) -> int:
    """应用（必要时先拟合）冻结 PCA16 → ann_cell_emb16.parquet（ane_0..15）。返回行数。"""
    import pandas as pd

    pf = pq.ParquetFile(CELL_EMB)
    n = pf.metadata.num_rows
    if refit or not os.path.exists(PCA_NPZ):
        rng = np.random.default_rng(0)          # 固定种子 ⇒ 跨机可复现
        take = np.sort(rng.choice(n, size=min(n, PCA_FIT_MAX), replace=False))
        want = np.zeros(n, bool)
        want[take] = True
        Xf = np.empty((len(take), DIM), np.float32)
        pos = off = 0
        for batch in pf.iter_batches(batch_size=CHUNK, columns=['emb']):
            m = want[off:off + batch.num_rows]
            off += batch.num_rows
            if m.any():
                xb = _flat_emb(batch.column(0))
                Xf[pos:pos + int(m.sum())] = xb[m].astype(np.float32)
                pos += int(m.sum())
        from sklearn.decomposition import PCA
        p = PCA(n_components=N_PCA, random_state=0).fit(Xf)
        np.savez(PCA_NPZ, mean=p.mean_.astype(np.float32),
                 comp=p.components_.astype(np.float32),
                 evr=p.explained_variance_ratio_.astype(np.float32),
                 n_fit=len(take))
        log(f'PCA{N_PCA} 拟合完成（子样本 {len(take):,d} 格，'
            f'解释方差 {p.explained_variance_ratio_.sum():.1%}）')
        del Xf
    z = np.load(PCA_NPZ)
    mu, comp = z['mean'], z['comp']

    Z = np.empty((n, N_PCA), np.float32)
    pos = 0
    for batch in pf.iter_batches(batch_size=CHUNK, columns=['emb']):
        xb = _flat_emb(batch.column(0)).astype(np.float32)
        Z[pos:pos + len(xb)] = (xb - mu) @ comp.T
        pos += len(xb)
    idx = pq.read_table(CELL_EMB, columns=['date', 'code']).to_pandas()

    out = pd.DataFrame({'date': idx['date'].astype(str), 'code': idx['code'].astype(str)})
    for i in range(N_PCA):
        out[f'ane_{i}'] = Z[:, i]
    from AshareData.datautils.dataloaders.feature_build.feature_utils import (
        get_code_idx, get_trade_date_idx)
    out['trade_date_idx'] = get_trade_date_idx(out.date.tolist())
    out['code_idx'] = get_code_idx(out.code.tolist())
    out = out[(out.trade_date_idx >= 0) & (out.code_idx >= 0)]
    out = out.drop_duplicates(['trade_date_idx', 'code_idx'], keep='last').copy()
    out['trade_date_idx'] = out.trade_date_idx.astype('int64')
    out['code_idx'] = out.code_idx.astype('int64')
    out = out[['date', 'trade_date_idx', 'code_idx'] + ANE_COLUMNS]
    out.to_parquet(EMB_N, index=False)
    log(f'emb{N_PCA} {len(out):,d} 行（原始 {n:,d}）→ ann_cell_emb{N_PCA}.parquet')
    return len(out)


# --------------------------------------------------------------------------- #
# 5) 并入 event_feat
# --------------------------------------------------------------------------- #
def merge_event_feat(force: bool = False) -> int:
    """ane_* 并入 event_feat.parquet（outer merge，ev_*/kp_*/kpe_* 原样保留）。返回行数。"""
    import pandas as pd

    if not os.path.exists(EVENT_FEAT_PARQUET) or not os.path.exists(EMB_N):
        log('event_feat 或 emb16 不存在，跳过并表')
        return 0
    feat = pd.read_parquet(EVENT_FEAT_PARQUET)
    if all(c in feat.columns for c in ANE_COLUMNS) and not force:
        return 0
    b16 = pd.read_parquet(EMB_N, columns=['trade_date_idx', 'code_idx'] + ANE_COLUMNS)
    base = feat.drop(columns=[c for c in ANE_COLUMNS if c in feat.columns]).copy()
    for df in (base, b16):
        df['trade_date_idx'] = df.trade_date_idx.astype('int64')
        df['code_idx'] = df.code_idx.astype('int64')
    out = base.merge(b16, on=['trade_date_idx', 'code_idx'], how='outer')
    lead = ['trade_date_idx', 'code_idx']
    rest = [c for c in out.columns if c not in lead]
    for c in rest:
        if out[c].dtype.kind == 'f':
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0).astype(np.float32)
    out = out[lead + rest].sort_values(lead).reset_index(drop=True)
    out.to_parquet(EVENT_FEAT_PARQUET, index=False)
    log(f'event_feat {feat.shape} → {out.shape}（+{len(ANE_COLUMNS)} 列 ane_*）')
    return len(out)


def update(limit_files: int = 0) -> int:
    """每日链入口：扫条目 → 增量嵌入 → 格子均值 → PCA → 并入 event_feat。返回新编码标题数。"""
    keys, titles = scan_zero_anchors(limit_files)
    if not len(keys):
        log('无归零条目')
        return 0
    n0 = pq.read_table(TITLE_EMB, columns=['title']).num_rows if os.path.exists(TITLE_EMB) else 0
    t2i = build_title_emb(titles)
    n_new = len(t2i) - n0
    up_to_date = (os.path.exists(CELL_EMB)
                  and os.path.getmtime(CELL_EMB) >= os.path.getmtime(TITLE_EMB))
    if n_new or not up_to_date:
        build_cell_emb(t2i, keys, titles)
    stale = (not os.path.exists(EMB_N)) or os.path.getmtime(EMB_N) < os.path.getmtime(CELL_EMB)
    if stale:
        build_pca()
    merge_event_feat(force=n_new > 0)
    return n_new


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='巨潮公告嵌入 → event_feat（ane_*）')
    ap.add_argument('--limit-files', type=int, default=0, help='只扫前 N 个公告 JSON（冒烟）')
    ap.add_argument('--refit-pca', action='store_true', help='重拟合 PCA（改 N_PCA 后用）')
    ap.add_argument('--merge-only', action='store_true', help='只重跑 PCA+并表，不重算嵌入')
    a = ap.parse_args()
    if a.merge_only:
        build_pca(refit=a.refit_pca)
        merge_event_feat(force=True)
    else:
        update(limit_files=a.limit_files)
