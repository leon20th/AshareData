"""巨潮公告标题嵌入 → 格子级 768 维表（index + npy），模型原样读。

产物（`built_data/`，给模型/下游用的只有这两个文件）
  ann_index.npy      (n,) int64，升序格子键 = trade_date_idx * 100000 + code_idx
  ann_emb.f16.npy    (n, 768) fp16，格子内「归零」标题嵌入的均值，**按列除以 std**

不做的事（重要）：**不做 PCA / 不降维 / 不并进 event_feat / 不加 loader 列组**。
768 维原样交给模型，后续投影、池化、门控都在模型内部做
（模型侧按 key = trade_date_idx * 100000 + code_idx 查表，见 model_v24.set_ann_table）。

口径（与实验线 business_tmp_files/cninfo_*/build_cell_text768.py 的 cell_v3 逐位一致，勿改）
- 只保留 announcementTime 时刻**恰为 00:00:00** 的「归零」条目；带时刻的那群已证是不同分布。
- 键锚定：k = bisect_left(TD, 归零日) − 1。该公告在**样本日（= k）**收盘后、次日开盘前发布
  ⇒ 用 k 当样本日**无前视**，且与模型 batch 里的 trade_date_idx 同一坐标系。
- 格子值 = 该格内标题嵌入（Fin-Retriever 768，L2 归一化）的**均值**；随后**按列除以该列 std**
  （**不中心化**——实验线 build_cell_text768.py 就是 `mean / std`，保持逐位一致）。
  标题**不 strip**，与实验线逐位对齐。

增量缓存（另两个文件，只为避免重复编码 1.9M 标题的 ~1h；不是给模型读的）
  ann_title_index.txt     唯一标题清单（行号 == 矩阵行号）
  ann_title_emb.f16.npy   (n, 768) fp16 标题嵌入，mmap 随机访问
  为何不是 parquet：全量 1.9M×768 = 2.8GB，而池化要按下标**随机取行**（行序=格子分组序
  ≠ 标题序）⇒ 必须能 mmap；parquet 要么整块进 heap，要么烧 IO。

用法
  python -m AshareData.datautils.dataloaders.feature_build.ann_emb              # 全量
  python -m AshareData.datautils.dataloaders.feature_build.ann_emb --limit-files 20   # 冒烟
"""
from __future__ import annotations

import bisect
import glob
import json
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from AshareData.paths import BUILT_DATA_DIR, FIN_RETRIEVER_DIR, SCRAP_DATA_DIR
from AshareData.utils.exchanges_utils.a_open import get_trade_date_list
from AshareData.utils.exchanges_utils.stock_utils import get_code_list

DIM = 768
KEY_MUL = 100000
CHUNK = 20_000
TZ = timezone(timedelta(hours=8))

STOCK_DIR = f'{SCRAP_DATA_DIR}/cninfo/stock'

# ---- 产物（模型读这两个）----
ANN_INDEX = f'{BUILT_DATA_DIR}/ann_index.npy'
ANN_EMB = f'{BUILT_DATA_DIR}/ann_emb.f16.npy'

# ---- 增量缓存（构建用）----
TITLE_INDEX = f'{BUILT_DATA_DIR}/ann_title_index.txt'
TITLE_MAT = f'{BUILT_DATA_DIR}/ann_title_emb.f16.npy'

# ---- 临时文件 ----
MEAN_TMP = f'{BUILT_DATA_DIR}/_ann_cell_mean.f16.npy'
ANN_EMB_TMP = f'{BUILT_DATA_DIR}/_ann_emb.tmp.npy'
TITLE_MAT_TMP = f'{BUILT_DATA_DIR}/_ann_title_emb.tmp.npy'

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
    """扫原始 JSON 取归零条目 → (keys int64, titles object)。limit_files>0 只扫前 N 个文件。"""
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
            k -= 1                      # 键里的 trade_date_idx = 样本日（见模块头「键锚定」）
            t = r.get('announcementTitle') or ''      # 不 strip：与实验线逐位对齐
            if not t:
                n_empty += 1
                continue
            keys.append(k * KEY_MUL + ci)
            titles.append(t)
    log(f'扫描 {len(files)} 个文件：条目 {n_all:,d} 保留 {len(keys):,d} '
        f'剔除(带时刻) {n_timed:,d} 空标题 {n_empty:,d} 越界 {n_bad} 代码不在网格 {n_skip}')
    return np.asarray(keys, np.int64), np.asarray(titles, object)


# --------------------------------------------------------------------------- #
# 2) 标题级嵌入（增量缓存）
# --------------------------------------------------------------------------- #
def _title_index() -> dict:
    """{title: 行号}；行号与 ann_title_emb.f16.npy 行一致。"""
    if not os.path.exists(TITLE_INDEX):
        return {}
    with open(TITLE_INDEX, encoding='utf-8') as f:
        return {ln.rstrip('\n'): i for i, ln in enumerate(f)}


def build_title_emb(titles: np.ndarray, batch: int = 512) -> dict:
    """只对新标题编码 → 追加缓存；返回 {title: 行号}。"""
    have = _title_index()
    todo = sorted({t for t in titles.tolist() if t not in have})
    if not todo:
        return have
    from sentence_transformers import SentenceTransformer
    dev = _device()
    t0 = time.time()
    model = SentenceTransformer(MODEL_DIR, device=dev)
    if dev == 'cuda':
        model.half()
    v = model.encode(todo, batch_size=batch, show_progress_bar=False,
                     normalize_embeddings=True, convert_to_numpy=True).astype(np.float16)
    # 追加 = 写新文件（拷旧块 + 新行）再原子替换（npy 头存 shape，无法原地长）
    old_n = len(have)
    W = np.lib.format.open_memmap(TITLE_MAT_TMP, mode='w+', dtype=np.float16,
                                  shape=(old_n + len(todo), DIM))
    if old_n:
        E = np.load(TITLE_MAT, mmap_mode='r')
        for a in range(0, old_n, CHUNK):
            b = min(a + CHUNK, old_n)
            W[a:b] = E[a:b]
    W[old_n:] = v
    W.flush()
    del W
    os.replace(TITLE_MAT_TMP, TITLE_MAT)
    with open(TITLE_INDEX, 'a', encoding='utf-8') as f:
        f.write('\n'.join(todo) + '\n')
    log(f'新编码标题 {len(todo):,d}（{dev} {time.time() - t0:.0f}s）'
        f' → 缓存共 {old_n + len(todo):,d} 行')
    return _title_index()


# --------------------------------------------------------------------------- #
# 3) 格子均值 + 按列标准化 → ann_index.npy + ann_emb.f16.npy
# --------------------------------------------------------------------------- #
def build_cell_table(t2i: dict, keys: np.ndarray, titles: np.ndarray) -> int:
    """格子均值 + 按列标准化，写出 (index, npy) 两个文件。返回格子数。"""
    t0 = time.time()
    E = np.load(TITLE_MAT, mmap_mode='r')          # 2.8GB 在盘上，按页随机读
    uk, inv = np.unique(keys, return_inverse=True)
    n_cell = len(uk)
    per_cell = np.bincount(inv, minlength=n_cell).astype(np.int64)
    assert (per_cell > 0).all()
    order = np.argsort(inv, kind='stable')
    starts = np.concatenate([[0], np.cumsum(per_cell)[:-1]])
    trow = np.fromiter((t2i[t] for t in titles[order].tolist()), np.int64, len(keys))
    log(f'格子 {n_cell:,d}  条目 {len(keys):,d}（{len(keys) / n_cell:.2f} 条/格）')
    del inv, order

    W = np.lib.format.open_memmap(MEAN_TMP, mode='w+', dtype=np.float16,
                                  shape=(n_cell, DIM))
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

    mu = colsum / n_cell                      # 仅用于算 std（输出不做中心化）
    sd = np.sqrt(np.maximum(colsq / n_cell - mu ** 2, 0.0))
    sd[sd < 1e-6] = 1.0
    log(f'按列 std：中位 {np.median(sd):.4f} 范围 [{sd.min():.4f}, {sd.max():.4f}]'
        f'（输出 = 均值 / std，不中心化 ⇒ 与实验线逐位一致）')

    Wr = np.load(MEAN_TMP, mmap_mode='r')
    out = np.lib.format.open_memmap(ANN_EMB_TMP, mode='w+', dtype=np.float16,
                                    shape=(n_cell, DIM))
    for a in range(0, n_cell, CHUNK):
        b = min(a + CHUNK, n_cell)
        out[a:b] = (Wr[a:b].astype(np.float32) / sd).astype(np.float16)
    out.flush()
    del out, Wr, W
    os.replace(ANN_EMB_TMP, ANN_EMB)                # 原子替换
    np.save(ANN_INDEX + '.tmp.npy', uk)             # 键升序（searchsorted 前提）
    os.replace(ANN_INDEX + '.tmp.npy', ANN_INDEX)
    os.remove(MEAN_TMP)
    log(f'写出 {ANN_INDEX}（{len(uk):,d} 键）+ {ANN_EMB}'
        f'（{os.path.getsize(ANN_EMB) / 1e9:.2f}GB，{time.time() - t0:.0f}s）')
    return n_cell


def update(limit_files: int = 0, force: bool = False) -> int:
    """入口：扫条目 → 增量编码标题 → 格子均值+除 std → (index, npy)。返回新编码标题数。"""
    keys, titles = scan_zero_anchors(limit_files)
    if not len(keys):
        log('无归零条目')
        return 0
    n0 = len(_title_index())
    t2i = build_title_emb(titles)
    n_new = len(t2i) - n0
    stale = ((not os.path.exists(ANN_EMB))
             or os.path.getmtime(ANN_EMB) < os.path.getmtime(TITLE_MAT))
    if force or n_new or stale:
        build_cell_table(t2i, keys, titles)
    else:
        log('格子表已是最新（缓存无新增、产物不旧于缓存），跳过；改了口径请加 --force')
    return n_new


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='巨潮公告标题嵌入 → 格子 768 维表（index + npy）')
    ap.add_argument('--limit-files', type=int, default=0, help='只扫前 N 个公告 JSON（冒烟）')
    ap.add_argument('--force', action='store_true', help='无视新鲜度判断，强制重建格子表')
    a = ap.parse_args()
    update(limit_files=a.limit_files, force=a.force)
