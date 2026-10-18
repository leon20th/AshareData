# loader_tables —— 排序键表的落盘/读取契约（loader_npy 统一目录内）。
#
# 键表 = keys.npy(int64 升序, key = trade_date_idx * 100000 + code_idx) + vals.npy(float32) + meta.json。
# market 表无 code 轴：key = trade_date_idx。
# 发布侧（构建器/引导发布）原子写；dataloader 侧 mmap 只读，searchsorted 定位。
import json
import os

import numpy as np
import pandas as pd

from AshareData.datautils.dataloaders.feature_build.model_ready import LOADER_NPY_DIR


def table_dir(name: str) -> str:
    return os.path.join(LOADER_NPY_DIR, name)


def read_meta(name: str) -> dict:
    p = os.path.join(table_dir(name), "meta.json")
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}


def save_key_table(name: str, df: pd.DataFrame, cols, default: float,
                   source: str = "", extra: dict | None = None) -> int:
    """df(trade_date_idx[, code_idx], ...) → keys/vals/meta（tmp + rename 原子替换）。"""
    d = table_dir(name)
    os.makedirs(d, exist_ok=True)
    use = [c for c in cols if c not in ('trade_date_idx', 'code_idx')]
    for c in cols:
        if c not in df.columns:
            df[c] = np.float32(default)
    keys = df['trade_date_idx'].to_numpy(dtype=np.int64) * 100000 \
        + (df['code_idx'].to_numpy(dtype=np.int64) if 'code_idx' in cols else 0)
    vals = df[use].to_numpy(dtype=np.float32)
    order = np.argsort(keys, kind='stable')
    keys, vals = keys[order], vals[order]
    pid = os.getpid()
    for fname, arr in (('keys.npy', keys), ('vals.npy', vals)):
        tmp = os.path.join(d, f"{fname}.tmp{pid}.npy")
        np.save(tmp, arr)
        os.replace(tmp, os.path.join(d, fname))
    meta = {'cols': use, 'default': float(default), 'rows': int(len(keys)),
            'source': source, **(extra or {})}
    tmp = os.path.join(d, f"meta.json.tmp{pid}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False)
    os.replace(tmp, os.path.join(d, "meta.json"))
    return int(len(keys))


def load_key_table(name: str):
    """mmap 只读打开键表 → (keys int64 升序, vals float32)。"""
    d = table_dir(name)
    return (np.load(os.path.join(d, "keys.npy"), mmap_mode="r"),
            np.load(os.path.join(d, "vals.npy"), mmap_mode="r"))


def read_key_table_df(name: str) -> pd.DataFrame | None:
    """键表 → DataFrame（构建器增量状态用；无表返回 None）。"""
    d = table_dir(name)
    kp = os.path.join(d, "keys.npy")
    if not os.path.exists(kp):
        return None
    keys = np.load(kp, mmap_mode="r")
    vals = np.load(os.path.join(d, "vals.npy"), mmap_mode="r")
    df = pd.DataFrame(vals, columns=read_meta(name).get("cols") or [])
    df['trade_date_idx'] = (keys // 100000).astype(np.int64)
    df['code_idx'] = (keys % 100000).astype(np.int64)
    return df
