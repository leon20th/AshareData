"""模型直载产物（分页 .npy）的契约：build_all_feature 写出、dataloader/下游特征 mmap 直读。

布局：base_feature_npy/<code>/part_%05d.npy，每页 ≤ PAGE_SIZE 行、全字段 float32、
按 trade_date_idx 升序；封存页只读不动，尾页每次构建重写（特征回卷只影响尾页）；
columns.json 记录全字段列序（FULL_COLS），dataloader 按名映射模型列。
date/code 字符串列不入页 —— 需要处由 trade_date_idx/日历、code_idx 派生。
"""
import glob
import json
import os

from AshareData.paths import BUILT_DATA_DIR

CORE_REQUIRED_COLUMNS = [
    "trade_date_idx", "code_idx",
    "open", "high", "low", "close", "preclose", "volume", "amount",
    "turn", "pctChg", "isNew", "isST", "price_limit"
]

Daily_tech_COLUMNS = [
    "ma5", "ma10", "ma20", "ma30", "ma60",
    "macd", "macd_signal", "macd_hist",
    "rsi14",
    "boll_upper", "boll_mid", "boll_lower",
    "kdj_k", "kdj_d", "kdj_j",
]

_M15_TIME_SLOTS = [
    "0945", "1000", "1015", "1030", "1045", "1100", "1115", "1130",
    "1315", "1330", "1345", "1400", "1415", "1430", "1445", "1500",
]
M15_tech_COLUMNS = [
    f"m15_{t}_{feat}"
    for t in _M15_TIME_SLOTS
    for feat in ("rsi14", "kdj_k")
]

# 模型直载列序 = dataloader 默认 need_cols
MODEL_READY_COLS = CORE_REQUIRED_COLUMNS + Daily_tech_COLUMNS + M15_tech_COLUMNS

# npy 缓存目录（SSD 寻址层）
KLINE_NPY_DIR = os.path.join(BUILT_DATA_DIR, "base_feature_npy")

PAGE_SIZE = 256          # 每页最大行数（> 特征重算回卷窗 ~100 行：封存页永不被回卷波及）


def part_path(code: str, k: int) -> str:
    """第 k 页 npy 路径：KLINE_NPY_DIR/<code>/part_%05d.npy。"""
    return os.path.join(KLINE_NPY_DIR, str(code), f"part_{k:05d}.npy")


def part_glob(code: str) -> str:
    return os.path.join(KLINE_NPY_DIR, str(code), "part_*.npy")


def code_dir(code: str) -> str:
    return os.path.join(KLINE_NPY_DIR, str(code))


_FULL_COLS_CACHE = None


def full_cols() -> list:
    """columns.json 的全字段列序（进程内缓存）。"""
    global _FULL_COLS_CACHE
    if _FULL_COLS_CACHE is None:
        with open(os.path.join(KLINE_NPY_DIR, "columns.json"), "r", encoding="utf-8") as f:
            _FULL_COLS_CACHE = list(json.load(f)["cols"])
    return _FULL_COLS_CACHE


def save_manifest(cols: list) -> None:
    """原子写 columns.json（构建侧调用）。"""
    os.makedirs(KLINE_NPY_DIR, exist_ok=True)
    tmp = os.path.join(KLINE_NPY_DIR, f"columns.json.tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"cols": list(cols)}, f)
    os.replace(tmp, os.path.join(KLINE_NPY_DIR, "columns.json"))
