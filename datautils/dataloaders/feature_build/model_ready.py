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

# 直喂 dataloader/模型的构建结果统一管理目录（SSD mmap 直读层）：
#   kline/   per-code 分页全字段（唯一行情存储）
#   updown|market|pct_rank|longhu|event|traits/   排序键表 keys.npy + vals.npy + meta.json
#   ann/     公告嵌入（直喂模型）
LOADER_NPY_DIR = os.path.join(BUILT_DATA_DIR, "loader_npy")

# npy 缓存目录（SSD 寻址层）：kline 分页在统一目录内
KLINE_NPY_DIR = os.path.join(LOADER_NPY_DIR, "kline")

# 公告嵌入目录（ann_index.npy / ann_emb.f16.npy / ann_title_*）
ANN_DIR = os.path.join(LOADER_NPY_DIR, "ann")

# —— 各副表子列契约（dataloader 按此序物化 lookup；构建器/发布器共用）——
UPDOWN_SUB_COLUMNS = [
    'trade_date_idx', 'code_idx',
    'is_limit_up', 'first_limit_min', 'last_limit_min', 'consecutive_days',
    'seal_volume_wan', 'seal_value_wan', 'seal_pct', 'seal_flow_pct',
    'open_count', 'days', 'boards',
]

MARKET_SUB_COLUMNS = [
    'trade_date_idx', 'avg_pctChg', 'amount_sum', 'zhangting_count', 'dieting_count',
    'up_count', 'down_count', 'flat_count', 'stock_count'
]

LONGHU_SUB_COLUMNS = [
    'trade_date_idx', 'code_idx',
    'is_longhu', 'net_buy_ratio', 'buy_hhi', 'sell_hhi',
    'buy_lhasa_ratio', 'famous_net_bias', 'reason_turnover', 'reason_amplitude',
]

EVENT_SUB_COLUMNS = [
    'trade_date_idx', 'code_idx',
    'ev_n_known', 'ev_n_base', 'ev_n_drv', 'ev_n_both',
    'ev_share', 'ev_mix', 'ev_base_pure', 'ev_age_min_frac', 'ev_is_new',
    'kp_sz', 'kp_rk', 'kp_fb', 'kp_age', 'kp_new', 'kp_gap', 'kp_rev', 'kp_nf',
    # KPL 强度（2026-10-21）：题材强度 th_* + 个股题材内强度 st_* + 距事件天数
    'ks_days', 'st_boards', 'st_rank', 'st_gapmax', 'st_is_leader', 'st_first',
    'st_seal', 'st_open', 'st_hist20', 'st_lastgap',
    'th_max_b', 'th_age', 'th_seal', 'th_first', 'th_promote',
]

TRAIT_SUB_COLUMNS = [
    'trade_date_idx', 'code_idx',
    't_mv', 't_lu120', 't_ld120', 't_bu120', 't_bd120', 't_sru', 't_srd',
    't_dsl', 't_hi', 't_lo', 't_yz', 't_zb', 't_turn', 't_beta',
]

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
