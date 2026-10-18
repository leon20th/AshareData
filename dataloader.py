"""极简 dataloader —— parquet → numpy，不做列计算/特征工程。

【存储层 = SSD 寻址】行情数据先落 per-code float32 `.npy` 缓存，storage 以 mmap
只读映射（页面按需从 SSD 换入，不进进程私有内存）：
- 进程常驻内存 ≈ 0（页面由 OS page cache 共享，多进程训练不再 ×N 复制）；
- 缓存增量维护：npy 缺失或 mtime 早于 parquet → 只重建该票（列集变化全量兜底）；
- 逐位等价旧实现：同列序、同排序、同日期过滤，样本窗口内容不变。

【数据唯一出口】
- 所有数据必须经 `build_train_and_val_dataloaders` 获取：训练/eval 直接消费返回的
train/val DataLoader；预测/分析等只需 storage 的场景走 `storage_only=True` 轻量模式
（返回 (storage, None)，不构建样本池）。
- 看K线展示例外：`load_single_stock_kline` 为本模块提供的快速单文件读（仅展示
快照，非模型输入路径）；除此之外外部不得自行读 parquet / extra 特征源。
- 内部实现（_KlineDataStorage / _KlineDataset / _*BatchSampler）为模块私有细节，
仅本模块与白盒测试可直接引用。
"""
from __future__ import annotations

from bisect import bisect_right
from datetime import datetime
import glob
import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

import random

import numpy as np
import pandas as pd
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset, default_collate

from AshareData.paths import BASE_FEATURE_DIR, BUILT_DATA_DIR
from AshareData.datautils.dataloaders.feature_build.model_ready import (
    CORE_REQUIRED_COLUMNS,
    Daily_tech_COLUMNS,
    M15_tech_COLUMNS,
    MODEL_READY_COLS,
    KLINE_NPY_DIR,
    part_glob,
    full_cols,
    UPDOWN_SUB_COLUMNS,
    MARKET_SUB_COLUMNS,
    LONGHU_SUB_COLUMNS,
    EVENT_SUB_COLUMNS,
    TRAIT_SUB_COLUMNS,
)
from AshareData.datautils.dataloaders.feature_build.loader_tables import (
    load_key_table,
)
from AshareData.utils.exchanges_utils.a_open import get_trade_date_list
from AshareData.utils.log_util import get_logger

logger = get_logger("kline_dataset_build.dataloader_np")

class _PagedCodeArray:
    """只读分页矩阵视图：对外 (n_total, n_model)，底层 = 若干 (≤PAGE_SIZE, n_full) 页。

    页文件以 mmap 打开，访问只物化命中的行；行切片返回共享页的惰性视图；
    __array__ 支持 np.concatenate / np.asarray（物化）。列仅支持单个整数索引
    （覆盖现有消费方的全部用法）。
    """

    def __init__(self, pages: List[np.ndarray], col_idx: List[int],
                 row0: int = 0, offs: Optional[np.ndarray] = None,
                 row_end: Optional[int] = None) -> None:
        self._pages = pages
        self._col_idx = np.asarray(col_idx, dtype=np.int64)
        self._row0 = int(row0)
        if offs is None:
            offs = np.empty(len(pages) + 1, dtype=np.int64)
            offs[0] = 0
            for i, p in enumerate(pages):
                offs[i + 1] = offs[i] + p.shape[0]
        self._offs = offs
        self._row_end = int(offs[-1]) if row_end is None else int(row_end)
        self._full_ncols = int(pages[0].shape[1])

    @property
    def shape(self) -> Tuple[int, int]:
        return (self._row_end - self._row0, len(self._col_idx))

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype=None) -> np.ndarray:
        out = self._global_rows(np.arange(self.shape[0], dtype=np.int64))[:, self._col_idx]
        return out.astype(dtype) if dtype is not None else out

    def _global_rows(self, idxs: np.ndarray) -> np.ndarray:
        g = idxs + self._row0
        p = np.searchsorted(self._offs, g, side="right") - 1
        out = np.empty((len(g), self._full_ncols), dtype=np.float32)
        for k in range(len(self._pages)):
            m = p == k
            if m.any():
                out[m] = self._pages[k][g[m] - self._offs[k]]
        return out

    def _rows(self, rows) -> np.ndarray:
        n = self.shape[0]
        if isinstance(rows, (int, np.integer)):
            i = int(rows)
            return self._global_rows(np.array([i + n if i < 0 else i], dtype=np.int64))[0]
        if isinstance(rows, slice):
            lo, hi, st = rows.indices(n)
            return self._global_rows(np.arange(lo, hi, st, dtype=np.int64))
        if isinstance(rows, np.ndarray) and rows.dtype == bool:
            rows = np.nonzero(rows)[0]
        idxs = np.asarray(rows, dtype=np.int64)
        idxs = np.where(idxs < 0, idxs + n, idxs)
        return self._global_rows(idxs)

    def _row_view(self, i0: int, i1: int) -> "_PagedCodeArray":
        offs = self._offs
        k0 = int(np.searchsorted(offs, i0 + self._row0, side="right")) - 1
        k1 = int(np.searchsorted(offs, i1 - 1 + self._row0, side="right"))
        v = _PagedCodeArray.__new__(_PagedCodeArray)
        v._pages = self._pages[k0 : k1 + 1]
        v._col_idx = self._col_idx
        v._row0 = i0 + self._row0
        v._offs = offs[k0 : k1 + 2].copy()   # 页边界保持全局值（gather 按页内偏移定位）
        v._row_end = v._row0 + (i1 - i0)     # 视图逻辑终点 clamp 到切片右端
        v._full_ncols = self._full_ncols
        return v

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            if isinstance(item, slice) and item.step in (None, 1):
                lo, hi, _ = item.indices(self.shape[0])
                return self._row_view(lo, hi) if lo < hi else self._rows(item)[:, self._col_idx]
            if isinstance(item, (int, np.integer)):
                return self._rows(item)[self._col_idx]   # 单行 → 列映射
            return self._rows(item)[:, self._col_idx]
        rows, col = item
        if not isinstance(col, (int, np.integer)):
            raise NotImplementedError("列仅支持单个整数索引")
        base = self._rows(rows)
        j = int(self._col_idx[int(col)])
        return base[j] if isinstance(rows, (int, np.integer)) else base[:, j]


def _load_paged_code(code: str, model_cols: List[str]) -> Optional[_PagedCodeArray]:
    """打开某票全部分页（mmap），按 manifest 列名映射为模型列视图。"""
    parts = sorted(glob.glob(part_glob(code)))
    if not parts:
        return None
    full = full_cols()
    col_idx = [full.index(c) for c in model_cols if c in full]
    pages = [np.load(p, mmap_mode="r", allow_pickle=False) for p in parts]
    return _PagedCodeArray(pages, col_idx)


@dataclass
class KlineDataConfig:
    feature_len: int = 60
    max_codes: Optional[int] = None
    special_codes: List[str] = field(default_factory=list)
    begin_date: Optional[str] = None
    end_date: Optional[str] = None
    data_dir: str = BASE_FEATURE_DIR
    need_cols: Optional[List[str]] = None
    read_chunk_size: Optional[int] = 256


class _KlineDataStorage:
    def __init__(
        self,
        config: Optional[KlineDataConfig] = None,
        special_codes: Optional[List[str]] = None,
        begin_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        self.config = config or KlineDataConfig()
        self.feature_len = int(self.config.feature_len)
        selected_codes = self.config.special_codes if special_codes is None else special_codes
        selected_begin_date = self.config.begin_date if begin_date is None else begin_date
        selected_end_date = self.config.end_date if end_date is None else end_date
        self.special_codes: Set[str] = {code for code in selected_codes if code}
        self.begin_date = self._normalize_date_to_int(selected_begin_date)
        self.end_date = self._normalize_date_to_int(selected_end_date)
        self.trade_dates = [int(date) for date in get_trade_date_list()]
        self._date_to_idx_map: Dict[int, int] = {d: i for i, d in enumerate(self.trade_dates)}

        if self.feature_len <= 0:
            raise ValueError("feature_len must be positive")
        if self.begin_date is not None and self.end_date is not None and self.begin_date > self.end_date:
            raise ValueError("begin_date must be less than or equal to end_date")

        # data_meta
        with open(f'{self.config.data_dir}/meta.json', "r", encoding="utf-8") as f:
            self.data_meta = json.load(f)

        files = sorted(
            d for d in os.listdir(KLINE_NPY_DIR)
            if os.path.isdir(os.path.join(KLINE_NPY_DIR, d))
        )
        if self.special_codes:
            files = [d for d in files if d in self.special_codes]
        if self.config.max_codes is not None:
            files = files[: self.config.max_codes]

        self.code_arrays: Dict[str, np.ndarray] = {}       # code → (n, n_cols) float32
        self.code_columns: List[str] = []                   # 列名顺序

        logger.info(
            "start building kline storage: codes=%s seq_len=%s begin_date=%s end_date=%s special_codes=%s data_dir=%s",
            len(files),
            self.feature_len,
            self.begin_date,
            self.end_date,
            len(self.special_codes),
            self.config.data_dir,
        )

        need_cols = self.config.need_cols or MODEL_READY_COLS

        # manifest（构建侧维护的全字段列序）→ 模型列按名映射；缺列剔除同旧语义
        _full_cols = full_cols()
        missing = [c for c in need_cols if c not in _full_cols]
        if missing:
            logger.warning("need_cols 中以下列不在 npy manifest 中，已忽略: %s", missing)
        need_cols = [c for c in need_cols if c in _full_cols]
        self._effective_need_cols = need_cols

        if not files:
            logger.info("storage built: codes=0 skipped(short=0)")
            self.load_extra_data()
            return

        # SSD 寻址：build 产物 = 分页 npy（封存页不可变），mmap 只读打开、页面按需换入；
        # begin/end 行过滤在分页视图上惰性切片（共享页 mmap，不物化）。
        self.code_columns = need_cols
        date_col = (
            need_cols.index("trade_date_idx") if "trade_date_idx" in need_cols
            else need_cols.index("date") if "date" in need_cols
            else 0
        )
        lo, hi = self._date_row_range()

        skipped_short = 0
        for code in files:
            arr = _load_paged_code(code, need_cols)
            if arr is None:
                skipped_short += 1
                continue
            if self.begin_date is None and self.end_date is None:
                view = arr
            else:
                tdi = arr[:, date_col].astype(np.int64)
                view = arr[int(np.searchsorted(tdi, lo, "left")) : int(np.searchsorted(tdi, hi, "right"))]
            if view.shape[0] < self.feature_len + 1:
                skipped_short += 1
                continue
            self.code_arrays[code] = view

        logger.info(
            "storage built: codes=%s skipped(short=%s)",
            len(self.code_arrays),
            skipped_short,
        )
        self.load_extra_data()

    def load_extra_data(self):
        """副表统一从 loader_npy 键表 mmap 直读（keys int64 升序 + vals float32）。

        key = tdi * 100000 + code_idx（market 无 code 轴：key = tdi）；lookup 用
        searchsorted 定位，未命中返回各表默认值（0 / traits 0.5）。
        """
        self._updown_keys, self._updown_values = load_key_table("updown")
        _market_keys, self._market_values = load_key_table("market")
        # market 无 code 轴：旧链 key 为裸 trade_date_idx（键表统一存 tdi*100000+code）
        self._market_tdis = _market_keys // 100000
        self._rank_keys, _rank_vals = load_key_table("pct_rank")
        self._rank_values = _rank_vals[:, 0]
        self._longhu_keys, self._longhu_values = load_key_table("longhu")
        self._event_keys, event_vals = load_key_table("event")
        self._trait_keys, self._trait_values = load_key_table("traits")

        _updown_cols = [c for c in UPDOWN_SUB_COLUMNS if c not in ('trade_date_idx', 'code_idx')]
        _market_cols = [c for c in MARKET_SUB_COLUMNS if c != 'trade_date_idx']
        _lh_cols = [c for c in LONGHU_SUB_COLUMNS if c not in ('trade_date_idx', 'code_idx')]
        _ev_cols = [c for c in EVENT_SUB_COLUMNS if c not in ('trade_date_idx', 'code_idx')]
        _tr_cols = [c for c in TRAIT_SUB_COLUMNS if c not in ('trade_date_idx', 'code_idx')]
        # event 键表可能携带 kpe_* 附加列（kpl 原因嵌入，供其他实验），截取契约前段
        self._event_values = event_vals[:, :len(_ev_cols)]

        self._updown_sub_cols = _updown_cols
        self._market_sub_cols = _market_cols
        self._longhu_sub_cols = _lh_cols
        self._event_sub_cols = _ev_cols
        self._trait_sub_cols = _tr_cols
        self._updown_zeros = np.zeros(len(_updown_cols), dtype=np.float32)
        self._market_zeros = np.zeros(len(_market_cols), dtype=np.float32)
        self._longhu_zeros = np.zeros(len(_lh_cols), dtype=np.float32)
        self._event_zeros = np.zeros(len(_ev_cols), dtype=np.float32)
        self._trait_default = np.full(len(_tr_cols), np.float32(0.5), dtype=np.float32)
        logger.info("extra data loaded: updown=%d, market=%d, pct_rank=%d, longhu=%d, event=%d, traits=%d", len(self._updown_keys), len(self._market_tdis), len(self._rank_keys), len(self._longhu_keys), len(self._event_keys), len(self._trait_keys))

    def lookup_pct_rank(self, tdis: np.ndarray, code_idx: int) -> np.ndarray:
        """按 (tdi, code_idx) 批量查找当日横截面涨幅分位，返回 (len(tdis),)，无记录（ST/新股等）→ 0。"""
        keys = tdis * 100000 + code_idx
        indices = np.searchsorted(self._rank_keys, keys)
        valid = indices < len(self._rank_keys)
        hit = np.zeros(len(tdis), dtype=bool)
        hit[valid] = self._rank_keys[indices[valid]] == keys[valid]
        out = np.zeros(len(tdis), dtype=np.float32)
        out[hit] = self._rank_values[indices[hit]]
        return out

    def lookup_updown(self, tdis: np.ndarray, code_idx: int) -> Tuple[np.ndarray, List[str]]:
        """按 (tdi, code_idx) 批量查找涨跌停特征，返回 (len(tdis), n_cols) + 列名。"""
        keys = tdis * 100000 + code_idx
        indices = np.searchsorted(self._updown_keys, keys)
        valid = indices < len(self._updown_keys)
        hit = np.zeros(len(tdis), dtype=bool)
        hit[valid] = self._updown_keys[indices[valid]] == keys[valid]
        out = np.tile(self._updown_zeros, (len(tdis), 1))
        if hit.any():
            out[hit] = self._updown_values[indices[hit]]
        return out, self._updown_sub_cols

    def lookup_longhu(self, tdis: np.ndarray, code_idx: int) -> Tuple[np.ndarray, List[str]]:
        """按 (tdi, code_idx) 批量查找龙虎榜特征，返回 (len(tdis), n_cols) + 列名。未上榜日全零。"""
        keys = tdis * 100000 + code_idx
        indices = np.searchsorted(self._longhu_keys, keys)
        valid = indices < len(self._longhu_keys)
        hit = np.zeros(len(tdis), dtype=bool)
        hit[valid] = self._longhu_keys[indices[valid]] == keys[valid]
        out = np.tile(self._longhu_zeros, (len(tdis), 1))
        if hit.any():
            out[hit] = self._longhu_values[indices[hit]]
        return out, self._longhu_sub_cols

    def lookup_event(self, tdis: np.ndarray, code_idx: int) -> Tuple[np.ndarray, List[str]]:
        """按 (tdi, code_idx) 批量查找事件结构特征，返回 (len(tdis), n_cols) + 列名。无事件日全零。"""
        keys = tdis * 100000 + code_idx
        indices = np.searchsorted(self._event_keys, keys)
        valid = indices < len(self._event_keys)
        hit = np.zeros(len(tdis), dtype=bool)
        hit[valid] = self._event_keys[indices[valid]] == keys[valid]
        out = np.tile(self._event_zeros, (len(tdis), 1))
        if hit.any():
            out[hit] = self._event_values[indices[hit]]
        return out, self._event_sub_cols

    def lookup_traits(self, tdis: np.ndarray, code_idx: int) -> Tuple[np.ndarray, List[str]]:
        """按 (tdi, code_idx) 批量查找股性 traits，返回 (len(tdis), n_cols) + 列名。缺失→0.5 中性。"""
        keys = tdis * 100000 + code_idx
        indices = np.searchsorted(self._trait_keys, keys)
        valid = indices < len(self._trait_keys)
        hit = np.zeros(len(tdis), dtype=bool)
        hit[valid] = self._trait_keys[indices[valid]] == keys[valid]
        out = np.tile(self._trait_default, (len(tdis), 1))
        if hit.any():
            out[hit] = self._trait_values[indices[hit]]
        return out, self._trait_sub_cols

    def lookup_market(self, tdis: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """按 tdi 批量查找全市场特征，返回 (len(tdis), n_cols) + 列名。"""
        indices = np.searchsorted(self._market_tdis, tdis)
        valid = indices < len(self._market_tdis)
        hit = np.zeros(len(tdis), dtype=bool)
        hit[valid] = self._market_tdis[indices[valid]] == tdis[valid]
        out = np.tile(self._market_zeros, (len(tdis), 1))
        if hit.any():
            out[hit] = self._market_values[indices[hit]]
        return out, self._market_sub_cols

    def _normalize_date_to_int(self, value: Optional[str]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        value_str = str(value).strip()
        if not value_str:
            return None
        try:
            date_value = datetime.strptime(value_str, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"invalid date value: {value}. expected format YYYY-MM-DD") from exc
        return int(date_value.strftime("%Y%m%d"))

    def get_next_trade_date(self, current_date: int) -> Optional[int]:
        if not current_date:
            current_date = int(self.data_meta.get("latest_date", "0").replace("-", ""))
            if not current_date:
                return None
        next_idx = bisect_right(self.trade_dates, int(current_date))
        if next_idx >= len(self.trade_dates):
            return None
        return int(self.trade_dates[next_idx])

    def date_to_idx(self, date):
        """YYYYMMDD → trade_dates 下标。支持 int / list / ndarray，未找到返回 -1。"""
        m = self._date_to_idx_map
        if isinstance(date, (int, np.integer)):
            return m.get(int(date), -1)
        if isinstance(date, np.ndarray):
            return np.array([m.get(int(d), -1) for d in date], dtype=np.int64)
        return [m.get(int(d), -1) for d in date]

    def idx_to_date(self, idx):
        """trade_dates 下标 → YYYYMMDD。支持 int / list / ndarray。"""
        t = self.trade_dates
        if isinstance(idx, (int, np.integer)):
            return t[int(idx)]
        if isinstance(idx, np.ndarray):
            return np.array([t[int(i)] for i in idx], dtype=np.int64)
        return [t[int(i)] for i in idx]

    def _date_to_idx_or_bisect(self, date_int: int) -> int:
        """YYYYMMDD → trade_dates 下标，非交易日用 bisect 找位置。"""
        idx = self._date_to_idx_map.get(date_int, -1)
        return idx if idx >= 0 else bisect_right(self.trade_dates, date_int)


    def _date_row_range(self) -> Tuple[int, int]:
        """begin/end → 存储行过滤的 [lo, hi] trade_date_idx 值域（含 feature_len+5 预热行）。"""
        lo = self._date_to_idx_or_bisect(self.begin_date) if self.begin_date else 0
        hi = self._date_to_idx_or_bisect(self.end_date) if self.end_date else len(self.trade_dates) - 1
        return max(0, lo - self.feature_len - 5), hi


class _KlineDataset(Dataset):
    """极简 Dataset：stacked 2D numpy 切片 → tensor，一次 copy。

    每个样本 = arr[end - total_len : end, :] 的连续切片，
    由 collate 后统一 to(device)。

    fake_target_tail：为「末行日期 = 数据最新日」的股票追加一条虚拟样本，
    目标日 = last_tdi+1（尚未有数据）；__getitem__ 以末行副本伪造目标行。
    特征统计与模型输入只见前 feature_len 行（target_horizon 截尾），虚拟行仅暴露
    close_tgt=close_dec（0 收益）与 target_tdi 键 ⇒ eval 循环对最后决策日自然接上。
    """

    def __init__(
        self,
        storage: _KlineDataStorage,
        target_horizon: int = 1,
        sample_ratio: float = 1.0,
        sample_seed: int = 1232,
        seed: int = 1221,
        feature_len: Optional[int] = 60,
        target_begin: Optional[int] = None,
        target_end: Optional[int] = None,
        need_m15: bool = True,
        use_pct_rank: bool = False,
        use_longhu: bool = False,
        use_event: bool = False,
        use_traits: bool = False,
        fake_target_tail: bool = False,
    ) -> None:
        self.storage = storage
        self.use_pct_rank = use_pct_rank
        self.use_longhu = use_longhu
        self.use_event = use_event
        self.use_traits = use_traits
        self._fake_tail = bool(fake_target_tail)
        self.feature_len = feature_len
        self.target_horizon = target_horizon
        self.total_len = self.feature_len + self.target_horizon
        self.seed = seed
        self.current_epoch = 0
        self.order_start = 0
        self._columns = storage.code_columns

        self._date_col_idx = (
            self._columns.index("trade_date_idx") if "trade_date_idx" in self._columns
            else self._columns.index("date") if "date" in self._columns
            else 0
        )
        self._code_col_idx = (
            self._columns.index("code_idx") if "code_idx" in self._columns else 0
        )

        # 窗口过滤用列索引（不存在则 None）
        self._is_st_idx = self._columns.index("isST") if "isST" in self._columns else None
        self._is_new_idx = self._columns.index("isNew") if "isNew" in self._columns else None
        self._m15_idx = self._columns.index("m15_complete") if "m15_complete" in self._columns else None

        # YYYYMMDD 日期范围 → trade_dates 下标范围
        lo = storage._date_to_idx_or_bisect(int(target_begin)) if target_begin else 0
        hi = storage._date_to_idx_or_bisect(int(target_end)) + 1 if target_end else len(storage.trade_dates)
        # fake 目标日的「最新数据日」= 全体票末行的最大 tdi（trade_dates 是完整日历，可能
        # 比已入库数据更靠后，故不能用 len-1）。
        last_tdi = -1
        if self._fake_tail:
            for _st in storage.code_arrays.values():
                if _st.shape[0]:
                    _d = int(_st[-1, self._date_col_idx])
                    if _d > last_tdi:
                        last_tdi = _d
        self._last_data_tdi = last_tdi

        stock_indices: List[int] = []
        intra_offsets: List[int] = []
        date_indices: List[np.ndarray] = []
        for i, (code, stacked) in enumerate(storage.code_arrays.items()):
            n = int(stacked.shape[0])
            # valid 指向窗口最后一行（包含），窗口 = [valid-total_len+1, valid+1)
            valid = np.arange(self.total_len - 1, n, dtype=np.int64)
            target_dates = stacked[valid, self._date_col_idx].astype(np.int64)
            valid = valid[(target_dates >= lo) & (target_dates < hi)]

            if valid.shape[0] == 0:
                continue

            # 窗口质量过滤：isST / isNew / m15_complete（向量化 cumsum）
            window_starts = valid - self.total_len + 1
            bad = np.zeros(n, dtype=np.int64)
            if self._is_st_idx is not None:
                bad |= (stacked[:, self._is_st_idx] != 0).astype(np.int64)
            if self._is_new_idx is not None:
                bad |= (stacked[:, self._is_new_idx] != 0).astype(np.int64)
            if need_m15 and self._m15_idx is not None:
                bad |= (stacked[:, self._m15_idx] <= 0).astype(np.int64)
            cumbad = np.concatenate(([0], np.cumsum(bad)))
            ok = (cumbad[valid + 1] - cumbad[window_starts]) == 0
            valid = valid[ok]

            # sample_ratio：建完样本池后随机丢弃
            if sample_ratio < 1.0 and valid.shape[0] > 0:
                keep = max(1, int(valid.shape[0] * sample_ratio))
                rng = np.random.RandomState(sample_seed)
                idx = rng.choice(valid.shape[0], size=keep, replace=False)
                valid = valid[idx]

            # fake 目标日：该票末行 = 最新数据日 ⇒ 虚拟样本（end=n，窗口 =
            # 最后 total_len-1 行真实 + __getitem__ 伪造目标行；质量口径 = 这 60 行，
            # 与 predict 整窗逐行同集合）。不掺入 sample_ratio：决策日截面须完整。
            # 注册门 = 虚拟目标日 last_tdi+1 落在 target 范围内（target_end=None 时
            # hi=len(日历)，日历恰止于数据末日的场景须放行）。build_loaders 传
            # target_end=next(end_date)：end_date<数据末日 ⇒ 门关 ⇒ 逐位不变、零成本。
            dates = stacked[valid, self._date_col_idx].astype(np.int64)
            if (self._fake_tail and (target_end is None or last_tdi + 1 < hi)
                    and n >= self.total_len - 1
                    and int(stacked[n - 1, self._date_col_idx]) == last_tdi
                    and cumbad[n] - cumbad[n - (self.total_len - 1)] == 0):
                valid = np.concatenate((valid, [n]))
                dates = np.concatenate((dates, [last_tdi + 1]))

            if valid.shape[0] > 0:
                stock_indices.append(np.full(valid.shape[0], i, dtype=np.int64))
                intra_offsets.append(valid)
                date_indices.append(dates)

        if stock_indices:
            self._stock_idx = np.concatenate(stock_indices)
            self._intra_offset = np.concatenate(intra_offsets)
            self._sample_dates = np.concatenate(date_indices)
        else:
            self._stock_idx = np.empty(0, dtype=np.int64)
            self._intra_offset = np.empty(0, dtype=np.int64)
            self._sample_dates = np.empty(0, dtype=np.int64)

        self._n_samples = int(self._stock_idx.shape[0])
        self._codes = list(storage.code_arrays.keys())

    def set_epoch(self, epoch: int, start_offset: int = 0) -> None:
        # 样本排列不在 dataset 上存状态：由 _epoch_order() 从 (seed, epoch) 纯函数派生，
        # 主进程 sampler 与任意 worker 进程重算结果逐位一致，天然幂等、无持久 worker 陈旧问题。
        self.current_epoch = epoch
        self.order_start = int(start_offset)

    def __len__(self) -> int:
        return self._n_samples - self.order_start

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # idx = 绝对样本位（batch_sampler 已在主进程完成 epoch 排列），worker 侧不再二次索引
        pos = int(idx)
        si = int(self._stock_idx[pos])
        end = int(self._intra_offset[pos])
        stacked = self.storage.code_arrays[self._codes[si]]
        if end >= stacked.shape[0]:        # fake-tail 虚拟样本：末 total_len-1 行真实 + 伪造目标行
            rows = [stacked[end - self.total_len + 1 : end]]
            fake = stacked[end - 1].copy()  # 目标行 = 末行（决策日）副本：close_tgt=close_dec ⇒ 0 收益
            fake[self._date_col_idx] = float(stacked[end - 1, self._date_col_idx] + 1)
            rows.append(fake[None, :])
            window = np.concatenate(rows, axis=0)  # (total_len, n_cols)
        else:
            window = stacked[end - self.total_len + 1 : end + 1]  # (feature_len + horizon, n_cols)

        window_tdis = window[:, self._date_col_idx].astype(np.int64)
        window_cidx = int(window[0, self._code_col_idx])
        all_cols = list(self._columns)
        all_arrays = [window]

        # 涨跌停
        updown_arr, updown_cols = self.storage.lookup_updown(window_tdis, window_cidx)
        all_arrays.append(updown_arr)
        all_cols.extend(updown_cols)

        # 市场表现
        market_arr, market_cols = self.storage.lookup_market(window_tdis)
        all_arrays.append(market_arr)
        all_cols.extend(market_cols)

        # 横截面涨幅分位排名（可选特征，默认关闭）
        if self.use_pct_rank:
            all_arrays.append(self.storage.lookup_pct_rank(window_tdis, window_cidx).reshape(-1, 1))
            all_cols.append("pct_rank")

        # 龙虎榜事件特征（可选，默认关闭）
        if self.use_longhu:
            longhu_arr, longhu_cols = self.storage.lookup_longhu(window_tdis, window_cidx)
            all_arrays.append(longhu_arr)
            all_cols.extend(longhu_cols)

        # 事件结构特征（可选，默认关闭）
        if self.use_event:
            event_arr, event_cols = self.storage.lookup_event(window_tdis, window_cidx)
            all_arrays.append(event_arr)
            all_cols.extend(event_cols)

        # 股性 traits（可选，默认关闭）
        if self.use_traits:
            trait_arr, trait_cols = self.storage.lookup_traits(window_tdis, window_cidx)
            all_arrays.append(trait_arr)
            all_cols.extend(trait_cols)

        merged = np.concatenate(all_arrays, axis=1)  # (total_len, all_cols)
        merged = np.nan_to_num(merged, nan=0.0, posinf=0.0, neginf=0.0)

        # 按列名返回完整窗口（含 feature + target），下游自行划分
        out: Dict[str, torch.Tensor] = {}
        for i, col_name in enumerate(all_cols):
            out[col_name] = torch.from_numpy(merged[:, i].copy())  # (total_len,)
        return out


def _epoch_order(ds: "_KlineDataset") -> np.ndarray:
    """按 (seed, epoch) 重建样本排列（绝对样本位），纯函数：任何进程任意时刻重算逐位一致。"""
    order = np.arange(ds._n_samples, dtype=np.int64)
    np.random.RandomState(ds.seed + ds.current_epoch).shuffle(order)
    return order[ds.order_start:] if ds.order_start > 0 else order


class _DateGroupBatchSampler:
    """保证每个 batch 内样本的 target_date 相同。

    三种模式（互斥优先级：sort_by_date > group_by_date > 默认）：
    - sort_by_date:   日期升序，同日 batch 连续返回
    - group_by_date:  日期随机，同日 batch 交错返回（round-robin 跨日期）
    - intra_day_repeat: 同日样本多次打乱重采样（可叠加前两者）
    """

    def __init__(
        self,
        dataset: "_KlineDataset",
        batch_size: int,
        drop_last: bool = False,
        sort_by_date: bool = False,
        group_by_date: bool = False,
        intra_day_repeat: int = 1,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if intra_day_repeat < 1:
            raise ValueError("intra_day_repeat must be >= 1")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.sort_by_date = bool(sort_by_date)
        self.group_by_date = bool(group_by_date)
        self.intra_day_repeat = int(intra_day_repeat)
        self.skip_batches: int = 0  # 续训：跳过本 epoch 已消费的 batch 数（构建批序列时消费）
        self._cache_key: Optional[Tuple[int, int]] = None
        self._cached_groups: Optional[List[Tuple[int, List[int]]]] = None
        self._batches_cache: Optional[List[List[int]]] = None
        self._batches_cache_key: Optional[Tuple[int, int]] = None

    def _get_groups(self) -> List[Tuple[int, List[int]]]:
        key = (self.dataset.current_epoch, self.dataset.order_start)
        if self._cached_groups is None or self._cache_key != key:
            self._cached_groups = self._build_groups()
            self._cache_key = key
        return self._cached_groups

    def _build_groups(self) -> List[Tuple[int, List[int]]]:
        ds = self.dataset
        dates = ds._sample_dates

        grouped: Dict[int, List[int]] = {}
        ordered_dates: List[int] = []
        for pos in _epoch_order(ds).tolist():
            d = int(dates[pos])
            if d not in grouped:
                grouped[d] = []
                ordered_dates.append(d)
            grouped[d].append(pos)
        if self.sort_by_date:
            ordered_dates.sort()
        return [(d, grouped[d]) for d in ordered_dates]

    @staticmethod
    def _split_batches_static(indices: List[int], bs: int, drop_last: bool) -> List[List[int]]:
        batches = [indices[i : i + bs] for i in range(0, len(indices), bs)]
        if drop_last:
            batches = [b for b in batches if len(b) == bs]
        return batches

    def _split_batches(self, indices: List[int]) -> List[List[int]]:
        return self._split_batches_static(indices, self.batch_size, self.drop_last)

    def _iter_group_batches(self, target_date: int, group_indices: List[int]) -> Iterator[List[int]]:
        """单日期组的 batch 迭代器，支持 intra_day_repeat。"""
        epoch = self.dataset.current_epoch
        yield from self._split_batches(group_indices)
        for repeat_idx in range(self.intra_day_repeat - 1):
            shuffled = list(group_indices)
            random.Random(epoch * 1_000_000 + int(target_date) * 1000 + repeat_idx).shuffle(shuffled)
            yield from self._split_batches(shuffled)

    def __iter__(self) -> Iterator[List[int]]:
        # 批序列按 (current_epoch, order_start) 缓存并在此立即消费 skip：DataLoader 一次
        # __iter__ 内部会对 batch_sampler 调用多次 __iter__（_BaseDataLoaderIter
        # 构造 + _reset），若 skip 写在生成器体内（惰性）会被错误的迭代器消费，
        # 导致续训整 epoch 重放。缓存后所有迭代共享同一序列，语义唯一。
        epoch = self.dataset.current_epoch
        key = (epoch, self.dataset.order_start)
        if self._batches_cache is None or self._batches_cache_key != key:
            groups = self._get_groups()
            batches: List[List[int]] = []
            if self.group_by_date:
                # 展平：每个 date 出现次数 = 它的 batch 数，整体 shuffle
                rng = random.Random(epoch + self.dataset.seed)
                for target_date, group_indices in groups:
                    batches.extend(self._iter_group_batches(target_date, group_indices))
                rng.shuffle(batches)
            else:
                batches = [b for d, gi in groups for b in self._iter_group_batches(d, gi)]
            if self.skip_batches:
                batches = batches[int(self.skip_batches):]
                self.skip_batches = 0
            self._batches_cache = batches
            self._batches_cache_key = key
        return iter(self._batches_cache)

    def __len__(self) -> int:
        batch_count = 0
        for _, group_indices in self._get_groups():
            group_size = len(group_indices)
            batches_per_pass = (
                group_size // self.batch_size
                if self.drop_last
                else (group_size + self.batch_size - 1) // self.batch_size
            )
            batch_count += batches_per_pass * self.intra_day_repeat
        return batch_count


class _OrderBatchSampler:
    """非 group 分支：batch 按 _epoch_order 切分，yield 绝对样本位（worker 无顺序状态）。"""

    def __init__(self, dataset: "_KlineDataset", batch_size: int, drop_last: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.drop_last = bool(drop_last)
        self.skip_batches: int = 0
        self._batches_cache: Optional[List[List[int]]] = None
        self._batches_cache_key: Optional[Tuple[int, int]] = None

    def __iter__(self) -> Iterator[List[int]]:
        key = (self.dataset.current_epoch, self.dataset.order_start)
        if self._batches_cache is None or self._batches_cache_key != key:
            batches = _DateGroupBatchSampler._split_batches_static(
                _epoch_order(self.dataset).tolist(), self.batch_size, self.drop_last)
            if self.skip_batches:
                batches = batches[int(self.skip_batches):]
                self.skip_batches = 0
            self._batches_cache = batches
            self._batches_cache_key = key
        return iter(self._batches_cache)

    def __len__(self) -> int:
        n = len(_epoch_order(self.dataset))
        return n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size


class _TransformCollate:
    """collate_fn：default_collate 堆叠后接可插拔 data_transform，在 worker 内执行。

    做成顶层类（而非闭包）以便 spawn 模式下 pickle；只要传入的 data_transform
    本身可 pickle（顶层函数 / functools.partial 包装顶层函数）即可跨进程复用。
    """

    def __init__(self, transform: Callable[[Any], Any]) -> None:
        self.transform = transform

    def __call__(self, samples: List[Any]) -> Any:
        return self.transform(default_collate(samples))


def build_train_and_val_dataloaders(
    config: Optional[KlineDataConfig] = None,
    feature_len: int = 60,
    batch_size: int = 64,
    target_horizon: int = 1,
    need_cols: Optional[List[str]] = None,
    begin_date: Optional[str] = None,
    end_date: Optional[str] = None,
    train_sample_ratio: float = 1.0,
    val_sample_ratio: float = 1.0,
    split_train_and_val_date: str = "20250901",
    need_m15: bool = True,
    num_workers: int = 4,
    val_batch_size: Optional[int] = None,
    val_num_workers: Optional[int] = None,
    special_codes: Optional[List[str]] = [],
    group_by_date: bool = False,
    sort_by_date: bool = False,
    intra_day_repeat: int = 1,
    data_transform: Optional[Callable[[Any], Any]] = None,
    use_pct_rank: bool = False,
    use_longhu: bool = False,
    use_event: bool = False,
    use_traits: bool = False,
    train_target_horizon: Optional[int] = None,
    train_data_transform: Optional[Callable[[Any], Any]] = None,
    val_data_transform: Optional[Callable[[Any], Any]] = None,
    train_drop_last: bool = False,
    val_sort_by_date: Optional[bool] = None,
    val_group_by_date: Optional[bool] = None,
    fake_target_tail: bool = False,
    storage_only: bool = False,
) -> Tuple[Any, Optional[DataLoader]]:
    """构建训练/验证 DataLoader —— 数据唯一出口。

    data_transform: 可插拔的 batch 级变换。传入后在 worker 进程内
    default_collate 之后执行（CPU 并行、不阻塞主循环）；
    不传则维持默认 default_collate。需 num_workers>0 才有加速效果，且 transform
    必须是可 pickle 的顶层函数（或 functools.partial 包装）。
    val_batch_size / val_num_workers: 验证集独立批次大小与 worker 数（None=沿用
    train 值）。val 是完整遍历、无梯度，瓶颈在 GPU forward，batch 调大只减 kernel
    次数不改任何指标（日期分组采样保证 daily-AUC 分组不受 batch 影响）。

    以下 7 参用于 train/val 异构采样，默认值 = 原逐位行为：
    train_target_horizon: train 侧独立 horizon（None=沿用 target_horizon）。可传
      H+1（滑窗多步样本）而 val 仍 1（单步）——两侧窗口长度可不同。
    train_data_transform / val_data_transform: 分侧 transform，覆盖 data_transform
      （None=沿用）——同一 batch 级插桩点，两侧可异构。
    train_drop_last: 仅 train 侧 sampler（用于要求 train 每批满 B 的场景）。
    val_sort_by_date / val_group_by_date: 仅 val 侧采样模式（None=跟 train 同名参）。
      val 评估若依赖批内目标日唯一（如取 `batch["target_tdi"][0]`），须
      sort_by_date 且**不**分组。
    fake_target_tail: 仅 val 侧透传 _KlineDataset（末决策日虚拟目标日）。
    storage_only: 轻量模式——只构建 storage（范围/列由 config 决定）并返回
      (storage, None)，不构建 train/val 样本池与 DataLoader。预测/分析等只需
      storage 的调用方一律走这里（数据唯一出口）。
    注意：storage 范围由 `config` 决定（须传日期=None 的全量 config，
    否则 fake_tail 的 last_tdi 判定与 eval 末日会错）；begin/end/split 三参只进
    两侧 dataset 的 target 窗口。
    """
    digit_date = lambda s: int(s.replace("-", "")) if s else None
    begin_date, end_date = digit_date(begin_date), digit_date(end_date)
    split_train_and_val_date = digit_date(split_train_and_val_date)
    cfg = config or KlineDataConfig(
        feature_len=feature_len,
        begin_date=begin_date,
        end_date=end_date,
        need_cols=need_cols,
        special_codes=special_codes
    )
    storage = _KlineDataStorage(config=cfg)
    if storage_only:
        # 统一出口的轻量取 storage 模式：跳过样本池与 DataLoader 构建。
        return storage, None
    split_date = int(split_train_and_val_date)
    next_trade_date = storage.get_next_trade_date(end_date)

    train_th = target_horizon if train_target_horizon is None else int(train_target_horizon)
    tr_tf = data_transform if train_data_transform is None else train_data_transform
    v_tf = data_transform if val_data_transform is None else val_data_transform
    v_sort = sort_by_date if val_sort_by_date is None else bool(val_sort_by_date)
    v_group = group_by_date if val_group_by_date is None else bool(val_group_by_date)

    train_ds = _KlineDataset(
        storage, target_horizon=train_th, feature_len=feature_len,
        sample_ratio=train_sample_ratio, target_begin=begin_date, target_end=split_date,
        need_m15=need_m15, use_pct_rank=use_pct_rank, use_longhu=use_longhu,
        use_event=use_event, use_traits=use_traits,
    )
    val_ds = _KlineDataset(
        storage, target_horizon=target_horizon, feature_len=feature_len,
        sample_ratio=val_sample_ratio, target_begin=split_date, target_end=next_trade_date,
        need_m15=need_m15, use_pct_rank=use_pct_rank, use_longhu=use_longhu,
        use_event=use_event, use_traits=use_traits,
        fake_target_tail=fake_target_tail,
    )

    logger.info("[train] samples=%d, [val] samples=%d", len(train_ds), len(val_ds))

    vbs = int(val_batch_size) if val_batch_size else int(batch_size)
    vnw = int(val_num_workers) if val_num_workers is not None else int(num_workers)
    cf_tr = None if tr_tf is None else _TransformCollate(tr_tf)
    cf_v = None if v_tf is None else _TransformCollate(v_tf)
    # sampler 按侧独立判定：train 仅看 train 的 sort/group，val 仅看 v_sort/v_group。
    # （两侧共用一个判定会让「train=Order × val=DateGroup」的异构组合失效；
    #  v_* 默认跟随同名参 ⇒ 与旧实现等价。）
    if sort_by_date or group_by_date:
        train_sampler = _DateGroupBatchSampler(
            train_ds, batch_size, drop_last=train_drop_last,
            sort_by_date=sort_by_date, group_by_date=group_by_date,
            intra_day_repeat=intra_day_repeat,
        )
    else:
        train_sampler = _OrderBatchSampler(train_ds, batch_size,
                                          drop_last=train_drop_last)
    if v_sort or v_group:
        val_sampler = _DateGroupBatchSampler(
            val_ds, vbs,
            sort_by_date=v_sort, group_by_date=v_group,
            intra_day_repeat=1,
        )
    else:
        val_sampler = _OrderBatchSampler(val_ds, vbs)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, num_workers=num_workers, pin_memory=True, persistent_workers=num_workers > 0, collate_fn=cf_tr)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler, num_workers=vnw, pin_memory=True, persistent_workers=vnw > 0, collate_fn=cf_v)

    return train_loader, val_loader


_FAST_KLINE_UPDOWN: Optional[pd.DataFrame] = None


def _limit_flags(tdis: np.ndarray, code_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """(trade_date_idx, code_idx) → (is_uplimit 1/-1/0 表) 涨/跌停 0/1 标记。

    权威源 = updown_limit.parquet（与存储层 lookup_updown 同表，进程内缓存）。
    """
    global _FAST_KLINE_UPDOWN
    if _FAST_KLINE_UPDOWN is None:
        from AshareData.datautils.dataloaders.feature_build.extra_features import (
            build_updown_limit_feature)
        table = build_updown_limit_feature()
        _FAST_KLINE_UPDOWN = table[["trade_date_idx", "code_idx", "is_limit_up"]].dropna()
    code_rows = _FAST_KLINE_UPDOWN[_FAST_KLINE_UPDOWN["code_idx"] == int(code_idx)]
    flags = code_rows.set_index("trade_date_idx")["is_limit_up"].reindex(tdis).fillna(0.0).to_numpy()
    return (flags > 0.5).astype(np.float64), (flags < -0.5).astype(np.float64)


def load_single_stock_kline(
    code: str,
    seq_len: int = 60,
    last_date: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """纯看K线：单股行情快照（展示专用快速读，读分页 npy，不做窗口/质量过滤）。

    返回末 seq_len 行（末行日期 ≤ last_date）的 12 列 OHLCVT 展示表；不构建
    storage/样本池，涨跌停标记只查 updown_limit 表（不加载其余 extra 特征）。
    模型输入严禁走此函数，一切样本仍必须经 build_train_and_val_dataloaders。
    """
    code = str(code).strip()
    if not code:
        return None
    arr = _load_paged_code(code, list(full_cols()))
    if arr is None:
        return None
    full = full_cols()
    j = [full.index(c) for c in ("trade_date_idx", "code_idx", "open", "high",
                                 "low", "close", "preclose", "volume", "turn",
                                 "pctChg", "isST")]
    cal = np.asarray([int(d) for d in get_trade_date_list()], dtype=np.int64)
    tdis = arr[:, j[0]].astype(np.int64)
    sel = str(last_date).strip().replace("-", "").replace("/", "")[:8] if last_date else ""
    if len(sel) == 8 and sel.isdigit():
        keep = tdis < int(np.searchsorted(cal, int(sel), side="right"))
    else:
        keep = np.ones(len(tdis), dtype=bool)
    rows = np.nonzero(keep)[0]
    rows = rows[max(0, len(rows) - max(1, int(seq_len))):]
    tdis = tdis[rows]
    is_up, is_dn = _limit_flags(tdis, int(arr[0, j[1]]))
    return pd.DataFrame({
        "date": pd.to_datetime(cal[tdis].astype(str), format="%Y%m%d", errors="coerce"),
        "open": arr[rows, j[2]].astype(np.float64), "high": arr[rows, j[3]].astype(np.float64),
        "low": arr[rows, j[4]].astype(np.float64), "close": arr[rows, j[5]].astype(np.float64),
        "volume": arr[rows, j[7]].astype(np.float64), "turnoverRate": arr[rows, j[8]].astype(np.float64),
        "preclose": arr[rows, j[6]].astype(np.float64), "pctChg": arr[rows, j[9]].astype(np.float64),
        "is_uplimit": is_up, "is_downlimit": is_dn,
        "isST": arr[rows, j[10]].astype(np.float64),
    }).reset_index(drop=True)


if __name__ == "__main__":
    train_loader, val_loader = build_train_and_val_dataloaders(batch_size=64, num_workers=4)
    for batch in tqdm.tqdm(train_loader):
        pass