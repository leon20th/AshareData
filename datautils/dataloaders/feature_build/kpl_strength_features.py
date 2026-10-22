"""KPL 强度特征（题材强度 + 题材内强度）→ event 键表 ks_* 列（2026-10-21）。

和 kp_*（事件当日快照）的区别：
  1. 题材级（entity × day）：th_* —— 含"今天这个方向还有几家涨停/最高板/晋级率/封单"，
     通过 **最近 4 个交易日内的 asof 携带** 贴到非事件行上（kp_* 在这里全是 0）。
  2. 个股级（仅事件当日）：st_* —— 个股在同题材内的位置与质量（板数分位/首封/封单/距上次上榜）。
  3. ks_days：距最近一次事件的天数（0=今天，1..4=携带，4=更久/无）。

产物：并回 `built_data/event_feat.parquet`（ks_* 列）+ 重写 loader 键表 'event'（列序 = 契约序）。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

from AshareData.paths import BUILT_DATA_DIR
from AshareData.utils.exchanges_utils import a_open

BUILT = f"{BUILT_DATA_DIR}/misc"

KS_COLUMNS = ['ks_days', 'st_boards', 'st_rank', 'st_gapmax', 'st_is_leader', 'st_first',
              'st_seal', 'st_open', 'st_hist20', 'st_lastgap',
              'th_max_b', 'th_age', 'th_seal', 'th_first', 'th_promote']
ST_COLS = ['st_boards', 'st_rank', 'st_gapmax', 'st_is_leader', 'st_first', 'st_seal',
           'st_open', 'st_hist20', 'st_lastgap']
TH_COLS = ['th_max_b', 'th_age', 'th_seal', 'th_first', 'th_promote']
CARRY = 4          # 事件后携带天数


def build_ks() -> pd.DataFrame:
    """→ (trade_date_idx, code_idx, ks_*)；覆盖事件当日与后 4 个交易日的携带行。"""
    up = pd.read_parquet(f"{BUILT}/updown_limit.parquet")
    up["date"] = up.date.astype(str)
    up["code"] = up.code.astype(str)
    up = up[up.is_limit_up == 1]
    kp = pd.read_parquet(f"{BUILT}/kpl_events.parquet")
    kp["date"] = kp.date.astype(str)
    kp["code"] = kp.code.astype(str)
    kp = kp[(kp.date >= "20190101") & (kp.zscode.astype(str) != "0")]
    mem = pd.read_parquet(f"{BUILT}/kpl_theme_members.parquet")
    mem["date"] = mem.date.astype(str)
    mem["code"] = mem.code.astype(str)
    mem = mem[(mem.entity_id > 0) & (mem.primary)].drop_duplicates(["date", "code"])

    up = up.assign(di=up.date.map(lambda d: a_open.get_trade_date_idx(d)))

    ev = (mem[["date", "code", "code_idx", "entity_id"]]
          .merge(kp[["date", "code", "boards"]].drop_duplicates(["date", "code"]),
                 on=["date", "code"], how="left")
          .merge(up[["date", "code", "consecutive_days", "first_limit_min", "seal_pct",
                     "open_count", "boards"]].rename(columns={"boards": "up_boards"}),
                 on=["date", "code"], how="left"))
    ev["boards"] = ev.boards.fillna(ev.up_boards).fillna(1).astype(float)
    ev["di"] = ev.date.map(lambda d: a_open.get_trade_date_idx(d))
    ev = ev.sort_values(["di", "entity_id", "code"]).reset_index(drop=True)
    # 携带行只能落在「数据实际存在」的日期内。交易日历是预生成到 2026-12-31 的（len(tds)=8797），
    # 拿 len(tds) 当上界会往未来投影出空壳行（2026-10-22 审计：tdi 8747~8750 共 302 行，只有 ks_days 非零）。
    di_max = int(ev.di.max())

    # ---- 题材强度（entity × day）----
    th = (ev.groupby(["date", "entity_id"], sort=False)
            .agg(th_max_b=("boards", "max"), th_seal=("seal_pct", "mean"),
                 th_first=("first_limit_min", "mean"), di=("di", "first"),
                 th_n=("code", "size")).reset_index())
    th = th.sort_values(["entity_id", "di"]).reset_index(drop=True)
    prev_di = th.groupby("entity_id").di.shift(1)
    step = th.di - prev_di
    run = (~(step <= 3).fillna(False)).cumsum()
    th["th_age"] = th.groupby(["entity_id", run]).cumcount().astype(float)

    # 昨日晋级率
    up_set = up[["di", "code"]].drop_duplicates()
    prev_map = th[["entity_id", "di", "th_n"]].copy()
    prev_map["di"] += 1
    prev_map = prev_map.rename(columns={"th_n": "prev_n"})
    ev_prev = ev[["entity_id", "code", "di"]].copy()
    ev_prev["di"] += 1
    promo = (ev_prev.merge(up_set, on=["di", "code"], how="inner")
             .groupby(["entity_id", "di"]).size().rename("promoted").reset_index()
             .merge(prev_map, on=["entity_id", "di"], how="left"))
    promo["th_promote"] = np.where(promo.prev_n > 0, promo.promoted / promo.prev_n, 0.0)
    th = th.merge(promo[["entity_id", "di", "th_promote"]], on=["entity_id", "di"],
                  how="left")
    th["th_promote"] = th.th_promote.fillna(0.0)
    th["th_max_b"] = np.minimum(th.th_max_b, 10) / 10.0
    th["th_age"] = np.minimum(th.th_age, 20) / 20.0
    th["th_seal"] = np.log1p(th.th_seal.fillna(0)) / 10.0
    th["th_first"] = th.th_first.fillna(240) / 240.0
    th_use = th[["entity_id", "di"] + TH_COLS]

    # ---- 个股题材内强度（事件当日）----
    st = ev.sort_values(["di", "entity_id", "code"]).reset_index(drop=True)
    n = st.groupby(["date", "entity_id"]).code.transform("size")
    mx = st.groupby(["date", "entity_id"]).boards.transform("max")
    rk = st.groupby(["date", "entity_id"]).boards.rank(method="min")
    st["st_boards"] = np.minimum(st.boards, 10) / 10.0
    st["st_rank"] = ((rk - 1) / (n - 1)).where(n > 1, 0.5)
    st["st_gapmax"] = (mx - st.boards) / 10.0
    st["st_is_leader"] = (st.boards >= mx).astype(float)
    st["st_first"] = st.first_limit_min.fillna(240) / 240.0
    st["st_seal"] = np.log1p(st.seal_pct.fillna(0)) / 10.0
    st["st_open"] = st.open_count.fillna(0) / 10.0
    st["st_lastgap"] = np.minimum(st.groupby(["entity_id", "code"]).di.diff()
                                  .fillna(999), 60) / 60.0
    st["one"] = 1.0
    st["st_hist20"] = (st.groupby(["entity_id", "code"]).one
                       .transform(lambda s: s.shift(1).rolling(20, min_periods=1).sum())
                       .fillna(0))
    st["st_hist20"] = np.minimum(st.st_hist20, 10) / 10.0

    ev_row = st[["di", "code_idx", "entity_id"] + ST_COLS].copy()
    ev_row = ev_row.merge(th_use, on=["entity_id", "di"], how="left")
    ev_row["ks_days"] = 0.0

    # ---- 携带行（事件后 1..4 个交易日；st 归零，只留题材状态 + 天数）----
    base = st[["di", "code_idx", "entity_id"]].copy()
    carry = []
    for k in range(1, CARRY + 1):
        c = base.copy()
        c["di"] = c.di + k
        c = c[c.di <= di_max]
        c = c.merge(th_use, on=["entity_id", "di"], how="left")
        for col in ST_COLS:
            c[col] = 0.0
        c["ks_days"] = k / CARRY
        carry.append(c[["di", "code_idx"] + KS_COLUMNS])
    ev_rows = ev_row[["di", "code_idx"] + KS_COLUMNS]
    out = pd.concat([ev_rows] + carry, ignore_index=True)
    # 同一 (di, code_idx) 可能既是事件行又是别人的携带行 → 保留 ks_days 最小（事件优先）
    out = (out.sort_values(["di", "code_idx", "ks_days"])
              .drop_duplicates(["di", "code_idx"], keep="first")
              .reset_index(drop=True))
    out = out.rename(columns={"di": "trade_date_idx"})
    out[KS_COLUMNS] = out[KS_COLUMNS].fillna(0.0).astype(np.float32)
    print(f"[ks] 行数 {len(out):,}（事件 {len(ev_rows):,} + 携带 {len(out) - len(ev_rows):,}）"
          f" 日期 {out.trade_date_idx.min()}..{out.trade_date_idx.max()}", flush=True)
    return out


def merge_into_event_table(force: bool = False) -> int:
    """ks_* 并入 event 键表并按契约列序重写 loader 键表。"""
    from AshareData.datautils.dataloaders.feature_build.loader_tables import (
        load_key_table, read_key_table_df, save_key_table)
    from AshareData.datautils.dataloaders.feature_build.model_ready import (
        EVENT_SUB_COLUMNS)

    feat = read_key_table_df("event")
    # 已有 ks 产物 = 前缀 ks_* ∪ KS_COLUMNS（键表里列名是 st_*/th_*，没有 ks_ 前缀；
    # 只按前缀判断会让 --force 二次运行时与新 ks 冲突成 _x/_y ⇒ KeyError）
    old = [c for c in feat.columns if c.startswith("ks_") or c in KS_COLUMNS]
    if old and not force:
        print(f"[ks] event 键表已有 ks 列（{len(old)}），跳过（force 可重建）", flush=True)
        return len(feat)
    feat = feat[[c for c in feat.columns if c not in set(old)]].copy()
    ks = build_ks()
    out = feat.merge(ks, on=["trade_date_idx", "code_idx"], how="outer")
    for c in KS_COLUMNS:
        out[c] = out[c].fillna(0.0)
    for c in [c for c in feat.columns if c != "trade_date_idx" and c != "code_idx"]:
        out[c] = out[c].fillna(0.0)

    # 键表不得引用「数据尚不存在」的日期：交易日历是预生成到年底的，旧版 build_ks 拿
    # len(tds) 当上界，把携带行投影到未来（tdi 8747~8750 共 302 行，只有 ks_days 非零，
    # 2026-10-22 审计）。outer join 会把旧键表里的这些空壳行留下来，所以这里按 data
    # horizon（market 键表最后一天）再剪一次。
    horizon = int(load_key_table("market")[0][-1]) // 100000
    beyond = out.trade_date_idx > horizon
    if beyond.any():
        print(f"[ks] 剔除 {int(beyond.sum())} 行超出数据地平线（tdi>{horizon}，空壳行）",
              flush=True)
        out = out[~beyond]
    # 无效 code_idx：-1 = 代码不在股票池（退市股，250 个代码 / 1 万行成员记录）；
    # 99999 = 历史 -1 经 key 编码回读出来的「前一天」伪下标（全零空壳）。
    bad_code = (out.code_idx < 0) | (out.code_idx >= 99999)
    if bad_code.any():
        print(f"[ks] 剔除 {int(bad_code.sum())} 行无效 code_idx（-1 未识别 / 99999 伪下标）",
              flush=True)
        out = out[~bad_code]

    # 契约列序：ev_* → kp_* → ks_* → 其余（kpe_*）
    order = [c for c in EVENT_SUB_COLUMNS if c in out.columns]
    rest = [c for c in out.columns if c not in order]
    cols = order + rest
    out = out[cols].sort_values(["trade_date_idx", "code_idx"]).reset_index(drop=True)
    n = save_key_table("event", out, cols, default=0.0, source="event_feat+ks(20261021)")
    out.to_parquet(f"{BUILT}/event_feat.parquet", index=False)
    print(f"[ks] event 键表重写 {n:,} 行，列 {len(cols)}（契约 {len(order)}）", flush=True)
    return n


if __name__ == "__main__":
    merge_into_event_table(force="--force" in sys.argv)
