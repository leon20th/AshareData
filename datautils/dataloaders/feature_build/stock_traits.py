# stock_traits —— 个股"股性"长历史统计（B/C 族）→ built_data/stock_traits.parquet
# 数据源：base_feature 日线（全历史）+ updown_limit.parquet（涨停质量）+ market_daily_stats.parquet（beta 基准）
# 口径：涨停/跌停 = close 触板（round(preclose*(1±price_limit),2)）；rel=(pctChg/100)/price_limit；
#       长统计 shift(1) 防泄漏；输出 = 逐日横截面升序 rank01（0=最小，1=最大），空值→0.5（中性）。
# 列（14）：
#   t_mv   流通市值（=100*amount/turn，用户恒等式已核验 ~1bp）
#   t_lu120/t_ld120/t_bu120/t_bd120  120 日涨停/跌停/大涨/大跌次数（大涨: rel≥0.7）
#   t_sru  封板率 lu120/bu120（bu120≥5 才有效）/ t_srd 跌封率
#   t_dsl  距上次涨停天数（cap 250，从未涨停=250）/ t_hi 距 250 日高点 / t_lo 距 250 日低点
#   t_yz   一字率先验（过去 120 涨停中首封 0 分钟且未开板占比，≥20 个才有效）/ t_zb 炸板率
#   t_turn 120 日均换手 / t_beta 对市场 avg_pctChg 的 120 日 beta（≥60 天）
# 重建式（全量重算，实测 ~10s）；无跳过逻辑——调用即重建。
import glob
import time

import polars as pl

from AshareData.paths import BASE_FEATURE_DIR, BUILT_DATA_DIR

TRAIT_PARQUET = f"{BUILT_DATA_DIR}/stock_traits.parquet"
TRAIT_COLS = ["t_mv", "t_lu120", "t_ld120", "t_bu120", "t_bd120", "t_sru", "t_srd",
              "t_dsl", "t_hi", "t_lo", "t_yz", "t_zb", "t_turn", "t_beta"]
_RAW = {"t_mv": "fv", "t_lu120": "lu120", "t_ld120": "ld120", "t_bu120": "bu120",
        "t_bd120": "bd120", "t_sru": "sru", "t_srd": "srd", "t_dsl": "dsl",
        "t_hi": "hi", "t_lo": "lo", "t_yz": "yz120", "t_zb": "zb120",
        "t_turn": "turn120", "t_beta": "beta120"}


def build():
    """全量重建 stock_traits.parquet（重建式；调用即重建，不设跳过）。"""
    t0 = time.time()
    i8 = pl.Int8
    files = sorted(glob.glob(f"{BASE_FEATURE_DIR}/*.parquet"))
    d = pl.scan_parquet(files).select([
        "date", "trade_date_idx", "code_idx", "close", "preclose",
        "price_limit", "pctChg", "turn", "amount",
    ]).collect()
    d = (d.with_columns(trade_date_idx=pl.col("trade_date_idx").cast(pl.Int64),
                        code_idx=pl.col("code_idx").cast(pl.Int64))
         .sort(["code_idx", "trade_date_idx"]))
    d = d.with_columns(
        is_lu=((pl.col("close") - (pl.col("preclose") * (1 + pl.col("price_limit"))).round(2))
               .abs() < 0.001) & (pl.col("price_limit") >= 0.04),
        is_ld=((pl.col("close") - (pl.col("preclose") * (1 - pl.col("price_limit"))).round(2))
               .abs() < 0.001) & (pl.col("price_limit") >= 0.04),
        rel=pl.col("pctChg") / 100 / pl.col("price_limit"),
    )
    d = d.with_columns(
        big_up=((pl.col("rel") >= 0.7) & (pl.col("price_limit") >= 0.04)).cast(i8),
        big_dn=((pl.col("rel") <= -0.7) & (pl.col("price_limit") >= 0.04)).cast(i8),
        fv=pl.when((pl.col("turn") > 0.05) & (pl.col("amount") > 0))
        .then(100 * pl.col("amount") / pl.col("turn")).otherwise(None),
        t_lu=pl.when(pl.col("is_lu")).then(pl.col("trade_date_idx")).otherwise(None),
    )
    d = d.with_columns(
        t_lu=pl.col("t_lu").forward_fill().over("code_idx"),
        lu120=pl.col("is_lu").cast(i8).fill_null(0).shift(1)
        .rolling_sum(window_size=120, min_samples=1).over("code_idx"),
        ld120=pl.col("is_ld").cast(i8).fill_null(0).shift(1)
        .rolling_sum(window_size=120, min_samples=1).over("code_idx"),
        bu120=pl.col("big_up").fill_null(0).shift(1)
        .rolling_sum(window_size=120, min_samples=1).over("code_idx"),
        bd120=pl.col("big_dn").fill_null(0).shift(1)
        .rolling_sum(window_size=120, min_samples=1).over("code_idx"),
        turn120=pl.col("turn").shift(1).rolling_mean(window_size=120, min_samples=60).over("code_idx"),
        rmax=pl.col("close").rolling_max(window_size=250, min_samples=120).over("code_idx"),
        rmin=pl.col("close").rolling_min(window_size=250, min_samples=120).over("code_idx"),
    )
    d = d.with_columns(
        dsl=(pl.col("trade_date_idx") - pl.col("t_lu")).clip(0, 250).fill_null(250),
        hi=pl.col("close") / pl.col("rmax"),
        lo=pl.col("close") / pl.col("rmin"),
        sru=pl.when(pl.col("bu120") >= 5).then(pl.col("lu120") / pl.col("bu120")).otherwise(None),
        srd=pl.when(pl.col("bd120") >= 5).then(pl.col("ld120") / pl.col("bd120")).otherwise(None),
    )
    mkt = (pl.read_parquet(f"{BUILT_DATA_DIR}/market_daily_stats.parquet")
           .select(["trade_date_idx", "avg_pctChg"])
           .with_columns(trade_date_idx=pl.col("trade_date_idx").cast(pl.Int64)))
    d = d.join(mkt, on="trade_date_idx", how="left")
    d = d.with_columns(xy=pl.col("pctChg") * pl.col("avg_pctChg"),
                       y2=pl.col("avg_pctChg") ** 2)
    for nm, src in [("mx", "pctChg"), ("my", "avg_pctChg"), ("mxy", "xy"), ("my2", "y2")]:
        d = d.with_columns(**{nm: pl.col(src).shift(1)
                           .rolling_mean(window_size=120, min_samples=60).over("code_idx")})
    d = d.with_columns(beta120=(pl.col("mxy") - pl.col("mx") * pl.col("my"))
                       / (pl.col("my2") - pl.col("my") ** 2))
    up = (pl.read_parquet(f"{BUILT_DATA_DIR}/updown_limit.parquet")
          .filter(pl.col("is_limit_up") == 1)
          .select(["trade_date_idx", "code_idx", "open_count", "first_limit_min"])
          .with_columns(trade_date_idx=pl.col("trade_date_idx").cast(pl.Int64),
                        code_idx=pl.col("code_idx").cast(pl.Int64))
          .sort(["code_idx", "trade_date_idx"]))
    up = up.with_columns(
        zb=(pl.col("open_count") > 0).cast(i8),
        yz=((pl.col("first_limit_min") < 1) & (pl.col("open_count") == 0)).cast(i8),
    )
    up = up.with_columns(
        zb120=pl.col("zb").shift(1).rolling_mean(window_size=120, min_samples=20).over("code_idx"),
        yz120=pl.col("yz").shift(1).rolling_mean(window_size=120, min_samples=20).over("code_idx"),
    )
    d = d.join(up.select(["trade_date_idx", "code_idx", "zb120", "yz120"]),
               on=["trade_date_idx", "code_idx"], how="left")
    d = d.with_columns(zb120=pl.col("zb120").forward_fill().over("code_idx"),
                       yz120=pl.col("yz120").forward_fill().over("code_idx"))
    out = d.select(["trade_date_idx", "code_idx"]
                   + [pl.col(v).alias(k) for k, v in _RAW.items()])
    del d
    for k in _RAW:
        n = pl.col(k).count().over("trade_date_idx").cast(pl.Float64)
        out = out.with_columns(
            ((pl.col(k).rank("average").over("trade_date_idx") - 1)
             / pl.max_horizontal(n - 1, pl.lit(1.0))).alias(k))
    out = out.with_columns([pl.col(c).fill_null(0.5).cast(pl.Float32) for c in TRAIT_COLS])
    out = out.sort(["trade_date_idx", "code_idx"])
    out.write_parquet(TRAIT_PARQUET)
    print(f"[traits] {out.shape} → {TRAIT_PARQUET} ({time.time()-t0:.0f}s) "
          f"tdi {int(out['trade_date_idx'].min())}-{int(out['trade_date_idx'].max())}")
    return out


if __name__ == "__main__":
    df = build()
    pl.Config.set_tbl_rows(20)
    print(df.head(3))
    s = df.filter(pl.col("trade_date_idx") >= int(df["trade_date_idx"].max()) - 250)
    smp = s.select(TRAIT_COLS).sample(n=min(200_000, s.height), seed=0).to_pandas()
    corr = smp.corr()
    pairs = [(corr.columns[i], corr.columns[j], float(corr.iloc[i, j]))
             for i in range(len(TRAIT_COLS)) for j in range(i + 1, len(TRAIT_COLS))
             if abs(corr.iloc[i, j]) > 0.7]
    print(f"\n|rho|>0.7 共 {len(pairs)} 对：")
    for a, b, r in sorted(pairs, key=lambda x: -abs(x[2])):
        print(f"  {a:8s} x {b:8s} rho={r:+.2f}")
