# -*- coding: utf-8 -*-
"""
baostock_session.py
~~~~~~~~~~~~~~~~~~~
baostock 快速会话封装：login 时用漏斗策略挑选低延迟连接。

漏斗规则（串行 login/logout，无复杂构造）
  第一轮：测 5 个，取首个 login RTT < 700 ms 的
  第二轮：若第一轮未命中，再测 5 个，取首个 < 1000 ms 的
  兜底  ：若两轮均未命中，最后再 login 一次直接使用
  强制模式（force_fast=True）：无限轮询直到找到 < 700 ms 的连接

用法::

    # fast_login 直接管理全局 baostock 会话
    from AshareData.utils.kline_data_utils.baostock_session import fast_login
    fast_login()                        # 普通漏斗
    fast_login(force_fast=True)         # 强制找到 < 700 ms 的节点

    # QueryKlineUtils 已内置 force_fast 选速，直接使用即可
    from AshareData.utils.kline_data_utils.query_kline_utils import QueryKlineUtils
    query = QueryKlineUtils()
    df, msg = query.query_history('sz.000001', start_date='2026-01-01', end_date='2026-06-30')
"""

import contextlib
import datetime
import io
import time

import baostock as bs

from AshareData.utils.log_util import get_logger

logger = get_logger('baostock-session')


@contextlib.contextmanager
def _quiet():
    """屏蔽 baostock login/logout 的 print 输出。"""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def _do_login():
    """执行一次 login，返回 (elapsed_ms, success)。"""
    t0 = time.perf_counter()
    with _quiet():
        result = bs.login()
    ms = (time.perf_counter() - t0) * 1000
    ok = (result.error_code == '0')
    return ms, ok


def _do_logout():
    with _quiet():
        bs.logout()


# ---------------------------------------------------------------------------
# Benchmark query helpers
# ---------------------------------------------------------------------------
_BENCH_CODE  = 'sz.000001'   # 流动性强，数据完整
_BENCH_FIELD = 'date,close'
_BENCH_FREQ  = 'd'


def _bench_dates():
    """取最近 14 个日历天作为基准查询区间（跨越若干交易日，保证有数据）。"""
    today = datetime.date.today()
    end   = (today - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    start = (today - datetime.timedelta(days=14)).strftime('%Y-%m-%d')
    return start, end


def _measure_avg_query_ms(n: int = 3) -> float:
    """
    对当前 baostock 会话执行 n 次轻量查询，返回平均耗时（ms）。
    任意请求失败返回 inf。
    """
    start, end = _bench_dates()
    times = []
    for _ in range(n):
        try:
            t0 = time.perf_counter()
            rs = bs.query_history_k_data_plus(
                _BENCH_CODE, _BENCH_FIELD,
                start_date=start, end_date=end,
                frequency=_BENCH_FREQ, adjustflag='2',
            )
            while rs.error_code == '0' and rs.next():
                rs.get_row_data()
            elapsed = (time.perf_counter() - t0) * 1000
            times.append(elapsed if rs.error_code == '0' else float('inf'))
        except Exception:
            times.append(float('inf'))
    return sum(times) / len(times) if times else float('inf')


def fast_login(
    n_first: int = 5,
    threshold_fast_ms: float = 700,
    n_second: int = 5,
    threshold_slow_ms: float = 1000,
    force_fast: bool = False,
    n_bench_queries: int = 3,
    max_force_attempts: int = 0,
) -> float:
    """
    串行 login/logout 漏斗筛选，以登录后的平均查询耗时作为决策依据。

    每个候选会话的评估流程：login → 执行 n_bench_queries 次轻量查询取均值
    → 若均值 < 阈值则保留；否则 logout 换下一个。

    Parameters
    ----------
    n_first           : 第一轮尝试次数（默认 5）
    threshold_fast_ms : 第一轮查询均值阈值 ms（默认 700）
    n_second          : 第二轮尝试次数（默认 5）
    threshold_slow_ms : 第二轮查询均值阈值 ms（默认 1000）
    force_fast        : 若为 True，无限轮询直到查询均值 < threshold_fast_ms
    n_bench_queries   : 每个会话的基准查询次数（默认 3）
    max_force_attempts: force_fast 模式下最大尝试次数，0 表示无限循环（默认 0）

    Returns
    -------
    float : 最终选中会话的平均查询耗时（ms）
    """

    def _try(label: str, idx: int, threshold: float):
        """Login，测速，若满足阈值返回 (avg_ms, True)，否则 logout 返回 (avg_ms, False)。"""
        login_ms, ok = _do_login()
        if not ok:
            logger.info(f'[fast_login] {label} #{idx}: login 失败 ({login_ms:.0f} ms)')
            _do_logout()
            return float('inf'), False
        avg_ms = _measure_avg_query_ms(n_bench_queries)
        logger.info(
            f'[fast_login] {label} #{idx}: '
            f'login={login_ms:.0f} ms, 查询均值={avg_ms:.0f} ms  (阈值 {threshold:.0f} ms)'
        )
        if avg_ms < threshold:
            logger.info(f'[fast_login] ✓ 选中 #{idx} ({label}): 查询均值={avg_ms:.0f} ms')
            return avg_ms, True
        _do_logout()
        return avg_ms, False

    # ---- 强制模式：无限轮询直到 < threshold_fast_ms ----
    if force_fast:
        total = 0
        while True:
            total += 1
            avg_ms, accepted = _try('强制', total, threshold_fast_ms)
            if accepted:
                return avg_ms
            if max_force_attempts > 0 and total >= max_force_attempts:
                logger.warning(
                    f'[fast_login] 强制模式已达最大尝试次数 {max_force_attempts}，'
                    f'末次查询均值={avg_ms:.0f} ms，直接使用当前连接'
                )
                # 最后一次 _try 已 logout，需重新 login
                _do_login()
                return avg_ms

    # ---- 普通漏斗模式 ----
    rounds = [
        (n_first,  threshold_fast_ms, '第一轮'),
        (n_second, threshold_slow_ms, '第二轮'),
    ]
    total = 0
    for n, threshold, label in rounds:
        for _ in range(n):
            total += 1
            avg_ms, accepted = _try(label, total, threshold)
            if accepted:
                return avg_ms

    # 兜底：再 login 一次直接使用
    total += 1
    login_ms, ok = _do_login()
    avg_ms = _measure_avg_query_ms(n_bench_queries) if ok else float('inf')
    logger.info(
        f'[fast_login] 兜底 #{total}: '
        f'login={login_ms:.0f} ms, 查询均值={avg_ms:.0f} ms  (两轮均未命中，直接使用)'
    )
    return avg_ms


if __name__ == '__main__':
    fast_login(force_fast=True)