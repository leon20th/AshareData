# -*- coding: utf-8 -*-
"""quick_kline —— 日K v2 构建器（字段自维护：缺口自动判定、手段自动选择）

设计原则（2026-09-26 定稿）
  1. 每个字段只对外暴露一个 `update_<字段>` 接口：该补哪一段、用哪种手段，全在接口内部判断。
     调用方不需要、也不应该知道 rebuild / backfill / update 的区别——模式概念已从接口移除。
  2. 接口内按「每只票 × 每个字段」的缺口选手段：

     情形                     手段
     ---------------------   ---------------------------------------------------------------
     无本地数据（整段历史）    OHLCV: 扶摇 daily-k 全量 dump
                             m15:   旧器 update_kline.py（baostock 15 分钟全史）
                             isST:  baostock 回扫（有旧库则优先旧库移植）
     缺历史中间段              OHLCV: 扶摇 daily-k-10d（更早的缺口回落全量 dump）
                             m15:   新浪 getKLineData（≤1023 根 ≈ 64 交易日窗，超窗报不可回补）
                             isST:  baostock 只补缺段（逐票 checkpoint，可续跑）
                             turn:  新浪股本事件 as-of 回填
     只缺当天                  OHLCV: 扶摇 prices/snapshot（批量 100）
                             turn:  扶摇 auction/snapshot（竞价快照）
                             isST:  扶摇名单比对（ST 前缀变化）
     都不缺                    直接跳过（幂等：重复跑不重复抓）

     缺口起点由「上市日」决定：新股从上市日起算，上市前的空档不算缺口。
  3. 旧库（daily_kline）存在时，其 turn / pctChg / isST 作为历史权威值覆盖（移植）；
     没有旧库则自动回落 baostock / 新浪，无需任何参数。

产物: daily_kline_v2/{sh.600000.csv} 13 列（与旧库同 schema，OHLCV 为未复权原值）
辅助: daily_kline_v2/_aux/、_dumps/，以及 m15_kline/
用法:
  python AshareData/datautils/kline_scripts/quick_kline.py [--codes 600519,sz.000001] [--skip m15,isst]
"""
import argparse
import io
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
import requests

from AshareData.paths import ASHARE_ROOT, DAILY_KLINE_DIR, DAILY_KLINE_V2_DIR, M15_KLINE_DIR
from AshareData.utils.exchanges_utils.a_open import get_target_trade_date, get_trade_date_list
from AshareData.utils.log_util import get_logger
from AshareData.utils.read_file_utils import read_last_lines

logger = get_logger('日Kv2')

BASE = 'https://fuyao.aicubes.cn'
KEY_FILE = os.path.join(ASHARE_ROOT, '.keys', '.fuyao_api.json')
V2_DIR = DAILY_KLINE_V2_DIR
AUX_DIR = os.path.join(V2_DIR, '_aux')
DUMPS_DIR = os.path.join(V2_DIR, '_dumps')
ADJ_F = os.path.join(AUX_DIR, 'adjust_factors.parquet')
SHARE_F = os.path.join(AUX_DIR, 'share_events.parquet')
META_F = os.path.join(AUX_DIR, 'meta_tickers.parquet')
NAMES_F = os.path.join(AUX_DIR, 'name_snapshots.parquet')
PREVCLOSE_F = os.path.join(AUX_DIR, 'prev_close.parquet')
SHARE_LOG_F = os.path.join(AUX_DIR, 'share_fetch_log.json')
ISST_DIR = os.path.join(AUX_DIR, 'isst')
DELISTED_F = os.path.join(AUX_DIR, 'delisted.parquet')

COLS = ['date', 'code', 'open', 'high', 'low', 'close', 'preclose',
        'volume', 'amount', 'turn', 'tradestatus', 'pctChg', 'isST']
M15_COLS = ['date', 'time', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
FLOOR = '2020-01-02'                     # 库起点（与旧库/消费链对齐）
SINA_KLINE = 'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData'
SINA_SHARE = ('https://stock.finance.sina.com.cn/stock/api/jsonp.php/'
              'var%20KKE_ShareAmount_{sym}=/StockService.getAmountBySymbol?_=20&symbol={sym}')
SINA_MAX_BARS = 1023                     # 新浪单次请求上限（≈64 个交易日的 15 分钟 bar）
BARS_PER_DAY = 16

_CAL = None
_TLS = threading.local()


# ==================== 通用工具（与字段无关） ====================

def _cal():
    """全部交易日（'YYYY-MM-DD'，含已发布的前瞻日期）。"""
    global _CAL
    if _CAL is None:
        _CAL = [f'{d[:4]}-{d[4:6]}-{d[6:8]}' for d in get_trade_date_list()]
    return _CAL


def _sess():
    key = next(v for v in json.load(open(KEY_FILE)).values() if isinstance(v, str) and len(v) > 10)
    s = requests.Session()
    s.headers['X-api-key'] = key
    return s


def api_json(path, params=None, timeout=60):
    r = _sess().get(BASE + path, params=params, timeout=timeout)
    j = r.json()
    if j.get('code') != 0:
        raise RuntimeError(f'{path} -> code={j.get("code")} msg={j.get("message")}')
    return j.get('data')


def ensure_dump(name, max_age_h=6, force=False):
    """扶摇市场 dump：带缓存的下载（_dumps/，含重试）。"""
    path = os.path.join(DUMPS_DIR, f'{name}.parquet')
    if os.path.exists(path) and not force and (time.time() - os.path.getmtime(path)) < max_age_h * 3600:
        return pd.read_parquet(path)
    os.makedirs(DUMPS_DIR, exist_ok=True)
    last = None
    for _ in range(4):
        try:
            url = api_json(f'/api/dump/market-dumps/{name}/download-url')['presigned_url']
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                buf = io.BytesIO(b''.join(r.iter_content(1 << 20)))
            df = pd.read_parquet(buf)
            df.to_parquet(path)
            logger.info(f'[dump] 下载 {name}: {len(df)} 行')
            return df
        except Exception as e:
            last = e
            time.sleep(2)
    raise RuntimeError(f'{name} 下载失败: {last}')


def _to_local(d):
    """扶摇 thscode(.SH/.SZ) → 本地码 sh./sz.；date_ms → 北京日期。"""
    d = d[d['thscode'].str.endswith(('.SH', '.SZ'))].copy()
    d['code'] = [f'{x.split(".")[1].lower()}.{x.split(".")[0]}' for x in d['thscode']]
    ms = 'date_ms' if 'date_ms' in d.columns else 'ex_date_ms'
    d['date'] = pd.to_datetime(d[ms], unit='ms', utc=True).dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d')
    return d


def norm_codes(text):
    """'600519,sz.000001' → ['sh.600519', 'sz.000001']。"""
    out = []
    for tok in re.split(r'[,\s]+', text or ''):
        tok = tok.strip()
        if not tok:
            continue
        if '.' in tok:
            a, b = tok.lower().split('.')
            if a.isdigit():          # '600000.sh' / '600000.SH'
                a, b = b, a
            if a not in ('sh', 'sz', 'bj') or not b.isdigit():
                raise ValueError(f'无法识别代码: {tok}')
            code = f'{a}.{b}'
        elif tok.startswith('6'):
            code = f'sh.{tok}'
        elif tok.startswith(('0', '2', '3')):
            code = f'sz.{tok}'
        else:
            raise ValueError(f'无法识别代码: {tok}')
        out.append(code)
    return sorted(set(out))


def read_v2(code):
    p = os.path.join(V2_DIR, f'{code}.csv')
    if not os.path.exists(p):
        return pd.DataFrame(columns=COLS)
    return pd.read_csv(p, dtype={'date': str})


def write_v2(code, df):
    df = df.reindex(columns=COLS).sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
    os.makedirs(V2_DIR, exist_ok=True)
    df.to_csv(os.path.join(V2_DIR, f'{code}.csv'), index=False)


def _is_st_name(name):
    return bool(re.match(r'^\*?ST', str(name or '').strip().upper()))


def _num(v):
    """CSV 字段 → float；空/nan/非数字 → None（'1' 与 '1.0' 都算 1）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _dur(sec):
    """秒 → 45s / 3m20s / 1h05m。"""
    sec = int(sec)
    if sec < 60:
        return f'{sec}s'
    if sec < 3600:
        return f'{sec // 60}m{sec % 60:02d}s'
    return f'{sec // 3600}h{sec % 3600 // 60:02d}m'


def _progress(i, n, t0, tag, extra='', every=500):
    """粗粒度进度：每轮迭代开头调用（i 从 1 起），首尾必打、每 every 只打一行。

    只为消除长时间静默：带已用与 ETA，不做花哨刷新，被管道捕获后同样可读。
    """
    if i != 1 and i != n and i % every:
        return
    el = time.time() - t0
    tail = f' ETA {_dur(el / (i - 1) * (n - i + 1))}' if 1 < i < n else ''
    logger.info(f'{tag} {i}/{n}' + (f' {extra}' if extra else '') + f' 已用 {_dur(el)}{tail}')


def _fail(tag, code, err, failed, cap=3):
    """逐票失败：前 cap 条详列，之后只计数（末尾汇总），避免整屏都是失败行。"""
    failed.append(code)
    if len(failed) <= cap:
        logger.info(f'{tag} {code} 失败: {err}')
    elif len(failed) == cap + 1:
        logger.info(f'{tag} 失败已超 {cap} 条，后续只计数')


class KlineV2Builder:
    """字段自维护构建器：对外只有 update_* 接口，缺口判定与手段选择全在内部。

    统一签名 update_<字段>(codes=None)：codes 为 None 时取 meta 全市场沪深。
    """

    def __init__(self, force=False):
        self.force = force                 # 强制重取（忽略"已齐/已抓过"的跳过判断）
        self._bs_logged = False
        self._prev_close_mem = {}
        self._list_dates = None

    # 字段执行顺序（--skip 的取值也来自这里）
    FIELDS = (('adjust', '复权事件'), ('ohlcvt', 'OHLCV+停牌占位'), ('preclose', '前收/涨跌幅'),
              ('turn', '换手率'), ('isst', 'isST'), ('m15', 'm15 分钟线'))

    # ==================== 对外：字段接口 ====================

    def update_meta(self, codes=None):
        """代码表 / 名称 / 上市日（扶摇 meta，1 组请求）+ 退市表维护。其余字段的"新股判定"依赖它。"""
        prev = pd.read_parquet(META_F) if os.path.exists(META_F) else None
        items, off = [], 0
        while True:
            d = api_json('/api/meta/tickers/list', {'asset_type': 'a-share', 'limit': 10000, 'offset': off})
            its = d.get('item') or []
            items += its
            if len(its) < 10000:
                break
            off += 10000
        df = pd.DataFrame([{'thscode': it['thscode'], 'name': it.get('name'),
                            'list_date': it.get('list_date'), 'exchange': it.get('exchange')} for it in items])
        os.makedirs(AUX_DIR, exist_ok=True)
        df.to_parquet(META_F)
        self._sync_delisted(prev, df)
        logger.info(f'[meta] {len(df)} 只')

    def update_adjust(self, codes=None):
        """复权事件表（扶摇 adjustment-factors dump）：preclose 的除权参考价依赖它。"""
        d = _to_local(ensure_dump('adjustment-factors'))
        d = d[['code', 'date', 'dividend_per_share', 'per_share_bonus', 'allotment_ratio', 'allotment_price']]
        d = d.rename(columns={'date': 'ex_date'}).sort_values(['code', 'ex_date'])
        os.makedirs(AUX_DIR, exist_ok=True)
        d.to_parquet(ADJ_F)
        logger.info(f'[adjust] {len(d)} 事件 / {d["code"].nunique()} 只')

    def update_ohlcvt(self, codes=None):
        """OHLCV+额（含停牌占位行与 tradestatus）：按缺口自动选 全量 dump / 增量 dump / 当天快照。"""
        codes = self._codes(codes)
        target = self._target_date()
        full, recent, today, repair = [], [], [], []
        for code in codes:
            df = read_v2(code)
            if df.empty:
                full.append(code)                     # 无本地数据 → 整段历史
                continue
            miss = self._missing(code, df)
            if [d for d in miss if d < target]:
                recent.append(code)                   # 缺历史中间段
            if target in miss or self._stale_today(code):
                today.append(code)                    # 缺当天 / 当天行是收盘前写的
            if self._bad_today(code):
                repair.append(code)                   # 当天行有量无价 → 无论快照回不回都要重建
        if not (full or recent or today or repair):
            logger.info('[ohlcvt] 各票均已齐 → 跳过')
            return
        logger.info(f'[ohlcvt] 全史 {len(full)} 只 / 历史缺口 {len(recent)} 只 / 当天 {len(today)} 只'
                    + (f' / 待重建当天行 {len(repair)} 只' if repair else ''))
        src = {}
        if full:
            src.update(self._fetch_history(full))
        if recent:
            gaps = {c: [d for d in self._missing(c, read_v2(c)) if d < target] for c in recent}
            src.update(self._fetch_recent(gaps))
        if today:
            src.update(self._fetch_today(today))
        for code in repair:
            src.setdefault(code, (None, target))       # 无真成交也要落一行（占位），否则当天永远缺行
        n, t0 = 0, time.time()
        # 源里没有这些天行情（长期停牌、库起点前）时仍需落行：按“每交易日一行”补占位行
        todo = [c for c in codes if c in src] + [c for c in recent if c not in src]
        for i, code in enumerate(todo, 1):
            _progress(i, len(todo), t0, '[ohlcvt]', f'已写 {n}')
            rows, market_last = src.get(code, (None, None))
            df = read_v2(code)
            if rows is not None and len(rows):
                rows = rows.copy()
                rows['code'] = code
                for col in ('preclose', 'turn', 'pctChg', 'isST'):
                    rows[col] = np.nan
                # 空骨架帧不进 concat（pandas 官方做法：concat 前排除空/全 NA 帧）
                df = rows if df.empty else pd.concat([df, rows], ignore_index=True)
            if df.empty:
                continue
            write_v2(code, self._materialize(df, market_last, code))
            n += 1
        logger.info(f'[ohlcvt] 写 {n} 只')

    def update_preclose(self, codes=None):
        """preclose/pctChg：纯本地派生（依赖 adjust 事件表），旧库有值的历史行以旧库为准。"""
        codes = self._codes(codes)
        prev_map = self._load_prev_close()
        need_pre = [c for c in codes if c not in prev_map and self._first_traded_blank(c, 'preclose')]
        if need_pre:
            logger.info(f'[preclose] {len(need_pre)} 只缺首行前收 → 取全量 dump 库起点前收盘')
            self._remember_prev_close(self._dump_daily('daily-k'), need_pre)
            prev_map = self._load_prev_close()
        adj = pd.read_parquet(ADJ_F) if os.path.exists(ADJ_F) else pd.DataFrame(
            columns=['code', 'ex_date', 'dividend_per_share', 'per_share_bonus', 'allotment_ratio', 'allotment_price'])
        n, t0 = 0, time.time()
        for i, code in enumerate(codes, 1):
            _progress(i, len(codes), t0, '[preclose]', f'已写 {n}')
            df = read_v2(code)
            if df.empty:
                continue
            ev = adj[adj['code'] == code]
            g = ev.groupby('ex_date').agg(D=('dividend_per_share', 'sum'), B=('per_share_bonus', 'sum'),
                                          R=('allotment_ratio', 'sum')).reset_index()
            ev2 = ev.copy()
            ev2['Rp'] = ev2['allotment_ratio'].fillna(0) * ev2['allotment_price'].fillna(0)
            g = g.merge(ev2.groupby('ex_date')['Rp'].sum().reset_index(), on='ex_date', how='left')
            f_of = {e['ex_date']: (e['D'], e['B'], e['R'], e['Rp']) for _, e in g.iterrows()}
            ev_dates = list(f_of)
            cal = _cal()
            last_close, last_mark = None, None
            pre = np.full(len(df), np.nan)
            pch = np.full(len(df), np.nan)
            dates = df['date'].tolist()
            closes = pd.to_numeric(df['close'], errors='coerce').to_numpy()
            vols = pd.to_numeric(df['volume'], errors='coerce').to_numpy()
            for j in range(len(df)):
                if not np.isfinite(vols[j]):        # 停牌占位行：preclose 保持 carry
                    pre[j] = closes[j]
                    continue
                d = dates[j]
                if last_close is None:
                    base = prev_map.get(code, np.nan)
                    k = cal.index(d) if d in cal else 0
                    lo = cal[max(0, k - 1)] if cal else ''
                else:
                    base, lo = last_close, last_mark
                f, applied, seg = 1.0, False, []
                for ed in ev_dates:
                    if ed > lo and ed <= d:
                        seg.append(f_of[ed])
                        D, B, R, Rp = f_of[ed]
                        if base and np.isfinite(base) and base > 0:
                            f *= (base - D + Rp) / (base * (1 + B + R))
                            applied = True
                pc = base * f if (base is not None and np.isfinite(base)) else np.nan
                if applied and np.isfinite(pc):
                    # 交易所惯例：除权参考价四舍五入到 0.01（半格边界须十进制 half-up，float 银行家舍入会错）
                    if len(seg) == 1:
                        D, B, R, Rp = seg[0]
                        pc = float((Decimal(str(base)) - Decimal(str(D)) + Decimal(str(Rp)))
                                   / (Decimal(1) + Decimal(str(B)) + Decimal(str(R))))
                    pc = float(Decimal(str(pc)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                pre[j] = pc
                if np.isfinite(pc) and pc > 0 and np.isfinite(closes[j]):
                    pch[j] = (closes[j] / pc - 1.0) * 100.0
                last_close, last_mark = closes[j], d
            df['preclose'] = pre
            df['pctChg'] = pch
            overlay = self._old_col(code, 'pctChg', df['date'])   # 旧库历史值权威
            if overlay is not None:
                df['pctChg'] = np.where(overlay.notna(), overlay, df['pctChg'])
            write_v2(code, df)
            n += 1
        logger.info(f'[preclose] 写 {n} 只')

    def update_turn(self, codes=None):
        """换手率：历史段=新浪股本事件 as-of；当天=扶摇竞价快照优先；旧库有值则覆盖。"""
        codes = self._codes(codes)
        target = self._target_date()
        self._fetch_share_events(codes)                    # 每天只抓一次（share_fetch_log）
        events = self._load_share_events()
        need_snap = []
        for code in codes:
            df = read_v2(code)
            if df.empty or target not in set(df['date']):
                continue
            row = df.loc[df['date'] == target, 'turn']
            if len(row) and (_num(row.iloc[0]) is None or self._stale_today(code)):
                need_snap.append(code)
        snap = self._auction_snapshot(need_snap) if need_snap else {}
        logger.info(f'[turn] 股本事件 {len(events)} 条；需当天快照 {len(need_snap)} 只')
        refetch, n, t0 = [], 0, time.time()
        for i, code in enumerate(codes, 1):
            _progress(i, len(codes), t0, '[turn]', f'已写 {n}')
            df = read_v2(code)
            if df.empty:
                continue
            ev = events[events['code'] == code].sort_values('date')
            if not ev.empty:
                idx = np.searchsorted(ev['date'].to_numpy(), df['date'].to_numpy(), side='right') - 1
                shares = np.where(idx >= 0, ev['shares_wan'].to_numpy()[np.clip(idx, 0, None)], np.nan)
                vols = pd.to_numeric(df['volume'], errors='coerce').to_numpy()
                df['turn'] = vols / (shares * 1e4) * 100
            it = snap.get(code)
            if it and it.get('last_price') and it.get('float_market_cap'):
                shares = it['float_market_cap'] / it['last_price']     # 股
                if shares > 0:
                    idx = df.index[df['date'] == target]
                    vol = pd.to_numeric(df.loc[idx, 'volume'], errors='coerce').iloc[0]
                    df.loc[idx, 'turn'] = vol / shares * 100 if np.isfinite(vol) else np.nan
                    if len(ev):                                        # 快照股本与事件表偏差过大 → 下轮重抓
                        last_wan = float(ev['shares_wan'].iloc[-1])
                        if last_wan > 0 and abs(shares / 1e4 - last_wan) / last_wan > 0.005:
                            refetch.append(code)
            overlay = self._old_col(code, 'turn', df['date'])
            if overlay is not None:
                df['turn'] = np.where(overlay.notna(), overlay, df['turn'])
            write_v2(code, df)
            n += 1
        if refetch:
            logger.info(f'[turn] 快照股本与事件表偏差 → 下轮重抓 {len(refetch)} 只: {refetch[:8]}')
        logger.info(f'[turn] 写 {n} 只')

    def update_isst(self, codes=None):
        """isST：历史段=旧库移植（无则 baostock 只补缺段）；当天行=扶摇名单比对。按票缺口分档。"""
        codes = self._codes(codes)
        target = self._target_date()
        m = self._meta()
        m = m[m['thscode'].str.endswith(('.SH', '.SZ'))]
        code2name = {f"{x.split('.')[1].lower()}.{x.split('.')[0]}": n for x, n in zip(m['thscode'], m['name'])}
        self._snap_names(code2name)                        # 每次运行落一条名单快照，供改名判定
        scan, fix_local = [], []
        for code in codes:
            df = read_v2(code)
            if df.empty:
                continue
            vals = pd.to_numeric(df['isST'], errors='coerce')
            if not vals.isna().any():
                continue                                   # 整列已完整 → 不碰
            (fix_local if vals.notna().any() else scan).append(code)
        if scan:
            logger.info(f'[isst] 整列缺 → 回扫 {len(scan)} 只（逐票 checkpoint，可续跑）')
        n, t0, failed = 0, time.time(), []
        for i, code in enumerate(scan, 1):
            _progress(i, len(scan), t0, '[isst] 回扫', f'已写 {n} 失败 {len(failed)}', every=50)
            try:
                s = self._scan_isst(code)
            except Exception as e:
                _fail('[isst]', code, e, failed)
                continue
            df = read_v2(code)
            if df.empty:
                continue
            full = s.set_index('date')['isST']
            df['isST'] = pd.to_numeric(df['date'].map(full), errors='coerce').ffill().fillna(0).astype(int)
            write_v2(code, df)
            n += 1
        # 停牌/首行空缺：本地 ffill 补齐即可（无网络）；与 rebuild 路径同规矩，整列归一化为 0/1
        touched = fix_local + [c for c in scan if c not in failed]
        filled = 0
        for code in touched:
            df = read_v2(code)
            if df.empty:
                continue
            col = pd.to_numeric(df['isST'], errors='coerce')
            if not col.isna().any():
                continue
            df['isST'] = col.ffill().fillna(0).astype(int)
            write_v2(code, df)
            filled += 1
        filled += self._isst_by_name(touched, code2name, target)
        logger.info(f'[isst] 回扫 {n} 只，本地补齐 {filled} 只，失败 {len(failed)} 只（失败可重跑续传）')

    def update_m15(self, codes=None):
        """15 分钟线：无本地文件 → 旧器 update_kline.py（baostock 全史）；有文件缺段 → 新浪（≤64 交易日窗）。"""
        codes = self._codes(codes)
        cal, last_td = _cal(), self._target_date()
        history, sina, unhealed = [], {}, []
        for code in codes:
            p = os.path.join(M15_KLINE_DIR, f'{code}.csv')
            if self.force or not os.path.exists(p):
                history.append(code)
                continue
            try:
                plan = self._plan_m15(code, cal, last_td)
            except Exception as e:
                unhealed.append(code)
                _fail('[m15]', code, e, unhealed)
                continue
            if isinstance(plan, tuple):
                sina[code] = plan
            elif plan == 'old':
                unhealed.append(code)
        if history:
            logger.info(f'[m15] 无本地文件 {len(history)} 只 → 旧器 update_kline.py 全史')
            self._delegate_history(history)
        if unhealed:
            logger.info(f'[m15] {len(unhealed)} 只老缺口早于新浪窗，需 --force 重建（本机用旧器）')
        logger.info(f'[m15] 需抓 {len(sina)}/{len(codes)} 只（last_td={last_td}）')
        rows_by_code, failed = {}, []
        done, t0 = 0, time.time()
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(self._fetch_m15, c, dl): (c, fm) for c, (dl, fm) in sina.items()}
            for f in as_completed(futs):
                done += 1
                c, fm = futs[f]
                _progress(done, len(futs), t0, '[m15] 抓取', f'已得 {len(rows_by_code)} 失败 {len(failed)}')
                try:
                    rows_by_code[c] = (f.result(), fm)
                except Exception as e:
                    _fail('[m15]', c, e, failed)
        nw = 0
        for code, (rows, fm) in rows_by_code.items():
            if rows is None or rows.empty:
                continue
            if self._m15_upsert(code, rows, force_merge=fm):
                nw += 1
        logger.info(f'[m15] 新浪更新 {nw}/{len(codes)} 只，抓取失败 {len(failed)} 只')

    def run(self, codes=None, skip=()):
        """按固定顺序跑各字段接口（每个接口自己判断缺口与手段）。"""
        t0 = time.time()
        self.update_meta()                                  # 上市日/退市表：缺口判定依赖它
        codes = self._codes(codes)
        logger.info(f'== {len(codes)} 只 | 目标日 {self._target_date()} ==')
        for i, (name, desc) in enumerate(self.FIELDS, 1):
            if name in skip:
                logger.info(f'== [{i}/{len(self.FIELDS)}] 跳过 {name} ==')
                continue
            t = time.time()
            getattr(self, f'update_{name}')(codes)
            logger.info(f'== [{i}/{len(self.FIELDS)}] {name}（{desc}）完成，用时 {_dur(time.time() - t)} ==')
        logger.info(f'== 完成，共 {_dur(time.time() - t0)} ==')

    # ==================== 内部：缺口判定（全部"该补什么"的判断收敛在此） ====================

    def _codes(self, codes=None):
        if codes:
            return sorted(codes)
        return sorted(f"{x.split('.')[1].lower()}.{x.split('.')[0]}"
                      for x in self._meta()['thscode'] if x.endswith(('.SH', '.SZ')))

    def _meta(self):
        return pd.read_parquet(META_F)

    def _list_dates_map(self):
        if self._list_dates is None:
            m = self._meta()
            m = m[m['thscode'].str.endswith(('.SH', '.SZ'))]
            self._list_dates = {}
            for x, v in zip(m['thscode'], m['list_date']):
                d = str(v)[:10] if v and str(v) != 'nan' else None
                self._list_dates[f"{x.split('.')[1].lower()}.{x.split('.')[0]}"] = d
        return self._list_dates

    def _start_of(self, code):
        """该票应覆盖的首个交易日 = max(库起点, 上市日)：新股从上市日算，上市前不算缺口。

        无上市日（扶摇名单里的预留/未开板代码，如 301718.SZ）→ None：这类源里一行行情都没有，
        不能按库起点要求它，否则会凭空造出一整段停牌占位行。
        """
        ld = self._list_dates_map().get(code)
        return max(FLOOR, ld) if ld else None

    def _missing(self, code, df):
        """[上市日或库起点, 目标日] 内本地缺失的交易日；无本地数据则[]。"""
        if df.empty:
            return []
        have = set(df['date'])
        lo = self._start_of(code) or df['date'].iloc[0]     # 无上市日 → 不要求已有数据之前的日子
        tgt = self._target_date()
        return [d for d in _cal() if lo <= d <= tgt and d not in have]

    def _first_traded_blank(self, code, col):
        """首个真成交行的某字段是否为空（字段级缺口：如缺起始基准、缺原始数据）。"""
        df = read_v2(code)
        if df.empty:
            return False
        t = df[df['volume'].notna()]
        return bool(len(t)) and bool(pd.to_numeric(t[col], errors='coerce').isna().iloc[0])

    def _stale_today(self, code):
        """当天行是否是收盘前写的（收盘后需要重新覆盖）。"""
        now = pd.Timestamp.now()
        today = now.strftime('%Y-%m-%d')
        if self._target_date() != today or now.strftime('%H:%M') < '15:00':
            return False
        p = os.path.join(V2_DIR, f'{code}.csv')
        if not os.path.exists(p):
            return False
        return os.path.getmtime(p) < pd.Timestamp(f'{today} 15:00').timestamp()

    def _bad_today(self, code):
        """目标日行不可信：有量无价（停牌票被当天快照写成了成交行）→ 要重取/改判为停牌占位。"""
        df = read_v2(code)
        if df.empty:
            return False
        r = df[df['date'] == self._target_date()]
        if not len(r):
            return False
        r = r.iloc[-1]
        return _num(r['volume']) is not None and _num(r['close']) is None

    def _target_date(self):
        """目标交易日：get_target_trade_date（交易日 15:00 前取上一交易日）。"""
        t = get_target_trade_date()
        return pd.to_datetime(t).strftime('%Y-%m-%d') if t else pd.Timestamp.now().strftime('%Y-%m-%d')

    # ==================== 内部：手段（各数据源） ====================

    def _fetch_history(self, codes):
        """整段历史：扶摇 daily-k 全量 dump（一次下载，按票切片）→ {code: (rows, market_last)}。"""
        d = self._dump_daily('daily-k')
        market_last = d['date'].max()
        sub = d[d['code'].isin(codes)]
        self._remember_prev_close(d, codes)            # 首行前收：preclose 的起始基准
        sub = sub[sub['date'] >= FLOOR]
        return {c: (g.drop(columns='code'), market_last) for c, g in sub.groupby('code')}

    def _remember_prev_close(self, dump, codes):
        """记录这些票在库起点前的最后一个收盘（preclose 起始基准），合并写入 aux（不覆盖别的票）。"""
        sub = dump[(dump['code'].isin(codes)) & (dump['date'] < FLOOR)]
        if not len(sub):
            return
        prev = sub.sort_values('date').groupby('code')['close'].last()
        self._prev_close_mem.update(prev.to_dict())
        old = (pd.read_parquet(PREVCLOSE_F) if os.path.exists(PREVCLOSE_F)
               else pd.DataFrame(columns=['code', 'prev_close']))
        both = pd.concat([old, pd.DataFrame({'code': list(prev.index), 'prev_close': prev.values})],
                         ignore_index=True).drop_duplicates('code', keep='last')
        os.makedirs(AUX_DIR, exist_ok=True)
        both.to_parquet(PREVCLOSE_F)

    def _fetch_recent(self, gaps):
        """历史缺口：10d dump 覆盖窗口内直接补；更早的缺口回落到全量 dump。"""
        d10 = self._dump_daily('daily-k-10d')
        win_min = d10['date'].min() if len(d10) else '9999'
        out, older = {}, {c: [x for x in dates if x < win_min] for c, dates in gaps.items()}
        for code, dates in gaps.items():
            d = d10[(d10['code'] == code) & (d10['date'].isin(dates))]
            if len(d):
                out[code] = d
        older = {c: v for c, v in older.items() if v}
        if older:
            logger.info(f'[ohlcvt] {len(older)} 只缺口早于 10 日窗口 → 回落全量 dump')
            full = self._dump_daily('daily-k')
            self._remember_prev_close(full, list(older))      # 补早段历史 → 首行前收也要补上
            for code, dates in older.items():
                d = full[(full['code'] == code) & (full['date'].isin(dates))]
                out[code] = d if code not in out else pd.concat([out[code], d], ignore_index=True)
        miss = sum(len(v) for v in gaps.values())
        got = sum(len(v) for v in out.values())
        if miss > got:
            logger.info(f'[ohlcvt] 缺口 {miss} 个票日，源只覆盖 {got} 个（其余下轮再试）')
        tgt = self._target_date()
        return {c: (d.drop(columns='code'), tgt) for c, d in out.items() if len(d)}

    def _dump_daily(self, name):
        d = _to_local(ensure_dump(name))
        d = d.rename(columns={'open_price': 'open', 'high_price': 'high', 'low_price': 'low',
                              'close_price': 'close', 'turnover': 'amount'})
        if name == 'daily-k':
            d = d[d['adjusted'] == 'none']
        return d[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]

    def _fetch_today(self, codes):
        """只缺当天：扶摇 prices/snapshot（批量 100）。"""
        target = self._target_date()
        out = {}
        for it in self._batch_snapshot('/api/a-share/prices/snapshot', codes):
            code = f"{it['thscode'].split('.')[1].lower()}.{it['thscode'].split('.')[0]}"
            if code in codes:
                out[code] = (pd.DataFrame([{'date': target, 'open': it.get('open_price'),
                                            'high': it.get('high_price'), 'low': it.get('low_price'),
                                            'close': it.get('last_price'), 'volume': it.get('volume'),
                                            'amount': it.get('turnover')}]), target)
        return out

    # ==================== 内部：行网格 / 停牌占位 ====================

    def _materialize(self, df, market_last, code=None):
        """补齐 [max(库起点, 上市日), market_last] 的交易日网格：缺日 → 停牌占位行（OHLC=前收，量额空，tradestatus=0）。

        传 code 时左边界取 _start_of(code)（库起点/上市日），这样库起点附近缺行也能补成占位行。
        """
        df = df.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
        if df.empty:
            return df
        # 有量无价的行不是真成交（当天快照对停牌票写的 0 行）→ 丢掉，按缺行补成停牌占位行
        bad = df['volume'].notna() & pd.to_numeric(df['close'], errors='coerce').isna()
        if bad.any() and (~bad).any():
            df = df[~bad].reset_index(drop=True)
        cal = _cal()
        d0 = df['date'].iloc[0]
        if code:
            s = self._start_of(code)
            d0 = min(d0, s) if s else d0
        d1 = min(market_last, cal[-1]) if market_last else df['date'].iloc[-1]
        if d1 < d0:
            d1 = df['date'].iloc[-1]
        need = [x for x in cal if d0 <= x <= d1]
        missing = [x for x in need if x not in set(df['date'])]
        if missing:
            real = df[df['volume'].notna()].sort_values('date')
            if len(real):
                pos = np.searchsorted(real['date'].to_numpy(), np.array(missing), side='left') - 1
                rcs = pd.to_numeric(real['close'], errors='coerce').to_numpy()
                carry = np.where(pos >= 0, rcs[np.clip(pos, 0, None)], np.nan)  # 逐缺口前收（多段停牌各自 carry）
            else:
                carry = np.full(len(missing), np.nan)
            place = pd.DataFrame({'date': missing, 'code': df['code'].iloc[0],
                                  'open': carry, 'high': carry, 'low': carry, 'close': carry,
                                  'preclose': carry, 'volume': np.nan, 'amount': np.nan,
                                  'turn': np.nan, 'pctChg': np.nan, 'tradestatus': 0, 'isST': np.nan})
            df = pd.concat([df, place], ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)
        df.loc[df['volume'].notna(), 'tradestatus'] = 1
        df.loc[df['volume'].isna(), 'tradestatus'] = 0
        df['tradestatus'] = df['tradestatus'].astype(int)   # 0/1 标志列显式定型，不随 concat 推断变 float
        return df

    def _batch_snapshot(self, path, codes):
        items = []
        for i in range(0, len(codes), 100):
            ts = ','.join(f"{c.split('.')[1]}.{c.split('.')[0].upper()}" for c in codes[i:i + 100])
            items += api_json(path, {'thscodes': ts}).get('item') or []
        return items

    def _auction_snapshot(self, codes):
        out = {}
        for it in self._batch_snapshot('/api/a-share/auction/snapshot', codes):
            code = f"{it['thscode'].split('.')[1].lower()}.{it['thscode'].split('.')[0]}"
            out[code] = it
        return out

    # ==================== 内部：股本事件 ====================

    def _load_share_events(self):
        if os.path.exists(SHARE_F):
            return pd.read_parquet(SHARE_F)
        return pd.DataFrame(columns=['code', 'date', 'shares_wan'])

    def _fetch_share_events(self, codes):
        """新浪流通股本事件：每票一个 jsonp 全量事件表；每天只抓一次（share_fetch_log）。"""
        os.makedirs(AUX_DIR, exist_ok=True)
        log = json.load(open(SHARE_LOG_F)) if os.path.exists(SHARE_LOG_F) else {}
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        ev_all = self._load_share_events()
        todo = [c for c in codes if self.force or log.get(c) != today]
        if not todo:
            return
        tls = threading.local()

        def fetch(code):
            s = getattr(tls, 'sess', None)
            if s is None:
                s = tls.sess = requests.Session()
                s.headers['User-Agent'] = 'Mozilla/5.0'
            sym = code.replace('.', '')
            err = None
            for _ in range(2):
                try:
                    r = s.get(SINA_SHARE.format(sym=sym), timeout=15)
                    m = re.search(r'\((\[.*\])\)', r.content.decode('gbk', 'replace'), re.S)
                    arr = json.loads(m.group(1)) if m else []
                    time.sleep(random.uniform(0.02, 0.06))
                    return code, [{'code': code, 'date': pd.to_datetime(it['date']).strftime('%Y-%m-%d'),
                                   'shares_wan': float(it['amount'])} for it in arr], None
                except Exception as e:
                    err = e
                    time.sleep(0.5)
            return code, None, err

        new, done, failed = [], 0, []
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=6) as ex:
            for code, rows, err in ex.map(fetch, todo):
                done += 1
                _progress(done, len(todo), t0, '[share]', f'新增 {len(new)} 条 失败 {len(failed)} 只')
                if err is not None:
                    _fail('[share]', code, err, failed)
                else:
                    log[code] = today
                    new += rows
        if new:
            new = pd.DataFrame(new)
            # 空累加帧不进 concat（pandas 官方做法：concat 前排除空/全 NA 帧）
            ev_all = new if ev_all.empty else pd.concat([ev_all, new], ignore_index=True)
            ev_all = ev_all.drop_duplicates(['code', 'date'], keep='last').sort_values(['code', 'date'])
            ev_all.to_parquet(SHARE_F)
        json.dump(log, open(SHARE_LOG_F, 'w'))
        logger.info(f'[share] 新增 {len(new)} 条事件（累计 {len(ev_all)}），失败 {len(failed)} 只')

    # ==================== 内部：isST ====================

    def _scan_isst(self, code):
        """该票 isST 全史（旧库优先，无则 baostock 回扫；逐票 checkpoint 可续跑）。"""
        p = os.path.join(DAILY_KLINE_DIR, f'{code}.csv')
        if os.path.exists(p):
            df = pd.read_csv(p, dtype={'date': str})
            return df[['date', 'isST']]
        import baostock as bs
        os.makedirs(ISST_DIR, exist_ok=True)
        path = os.path.join(ISST_DIR, f'{code}.parquet')
        have = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=['date', 'isST'])
        have = have.astype({'date': str})
        tgt_end = self._target_date()
        segs = []
        if len(have):
            lo, hi = str(have['date'].min()), str(have['date'].max())
            if hi < tgt_end:
                segs.append(((pd.to_datetime(hi) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), tgt_end))
            if lo > FLOOR:
                segs.append((FLOOR, (pd.to_datetime(lo) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')))
        else:
            segs.append((FLOOR, tgt_end))
        if not segs:
            return have
        if not self._bs_logged:
            bs.login()
            self._bs_logged = True
        new = pd.DataFrame(columns=['date', 'isST'])
        for qs, qe in segs:
            rows = None
            for attempt in range(3):
                try:
                    rs = bs.query_history_k_data_plus(code, 'date,isST', start_date=qs, end_date=qe,
                                                      frequency='d', adjustflag='3')
                    rows = []
                    while rs.error_code == '0' and rs.next():
                        rows.append(rs.get_row_data())
                    if rs.error_code != '0':
                        raise RuntimeError(rs.error_msg)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(3)
                    bs.login()
            df = pd.DataFrame(rows, columns=['date', 'isST'])
            df['isST'] = pd.to_numeric(df['isST'], errors='coerce')
            new = df if new.empty else pd.concat([new, df], ignore_index=True)
        full = pd.concat([have, new], ignore_index=True) if len(have) else new
        full = full.drop_duplicates('date', keep='last').sort_values('date')
        full.to_parquet(path)
        return full

    def _snap_names(self, code2name):
        """落一条今天的名字快照（供"改名/ST 前缀变化"判定）。"""
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        hist = pd.read_parquet(NAMES_F) if os.path.exists(NAMES_F) else pd.DataFrame(columns=['code', 'date', 'name'])
        snap = pd.DataFrame([{'code': c, 'date': today, 'name': n} for c, n in code2name.items()])
        hist = pd.concat([hist, snap], ignore_index=True).drop_duplicates(['code', 'date'], keep='last')
        os.makedirs(AUX_DIR, exist_ok=True)
        hist.to_parquet(NAMES_F)

    def _isst_by_name(self, codes, code2name, target):
        """当天行以扶摇名单为准（ST 前缀变化的票记一行日志）；已正确则不动文件。"""
        hist = pd.read_parquet(NAMES_F) if os.path.exists(NAMES_F) else pd.DataFrame(columns=['code', 'date', 'name'])
        last = hist.sort_values('date').groupby('code').tail(1).set_index('code')['name'].to_dict()
        changed = [c for c in codes if c in code2name and _is_st_name(code2name[c]) != _is_st_name(last.get(c, ''))]
        filled = 0
        for code in codes:
            if code not in code2name:
                continue
            val = int(_is_st_name(code2name[code]))
            path = os.path.join(V2_DIR, f'{code}.csv')
            tail = read_last_lines(path, n_lines=1) if os.path.exists(path) else []
            parts = tail[-1].split(',') if tail else []
            if len(parts) < 13 or parts[0] != target:
                continue                              # 当天无行 → 无处标注
            cur = _num(parts[12])
            if cur is not None and int(cur) == val:
                continue                              # 当天行已正确 → 不重写文件
            df = read_v2(code)
            if df.empty:
                continue
            col = pd.to_numeric(df['isST'], errors='coerce').ffill().fillna(0).astype(int)
            col.loc[df['date'] == target] = val
            df['isST'] = col
            write_v2(code, df)
            filled += 1
        for code in changed:
            logger.info(f'[isst] 名字变化: {code} {last.get(code)} → {code2name[code]}')
        return filled

    # ==================== 内部：m15 ====================

    def _plan_m15(self, code, cal, last_td):
        """该票 m15 缺口计划 → (datalen, force_merge) / 'old' / None。

        新浪只支持"最近 N 根"：N 按缺口长度换算（16 根/日）；缺口早于 64 交易日窗 → 'old'（需全史手段）。
        """
        path = os.path.join(M15_KLINE_DIR, f'{code}.csv')
        i_last = cal.index(last_td)
        if not os.path.exists(path):
            return 1023, True
        dates = set(pd.read_csv(path, usecols=['date'], dtype={'date': str})['date'])
        i0 = cal.index(min(dates))
        miss = [d for d in cal[i0:i_last + 1] if d not in dates]
        if not miss:
            return None
        win_start = cal[max(0, i_last - 62)]
        win_miss = [d for d in miss if d >= win_start]
        if not win_miss:
            return 'old'
        if len(win_miss) < len(miss):
            logger.info(f'[m15] {code}: {len(miss) - len(win_miss)} 天缺口早于新浪窗（{miss[0]} 起），不可回补')
        datalen = min(SINA_MAX_BARS, (i_last - cal.index(win_miss[0]) + 1) * BARS_PER_DAY + BARS_PER_DAY)
        return datalen, min(win_miss) <= max(dates)   # 缺口中段→合并；仅尾部→追加

    def _delegate_history(self, codes):
        """整段历史：委托旧器 update_kline.py（baostock 15 分钟全史，列与 v2 同构，逐票续传）。"""
        from AshareData.datautils.kline_scripts.update_kline import UpdateKline
        UpdateKline().update_kline_daily(codes=codes, frequencies=['15'])

    def _fetch_m15(self, code, datalen):
        sym = code.replace('.', '')
        s = getattr(_TLS, 'sina', None)
        if s is None:
            s = _TLS.sina = requests.Session()
            s.headers['User-Agent'] = 'Mozilla/5.0'
        last = None
        for _ in range(3):
            try:
                r = s.get(SINA_KLINE, params={'symbol': sym, 'scale': '15', 'ma': 'no', 'datalen': str(datalen)},
                          timeout=10)
                arr = json.loads(r.text.strip())
                if not isinstance(arr, list) or not arr:
                    return pd.DataFrame()
                rows = [{'date': it['day'][:10],
                         'time': int(it['day'][11:13]) * 100 + int(it['day'][14:16]),
                         'open': float(it['open']), 'high': float(it['high']), 'low': float(it['low']),
                         'close': float(it['close']), 'volume': int(float(it['volume'])),
                         'amount': float(it['amount'])} for it in arr]
                time.sleep(random.uniform(0.02, 0.06))
                return pd.DataFrame(rows)
            except Exception as e:
                last = e
                time.sleep(random.uniform(0.5, 1.5))
        raise last

    def _m15_upsert(self, code, rows, force_merge=False):
        path = os.path.join(M15_KLINE_DIR, f'{code}.csv')
        os.makedirs(M15_KLINE_DIR, exist_ok=True)   # to_csv 不会建父目录；空库首跑否则 OSError
        if not os.path.exists(path):
            rows.insert(1, 'code', code)
            rows[M15_COLS].to_csv(path, index=False)
            return True
        tail = read_last_lines(path, n_lines=1)
        last = tail[-1].split(',') if tail else None
        last_dt = (last[0], int(last[1])) if last and len(last) >= 2 else None
        new_rows = rows if last_dt is None else rows[[(r.date, r.time) > last_dt for r in rows.itertuples()]]
        if len(new_rows) == 0 and not force_merge:
            return False
        if not force_merge and list(new_rows.index) == list(range(len(rows) - len(new_rows), len(rows))):
            out = new_rows.copy()   # 抓取窗与本地衔接（新区间=连续后缀）→ 直接追加
            out.insert(1, 'code', code)
            out[M15_COLS].to_csv(path, mode='a', header=False, index=False)
            return True
        old = pd.read_csv(path, dtype={'date': str})
        m = pd.concat([old, rows.assign(code=code)], ignore_index=True)
        m = m.drop_duplicates(['date', 'time'], keep='last').sort_values(['date', 'time'])
        m[M15_COLS].to_csv(path, index=False)
        return True

    # ==================== 内部：辅助表 ====================

    def _sync_delisted(self, prev, meta):
        """退市表维护：名单消失→记入；重新出现→移出；首建时用旧库独有票播种。"""
        def codes_of(frame):
            if frame is None:
                return set(), {}
            sub = frame[frame['thscode'].str.endswith(('.SH', '.SZ'))]
            cs = {f"{x.split('.')[1].lower()}.{x.split('.')[0]}" for x in sub['thscode']}
            nm = {f"{x.split('.')[1].lower()}.{x.split('.')[0]}": n for x, n in zip(sub['thscode'], sub['name'])}
            return cs, nm
        cur, _ = codes_of(meta)
        hist = (pd.read_parquet(DELISTED_F) if os.path.exists(DELISTED_F)
                else pd.DataFrame(columns=['code', 'name', 'last_date', 'detect_date']))
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        gone = {}
        if prev is None:
            if os.path.isdir(DAILY_KLINE_DIR):
                for f in sorted(os.listdir(DAILY_KLINE_DIR)):
                    c = f[:-4]
                    if f.endswith('.csv') and c not in cur:
                        tail = read_last_lines(os.path.join(DAILY_KLINE_DIR, f), n_lines=1)
                        last = tail[-1].split(',')[0] if tail else ''
                        gone[c] = {'code': c, 'name': '', 'last_date': last, 'detect_date': last}
        else:
            pc, pn = codes_of(prev)
            for c in sorted(pc - cur):
                p = os.path.join(V2_DIR, f'{c}.csv')
                tail = read_last_lines(p, n_lines=1) if os.path.exists(p) else []
                gone[c] = {'code': c, 'name': pn.get(c) or '',
                           'last_date': tail[-1].split(',')[0] if tail else '', 'detect_date': today}
        known = set(hist['code'])
        add = [v for k, v in gone.items() if k not in known]
        back = sorted(known & cur)
        if back:
            hist = hist[~hist['code'].isin(back)]
        if add:
            hist = pd.concat([hist, pd.DataFrame(add)], ignore_index=True)
        if add or back:
            os.makedirs(AUX_DIR, exist_ok=True)
            hist.sort_values('code').to_parquet(DELISTED_F)
        if add:
            logger.info(f'[meta] 新增退市 {len(add)} 只: {[r["code"] for r in add[:10]]}')
        if back:
            logger.info(f'[meta] 名单回归（移出退市表）: {back}')

    def _load_prev_close(self):
        d = dict(self._prev_close_mem)
        if os.path.exists(PREVCLOSE_F):
            p = pd.read_parquet(PREVCLOSE_F)
            for c, v in zip(p['code'], p['prev_close']):
                d.setdefault(c, v)
        return d

    def _old_col(self, code, col, dates):
        """旧库移植列：返回与 dates 对齐的 Series（仅旧库真实行有值，其余 NaN）。"""
        p = os.path.join(DAILY_KLINE_DIR, f'{code}.csv')
        if not os.path.exists(p):
            return None
        try:
            old = pd.read_csv(p, dtype={'date': str}, usecols=['date', 'tradestatus', col])
        except ValueError:
            return None
        old = old[pd.to_numeric(old['tradestatus'], errors='coerce') == 1]
        m = dict(zip(old['date'], pd.to_numeric(old[col], errors='coerce')))   # 旧库有重复日期行 → dict last-wins
        return pd.Series(dates).map(m)


def is_latest(quiet=False):
    """整库各字段是否都无缺口（复用与 update_* 同一套缺口判定，不另写一份）。

    逐票：①交易日覆盖 [上市日或库起点, 目标日] 无缺；②真成交行（volume 非空）的
    preclose/amount/turn/pctChg 与 isST 非空；③有成交的票 m15 无缺口（同一个 _plan_m15）。
    豁免「重跑也补不回来」的票：退市、本地无数据、结构性无量（全表从未算出过 turn，
    如新浪/扶摇都给不出股本的 sh.689009）。代价是要读全表，比抽末行慢（约十几秒）。
    """
    b = KlineV2Builder()
    tgt = b._target_date()
    codes = b._codes()
    if not codes:
        if not quiet:
            logger.warning(f'[is_latest] v2 库为空（{V2_DIR}）→ False')
        return False
    delisted = set(pd.read_parquet(DELISTED_F)['code']) if os.path.exists(DELISTED_F) else set()
    ev = pd.read_parquet(SHARE_F) if os.path.exists(SHARE_F) else pd.DataFrame(columns=['code', 'date', 'shares_wan'])
    first_ev = ev.groupby('code')['date'].min().to_dict() if len(ev) else {}
    lags, exempt = {}, {}
    for code in codes:
        if code in delisted:
            exempt.setdefault('delisted', []).append(code)
            continue
        df = read_v2(code)
        if df.empty:
            exempt.setdefault('no_data', []).append(code)      # 源里没有该票（或尚未建库）
            continue
        if b._missing(code, df):
            lags.setdefault('date', []).append(code)
        traded = df[df['volume'].notna()]
        if len(traded):
            for col in ('preclose', 'pctChg'):
                blank = pd.to_numeric(traded[col], errors='coerce').isna()
                if blank.any() and not (blank.sum() == 1 and bool(blank.iloc[0])):
                    lags.setdefault(col, []).append(code)      # 首行无前收（新股）当例外
            if pd.to_numeric(traded['amount'], errors='coerce').isna().any():
                lags.setdefault('amount', []).append(code)
            blank_turn = pd.to_numeric(traded['turn'], errors='coerce').isna()
            if blank_turn.any():
                ev0 = first_ev.get(code)
                if ev0 is not None and not blank_turn[traded['date'] >= ev0].any():
                    exempt.setdefault('turn', []).append(code)  # 只缺首个股本事件之前的行
                elif ev0 is None:
                    exempt.setdefault('turn', []).append(code)  # 源完全没有股本事件 → 算不出
                else:
                    lags.setdefault('turn', []).append(code)
            if b._plan_m15(code, _cal(), tgt) is not None:
                lags.setdefault('m15', []).append(code)
        if pd.to_numeric(df['isST'], errors='coerce').isna().any():
            lags.setdefault('isST', []).append(code)
    note = '，'.join(f'{k}={len(v)}' for k, v in sorted(exempt.items()))
    if not lags:
        if not quiet:
            logger.info(f'[is_latest] 全部字段已达 {tgt}（{len(codes)} 只'
                        + (f'；已豁免 {note}' if note else '') + '）')
        return True
    if not quiet:
        for k, v in sorted(lags.items()):
            logger.warning(f'[is_latest] {k} 未达 {tgt}: {len(v)} 只（样例: {v[:5]}）')
        if note:
            logger.info(f'[is_latest] 已豁免（源不提供，重跑无效）: {note}')
    return False


def main():
    ap = argparse.ArgumentParser(description='日K v2 构建器（字段自维护：缺口与手段自动判定）')
    ap.add_argument('--codes', default='', help='逗号分隔，如 600519,sz.000001（缺省=全市场沪深）')
    ap.add_argument('--skip', default='', help='跳过字段，如 m15,isst')
    ap.add_argument('--force', action='store_true', help='强制重取（忽略"已齐/已抓过"的跳过判断）')
    ap.add_argument('--latest', action='store_true', help='只做整库状态问答（is_latest），不抓数据')
    a = ap.parse_args()
    if a.latest:
        sys.exit(0 if is_latest() else 1)
    codes = norm_codes(a.codes) if a.codes else None
    KlineV2Builder(force=a.force).run(codes, skip=[s for s in a.skip.split(',') if s])


if __name__ == '__main__':
    main()

