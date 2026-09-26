# -*- coding: utf-8 -*-
"""quick_kline —— 日K v2 构建器（扶摇 OHLCV / 新浪股本·m15 / baostock isST 回扫）

设计（2026-09-25 定稿）：3 种处理模式 × 按列方法，统一签名 (mode, codes, start, end)
  - 模式: rebuild(空历史重建) / backfill(缺口日期补充) / update(当天更新)
  - 列方法（各自实现三种模式的处理方式）:
      update_meta      代码表/名称/上市日      (扶摇 meta，1 请求)
      update_adjust    复权事件表(aux)         (扶摇 adjustment-factors dump)
      update_ohlcvt    OHLCV+额 + 停牌占位行 + tradestatus 推导   (扶摇 dump/快照)
      update_preclose  preclose/pctChg 后处理  (依赖 adjust + ohlcvt)
      update_turn      换手率                  (历史/缺口=旧库移植→新浪事件兜底; 当天=扶摇竞价快照)
      update_isst      isST                    (重建=baostock 单线程回扫 或 旧库移植; 更新=扶摇名字对比)
      update_m15       15分钟线                (rebuild=baostock 单线程全史慢扫+续传；增量/缺口=新浪 getKLineData)
  - 产物: daily_kline_v2/{sh.600000.csv} 13 列，与旧库同 schema（OHLCV 为**未复权原值**）
    辅助: daily_kline_v2/_aux/、_dumps/
  - v2 起点: 2020-01-02（与旧库/消费链对齐；扶摇 dump 的 2016+ 仅用于边界前收盘）
用法:
  python AshareData/datautils/kline_scripts/quick_kline.py --mode rebuild  [--codes 600519,sz.000001] [--isst-from baostock|old]
  python AshareData/datautils/kline_scripts/quick_kline.py --mode backfill [--codes ...] [--start ...] [--end ...]
  python AshareData/datautils/kline_scripts/quick_kline.py --mode update   [--codes ...]
"""
import argparse
import io
import json
import os
import random
import re
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
from AshareData.utils.read_file_utils import get_first_last_line_from_csv, read_last_lines

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
DEFAULT_START = '2020-01-02'
SINA_KLINE = 'https://quotes.sina.cn/cn/api/json_v2.php/CN_MarketDataService.getKLineData'
SINA_SHARE = ('https://stock.finance.sina.com.cn/stock/api/jsonp.php/'
              'var%20KKE_ShareAmount_{sym}=/StockService.getAmountBySymbol?_=20&symbol={sym}')

_CAL = None
_TLS = threading.local()


def _cal():
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
    for i in range(4):
        try:
            url = api_json(f'/api/dump/market-dumps/{name}/download-url')['presigned_url']
            with requests.get(url, stream=True, timeout=300) as r:
                r.raise_for_status()
                buf = io.BytesIO(b''.join(r.iter_content(1 << 20)))
            df = pd.read_parquet(buf)
            df.to_parquet(path)
            logger.info(f'下载 {name}: {len(df)} 行 → _dumps/{name}.parquet')
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
    out = []
    for tok in re.split(r'[,\s]+', text or ''):
        tok = tok.strip()
        if not tok:
            continue
        if '.' in tok:
            a, b = tok.split('.')
            if a.isdigit():
                a, b = b, a
            code = f'{b.lower()}.{a}'
        elif tok.startswith('6'):
            code = f'sh.{tok}'
        elif tok.startswith(('0', '2', '3')):
            code = f'sz.{tok}'
        else:
            raise ValueError(f'无法识别代码: {tok}')
        out.append(code)
    return sorted(set(out))


def norm_date(s, default=None):
    if not s:
        return default
    s = str(s).strip()
    if len(s) == 8 and s.isdigit():
        return f'{s[:4]}-{s[4:6]}-{s[6:8]}'
    return s


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


class KlineV2Builder:
    def __init__(self, isst_from='baostock', force=False):
        self.isst_from = isst_from
        self.force = force
        self._bs_logged = False
        self._prev_close_mem = None

    # ==================== 列方法（统一签名） ====================

    def update_meta(self, mode, codes, start, end):
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

    def _sync_delisted(self, prev, meta):
        """退市表维护：meta 名单消失→记入（last_date=库内最后日期）；重新出现→移出；首建时用旧库独有票播种。"""
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
                last = tail[-1].split(',')[0] if tail else ''
                gone[c] = {'code': c, 'name': pn.get(c) or '', 'last_date': last, 'detect_date': today}
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

    def update_adjust(self, mode, codes, start, end):
        d = _to_local(ensure_dump('adjustment-factors'))
        d = d[['code', 'date', 'dividend_per_share', 'per_share_bonus', 'allotment_ratio', 'allotment_price']]
        d = d.rename(columns={'date': 'ex_date'}).sort_values(['code', 'ex_date'])
        os.makedirs(AUX_DIR, exist_ok=True)
        d.to_parquet(ADJ_F)
        logger.info(f'[adjust] {len(d)} 事件 / {d["code"].nunique()} 只')

    def update_ohlcvt(self, mode, codes, start, end):
        if mode == 'update':
            # 是否需要更新交给 is_latest 统一判定（其基准 get_target_trade_date 已含收盘时点：
            # 交易日 15:00 前取上一交易日）；不再自算 15:30，否则两套时点不一致时，15:00~15:30
            # 会出现「目标日已滚动到当天、却仍被判为未收盘」的空窗。
            # quiet=True：它是全库判据（m15/turn/isST 都在内），那些列各有自己的步骤负责，
            # 不能在本步的日志里替它们报账
            if not self.force and is_latest(quiet=True):
                logger.info('[ohlcvt] 已是最新，跳过（--force 可强跑）')
                return
            src = self._source_update(codes)
            market_last = norm_date(end) or self._target_date()
        elif mode == 'backfill':
            src, market_last = self._source_backfill(codes, norm_date(start, DEFAULT_START), norm_date(end))
        else:
            src, market_last = self._source_rebuild(codes, norm_date(start, DEFAULT_START), norm_date(end))
        n = 0
        for code in codes:
            df = pd.DataFrame(columns=COLS) if mode == 'rebuild' else read_v2(code)
            rows = src.get(code)
            if rows is not None and len(rows):
                rows = rows.copy()
                rows['code'] = code
                rows['preclose'] = np.nan
                rows['turn'] = np.nan
                rows['pctChg'] = np.nan
                rows['isST'] = np.nan
                df = pd.concat([df, rows], ignore_index=True)
            if df.empty:
                continue
            df = self._materialize(df, market_last)
            write_v2(code, df)
            n += 1
        logger.info(f'[ohlcvt] {mode}: 写 {n} 只，market_last={market_last}')

    def update_preclose(self, mode, codes, start, end):
        adj = pd.read_parquet(ADJ_F) if os.path.exists(ADJ_F) else pd.DataFrame(
            columns=['code', 'ex_date', 'dividend_per_share', 'per_share_bonus', 'allotment_ratio', 'allotment_price'])
        prev_map = self._load_prev_close()
        n = 0
        for code in codes:
            df = read_v2(code)
            if df.empty:
                continue
            ev = adj[adj['code'] == code]
            g = ev.groupby('ex_date').agg(D=('dividend_per_share', 'sum'), B=('per_share_bonus', 'sum'),
                                          R=('allotment_ratio', 'sum')).reset_index()
            ev2 = ev.copy()
            ev2['Rp'] = ev2['allotment_ratio'].fillna(0) * ev2['allotment_price'].fillna(0)
            g = g.merge(ev2.groupby('ex_date')['Rp'].sum().reset_index(), on='ex_date', how='left')
            ev_dates = g['ex_date'].tolist()
            f_of = {}
            for _, e in g.iterrows():
                f_of[e['ex_date']] = (e['D'], e['B'], e['R'], e['Rp'])
            cal = _cal()
            last_close, last_mark = None, None
            pre = np.full(len(df), np.nan)
            pch = np.full(len(df), np.nan)
            dates = df['date'].tolist()
            closes = pd.to_numeric(df['close'], errors='coerce').to_numpy()
            vols = pd.to_numeric(df['volume'], errors='coerce').to_numpy()
            for i in range(len(df)):
                if not np.isfinite(vols[i]):  # 占位行：preclose 保持 carry
                    pre[i] = closes[i]
                    continue
                d = dates[i]
                if last_close is None:
                    base = prev_map.get(code, np.nan)
                    j = cal.index(d) if d in cal else 0
                    lo = cal[max(0, j - 1)] if cal else ''
                else:
                    base = last_close
                    lo = last_mark
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
                    # 交易所惯例：除权参考价四舍五入到 0.01（半格边界须用十进制 half-up，float 银行家舍入会错）
                    if len(seg) == 1:
                        D, B, R, Rp = seg[0]
                        pc = float((Decimal(str(base)) - Decimal(str(D)) + Decimal(str(Rp)))
                                   / (Decimal(1) + Decimal(str(B)) + Decimal(str(R))))
                    pc = float(Decimal(str(pc)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP))
                pre[i] = pc
                if np.isfinite(pc) and pc > 0 and np.isfinite(closes[i]):
                    pch[i] = (closes[i] / pc - 1.0) * 100.0
                last_close, last_mark = closes[i], d
            df['preclose'] = pre
            df['pctChg'] = pch
            if mode != 'update':
                overlay = self._old_col(code, 'pctChg', df['date'])
                if overlay is not None:
                    df['pctChg'] = np.where(overlay.notna(), overlay, df['pctChg'])
            write_v2(code, df)
            n += 1
        logger.info(f'[preclose] {mode}: 写 {n} 只')

    def update_turn(self, mode, codes, start, end):
        target = None
        if mode == 'update':
            target = norm_date(end) or self._target_date()
            snap = self._auction_snapshot(codes)
            events = self._load_share_events()
            refetch = []
            for code in codes:
                df = read_v2(code)
                if df.empty or target not in set(df['date']):
                    continue
                it = snap.get(code)
                idx = df.index[df['date'] == target]
                if it and it.get('last_price') and it.get('float_market_cap'):
                    shares = it['float_market_cap'] / it['last_price']  # 股
                    if shares > 0:
                        vol = pd.to_numeric(df.loc[idx, 'volume'], errors='coerce').iloc[0]
                        df.loc[idx, 'turn'] = vol / shares * 100 if np.isfinite(vol) else np.nan
                        ev = events[events['code'] == code]
                        ev = ev[ev['date'] <= target]
                        if len(ev):
                            last_wan = float(ev.sort_values('date')['shares_wan'].iloc[-1])
                            if last_wan > 0 and abs(shares / 1e4 - last_wan) / last_wan > 0.005:
                                refetch.append(code)
                write_v2(code, df)
            if refetch:
                logger.info(f'[turn] 快照股本与事件表有差异 → 刷新 {len(refetch)} 只: {refetch[:8]}')
                self._fetch_share_events(refetch)
            logger.info(f'[turn] update: {len(codes)} 只（target={target}）')
            return
        # rebuild / backfill：历史段=旧库 turn 移植（替换连续性），缺口/新增票=新浪事件 as-of
        self._fetch_share_events(codes)
        events = self._load_share_events()
        n = 0
        for code in codes:
            df = read_v2(code)
            if df.empty:
                continue
            ev = events[events['code'] == code].sort_values('date')
            if not ev.empty:
                edates = ev['date'].to_numpy()
                eshares = ev['shares_wan'].to_numpy()
                idx = np.searchsorted(edates, df['date'].to_numpy(), side='right') - 1
                shares = np.where(idx >= 0, eshares[np.clip(idx, 0, None)], np.nan)
                vols = pd.to_numeric(df['volume'], errors='coerce').to_numpy()
                df['turn'] = vols / (shares * 1e4) * 100
            overlay = self._old_col(code, 'turn', df['date'])
            if overlay is not None:
                df['turn'] = np.where(overlay.notna(), overlay, df['turn'])
            write_v2(code, df)
            n += 1
        logger.info(f'[turn] {mode}: 写 {n} 只')

    def update_isst(self, mode, codes, start, end):
        if mode == 'update':
            self._isst_by_name(codes)
            return
        n = 0
        for code in codes:
            try:
                scan = self._scan_isst(code, norm_date(start, DEFAULT_START), norm_date(end))
            except Exception as e:
                logger.info(f'[isst] {code} 扫描失败（可重跑续传）: {e}')
                continue
            df = read_v2(code)
            if df.empty:
                continue
            m = dict(zip(scan['date'], scan['isST']))
            df['isST'] = [m.get(x, np.nan) for x in df['date']]
            df['isST'] = pd.to_numeric(df['isST'], errors='coerce').ffill().fillna(0).astype(int)
            write_v2(code, df)
            n += 1
        logger.info(f'[isst] {mode}: 写 {n} 只（来源={self.isst_from}）')

    def update_m15(self, mode, codes, start, end):
        if mode == 'rebuild':
            self._m15_rebuild_baostock(codes, norm_date(start, DEFAULT_START), norm_date(end))
            return
        cal = _cal()
        last_td = self._target_date()   # 统一到 get_target_trade_date（交易日 15:00 后含当天）
        plan = {}
        for code in codes:
            try:
                r = self._m15_plan(mode, code, cal, last_td)
            except Exception as e:
                logger.info(f'[m15] {code} 计划失败: {e}')
                continue
            if r is not None:
                plan[code] = r
        logger.info(f'[m15] {mode}: 需抓 {len(plan)}/{len(codes)} 只（last_td={last_td}）')
        rows_by_code = {}
        done = 0
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(self._fetch_m15, c, dl): (c, fm) for c, (dl, fm) in plan.items()}
            for f in as_completed(futs):
                done += 1
                c, fm = futs[f]
                try:
                    rows_by_code[c] = (f.result(), fm)
                except Exception as e:
                    logger.info(f'[m15] {c} 抓取失败: {e}')
                if done % 500 == 0:
                    logger.info(f'[m15] 抓取进度 {done}/{len(plan)}')
        nw = 0
        for code, (rows, fm) in rows_by_code.items():
            if rows is None or rows.empty:
                continue
            if self._m15_upsert(code, rows, force_merge=fm):
                nw += 1
        logger.info(f'[m15] {mode}: 更新 {nw}/{len(codes)} 只')

    def _m15_rebuild_baostock(self, codes, start, end):
        """m15 全史重建：baostock 单线程慢扫（全史唯一免费源，~55s/只）+ 逐票按本地覆盖续传。

        断点续传=以本地文件首/末 bar 判定缺口（上市日前不算缺口），只补缺段；--force 全区间重扫。
        注：停牌与数据缺失无法按日期区分，中段空洞不在此路径检测（需要时用 --force 重扫）。
        """
        from AshareData.utils.kline_data_utils.query_kline_utils import QueryKlineUtils
        qu = QueryKlineUtils()
        tgt_end = end or self._target_date()
        meta = pd.read_parquet(META_F) if os.path.exists(META_F) else None
        ld = {}
        if meta is not None:
            sub = meta[meta['thscode'].str.endswith(('.SH', '.SZ'))]
            for x, v in zip(sub['thscode'], sub['list_date']):
                d = pd.to_datetime(v, errors='coerce') if v else None
                if d is not None and pd.notna(d):
                    ld[f"{x.split('.')[1].lower()}.{x.split('.')[0]}"] = d.strftime('%Y-%m-%d')
        todo = []
        delisted = set(pd.read_parquet(DELISTED_F)['code']) if os.path.exists(DELISTED_F) else set()
        for code in codes:
            if code in delisted:
                continue   # 退市票不再产生新 bar
            p = os.path.join(M15_KLINE_DIR, f'{code}.csv')
            if self.force or not os.path.exists(p):
                todo.append((code, start, tgt_end))
                continue
            fl, msg = get_first_last_line_from_csv(p)
            if msg or fl is None or fl.empty:
                todo.append((code, start, tgt_end))
                continue
            first_d, last_d = str(fl['date'].iloc[0]), str(fl['date'].iloc[-1])
            exp_first = max(start, ld.get(code, start))
            if first_d > exp_first and last_d < tgt_end:
                todo.append((code, exp_first, tgt_end))
            elif first_d > exp_first:
                todo.append((code, exp_first, (pd.to_datetime(first_d) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')))
            elif last_d < tgt_end:
                todo.append((code, (pd.to_datetime(last_d) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), tgt_end))
        if not todo:
            logger.info('[m15] rebuild: 全部已覆盖，无需扫描')
            return
        logger.info(f'[m15] rebuild: baostock 单线程需扫 {len(todo)}/{len(codes)} 只（~55s/只，预计 {len(todo) * 55 / 3600:.1f}h）')
        ok = fail = empty = 0
        t0 = time.time()
        for i, (code, qs, qe) in enumerate(todo):
            df = msg = None
            for attempt in range(3):
                df, msg = qu.query_history(code, start_date=qs, end_date=qe, frequency='15', adjustflag='2')
                if df is not None:
                    break
                time.sleep(5 * (attempt + 1))
            if df is None:
                fail += 1
                logger.info(f'[m15] {code} 扫描失败（重跑续传）: {msg}')
            elif df.empty:
                empty += 1
            else:
                df = df[['date', 'time', 'open', 'high', 'low', 'close', 'volume', 'amount']]
                df['time'] = pd.to_numeric(df['time'], errors='coerce').astype(int)
                self._m15_upsert(code, df, force_merge=True)
                ok += 1
            if (i + 1) % 20 == 0 or i == len(todo) - 1:
                el = time.time() - t0
                eta = el / (i + 1) * (len(todo) - i - 1)
                logger.info(f'[m15] rebuild 进度 {i + 1}/{len(todo)} ok={ok} empty={empty} fail={fail} '
                            f'已用 {el / 3600:.1f}h ETA {eta / 3600:.1f}h')
        logger.info(f'[m15] rebuild 完成: ok={ok} empty={empty} fail={fail}，共 {(time.time() - t0) / 3600:.1f}h')

    def _m15_plan(self, mode, code, cal, last_td):
        """按本地缺口计算抓取计划 → (datalen, force_merge)；None=无需抓取。

        新浪只能取"最近 N 根"，不支持按日期区间查；N 按本地实际缺口长度换算（16 根/日），
        只对真有缺口的票发起请求；超出新浪 64 交易日窗的缺口记录为不可回补。
        """
        path = os.path.join(M15_KLINE_DIR, f'{code}.csv')
        i_last = cal.index(last_td)
        win_start = cal[max(0, i_last - 62)]
        if not os.path.exists(path):
            return 1023, True   # 无文件 → 尽量拉满窗
        if mode == 'update':
            tail = read_last_lines(path, n_lines=1)
            parts = tail[-1].split(',') if tail else []
            if len(parts) < 2:
                return 1023, True
            if parts[0] > last_td or (parts[0] == last_td and int(parts[1]) >= 1500):
                return None   # 本地已含最新交易日尾 bar，无需抓
            i0 = cal.index(parts[0]) if parts[0] in cal else max(0, i_last - 2)
            return min(1023, max(32, (i_last - i0 + 1) * 16)), False
        # backfill：扫描本地日期找缺口
        dates = set(pd.read_csv(path, usecols=['date'], dtype={'date': str})['date'])
        i0 = cal.index(min(dates))
        miss = [d for d in cal[i0:i_last + 1] if d not in dates]
        if not miss:
            return None
        win_miss = [d for d in miss if d >= win_start]
        if len(win_miss) < len(miss):
            logger.info(f'[m15] {code}: {len(miss) - len(win_miss)} 天缺口早于新浪窗（{miss[0]} 起），不可回补')
        if not win_miss:
            return None
        datalen = min(1023, (i_last - cal.index(win_miss[0]) + 1) * 16 + 16)
        return datalen, min(win_miss) <= max(dates)   # 缺口中段→合并；仅尾部→追加

    # ==================== 调度 ====================

    def run(self, mode, codes=None, start=None, end=None, skip=()):
        t0 = time.time()
        self.update_meta(mode, codes, start, end)
        if codes is None:
            codes = self._default_codes()
        logger.info(f'== {mode} | {len(codes)} 只 | start={start} end={end} ==')
        for name in ['adjust', 'ohlcvt', 'preclose', 'turn', 'isst', 'm15']:
            if name in skip:
                logger.info(f'== skip {name} ==')
                continue
            getattr(self, f'update_{name}')(mode, codes, start, end)
        logger.info(f'== {mode} 完成，共 {time.time() - t0:.0f}s ==')

    def _default_codes(self):
        meta = pd.read_parquet(META_F)
        return sorted(f"{x.split('.')[1].lower()}.{x.split('.')[0]}"
                      for x in meta['thscode'] if x.endswith(('.SH', '.SZ')))

    # ==================== 内部：数据源 ====================

    def _target_date(self):
        t = get_target_trade_date()
        return pd.to_datetime(t).strftime('%Y-%m-%d') if t else pd.Timestamp.now().strftime('%Y-%m-%d')

    def _source_rebuild(self, codes, start, end):
        d = _to_local(ensure_dump('daily-k'))
        d = d[d['adjusted'] == 'none']
        d = d.rename(columns={'open_price': 'open', 'high_price': 'high', 'low_price': 'low',
                              'close_price': 'close', 'turnover': 'amount'})
        market_last = d['date'].max()
        if end:
            market_last = min(market_last, end)
        sub = d[d['code'].isin(codes)]
        # 边界前收：每只票 start 前最后一个收盘（含 2016+ 段）
        prev = (sub[sub['date'] < start].sort_values('date').groupby('code')['close'].last()
                if len(sub) else pd.Series(dtype=float))
        self._prev_close_mem = prev.to_dict()
        prev_df = pd.DataFrame({'code': list(self._prev_close_mem), 'prev_close': list(self._prev_close_mem.values())})
        os.makedirs(AUX_DIR, exist_ok=True)
        prev_df.to_parquet(PREVCLOSE_F)
        sub = sub[sub['date'] >= start]
        if end:
            sub = sub[sub['date'] <= end]
        sub = sub[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        return {c: g.drop(columns='code') for c, g in sub.groupby('code')}, market_last

    def _source_backfill(self, codes, start, end):
        """缺口补充：10d dump 覆盖窗口内直接补；更早的缺口回落到全量 dump。"""
        d10 = _to_local(ensure_dump('daily-k-10d'))
        d10 = d10.rename(columns={'open_price': 'open', 'high_price': 'high', 'low_price': 'low',
                                  'close_price': 'close', 'turnover': 'amount'})
        d10 = d10[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        d10 = d10[d10['code'].isin(codes)]
        win_min = d10['date'].min() if len(d10) else '9999'
        out, older = {}, []
        for code in codes:
            df = read_v2(code)
            if df.empty:
                continue
            miss = self._missing_dates(df, start, end)
            if not miss:
                continue
            win = [x for x in miss if x >= win_min]
            old = [x for x in miss if x < win_min]
            if win:
                out.setdefault(code, []).append(d10[(d10['code'] == code) & (d10['date'].isin(win))])
            if old:
                older.append((code, old))
        if older:
            full = _to_local(ensure_dump('daily-k'))
            full = full.rename(columns={'open_price': 'open', 'high_price': 'high', 'low_price': 'low',
                                        'close_price': 'close', 'turnover': 'amount'})
            full = full[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
            for code, old in older:
                out.setdefault(code, []).append(full[(full['code'] == code) & (full['date'].isin(old))])
        src = {c: pd.concat(v, ignore_index=True) for c, v in out.items()}
        market_last = self._target_date()
        return src, market_last

    def _source_update(self, codes):
        """当天更新：扶摇 prices/snapshot（批量 100）。"""
        target = self._target_date()
        items = self._batch_snapshot('/api/a-share/prices/snapshot', codes)
        out = {}
        for it in items:
            code = f"{it['thscode'].split('.')[1].lower()}.{it['thscode'].split('.')[0]}"
            if code not in codes:
                continue
            out[code] = pd.DataFrame([{
                'date': target, 'open': it.get('open_price'), 'high': it.get('high_price'),
                'low': it.get('low_price'), 'close': it.get('last_price'),
                'volume': it.get('volume'), 'amount': it.get('turnover')}])
        return out

    def _batch_snapshot(self, path, codes):
        items = []
        for i in range(0, len(codes), 100):
            chunk = codes[i:i + 100]
            ts = ','.join(f"{c.split('.')[1]}.{c.split('.')[0].upper()}" for c in chunk)
            d = api_json(path, {'thscodes': ts})
            items += d.get('item') or []
        return items

    def _auction_snapshot(self, codes):
        items = self._batch_snapshot('/api/a-share/auction/snapshot', codes)
        out = {}
        for it in items:
            code = f"{it['thscode'].split('.')[1].lower()}.{it['thscode'].split('.')[0]}"
            out[code] = it
        return out

    # ==================== 内部：行网格 / 停牌占位 ====================

    def _missing_dates(self, df, start, end):
        have = set(df['date'])
        cal = _cal()
        lo = start or df['date'].iloc[0]
        hi = end or self._target_date()
        return [d for d in cal if lo <= d <= hi and d not in have]

    def _materialize(self, df, market_last):
        """补齐 [首行, market_last] 的交易日网格：缺日 → 停牌占位行（OHLC=前收，量额空，tradestatus=0）。"""
        df = df.sort_values('date').drop_duplicates('date', keep='last').reset_index(drop=True)
        if df.empty:
            return df
        cal = _cal()
        d0 = df['date'].iloc[0]
        d1 = min(market_last, cal[-1]) if market_last else df['date'].iloc[-1]
        if d1 < d0:
            d1 = df['date'].iloc[-1]
        need = [x for x in cal if d0 <= x <= d1]
        missing = [x for x in need if x not in set(df['date'])]
        if missing:
            real = df[df['volume'].notna()].sort_values('date')
            if len(real):
                rds = real['date'].to_numpy()
                rcs = pd.to_numeric(real['close'], errors='coerce').to_numpy()
                pos = np.searchsorted(rds, np.array(missing), side='left') - 1
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
        return df

    # ==================== 内部：股本 ====================

    def _load_share_events(self):
        if os.path.exists(SHARE_F):
            return pd.read_parquet(SHARE_F)
        return pd.DataFrame(columns=['code', 'date', 'shares_wan'])

    def _fetch_share_events(self, codes):
        """新浪流通股本事件（每只一个 jsonp 全量事件表；6 线程 + 线程内 session 复用 + 抖动）。"""
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
        with ThreadPoolExecutor(max_workers=6) as ex:
            for code, rows, err in ex.map(fetch, todo):
                done += 1
                if err is not None:
                    failed.append(code)
                    logger.info(f'[share] {code} 失败: {err}')
                else:
                    log[code] = today
                    new += rows
                if done % 500 == 0:
                    logger.info(f'[share] {done}/{len(todo)}')
        if new:
            ev_all = pd.concat([ev_all, pd.DataFrame(new)], ignore_index=True)
            ev_all = ev_all.drop_duplicates(['code', 'date'], keep='last').sort_values(['code', 'date'])
            ev_all.to_parquet(SHARE_F)
        json.dump(log, open(SHARE_LOG_F, 'w'))
        logger.info(f'[share] 新增 {len(new)} 条事件（累计 {len(ev_all)}），失败 {len(failed)} 只')

    # ==================== 内部：isST ====================

    def _scan_isst(self, code, start, end):
        if self.isst_from == 'old':
            p = os.path.join(DAILY_KLINE_DIR, f'{code}.csv')
            if os.path.exists(p):
                df = pd.read_csv(p, dtype={'date': str})
                df = df[['date', 'isST']]
                return df[(df['date'] >= start) & (df['date'] <= (end or '9999'))]
            # 旧库无此票（如新上市）→ 回落 baostock 扫描
        # baostock 单线程回扫（逐只 checkpoint，可断点续跑）
        import baostock as bs
        os.makedirs(ISST_DIR, exist_ok=True)
        path = os.path.join(ISST_DIR, f'{code}.parquet')
        have = pd.read_parquet(path) if os.path.exists(path) else pd.DataFrame(columns=['date', 'isST'])
        have = have.astype({'date': str})
        tgt_end = end or pd.Timestamp.now().strftime('%Y-%m-%d')
        # 只补缺口段：isST 对非 ST 明确返回 0（数据可信），续传=从本地边界接着请求
        segs = []
        if len(have):
            lo, hi = str(have['date'].min()), str(have['date'].max())
            if hi < tgt_end:
                segs.append(((pd.to_datetime(hi) + pd.Timedelta(days=1)).strftime('%Y-%m-%d'), tgt_end))
            if lo > start:
                segs.append((start, (pd.to_datetime(lo) - pd.Timedelta(days=1)).strftime('%Y-%m-%d')))
        else:
            segs.append((start, tgt_end))
        if not segs:
            return have[(have['date'] >= start) & (have['date'] <= tgt_end)]
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
            new = pd.concat([new, df], ignore_index=True)
        full = pd.concat([have, new], ignore_index=True).drop_duplicates('date', keep='last').sort_values('date')
        full.to_parquet(path)
        return full[(full['date'] >= start) & (full['date'] <= tgt_end)]

    def _isst_by_name(self, codes):
        """日常更新：扶摇名字对比（快照落 aux，ST 前缀变化的票改当天 isST）。"""
        meta = pd.read_parquet(META_F)
        meta = meta[meta['thscode'].str.endswith(('.SH', '.SZ'))]
        code2name = {f"{x.split('.')[1].lower()}.{x.split('.')[0]}": n for x, n in zip(meta['thscode'], meta['name'])}
        today = pd.Timestamp.now().strftime('%Y-%m-%d')
        hist = pd.read_parquet(NAMES_F) if os.path.exists(NAMES_F) else pd.DataFrame(columns=['code', 'date', 'name'])
        last = hist.sort_values('date').groupby('code').tail(1).set_index('code')['name'].to_dict()
        snap = pd.DataFrame([{'code': c, 'date': today, 'name': n} for c, n in code2name.items()])
        hist = pd.concat([hist, snap], ignore_index=True).drop_duplicates(['code', 'date'], keep='last')
        os.makedirs(AUX_DIR, exist_ok=True)
        hist.to_parquet(NAMES_F)
        changed = [c for c in codes if c in code2name and _is_st_name(code2name[c]) != _is_st_name(last.get(c, ''))]
        target = self._target_date()
        filled = 0
        for code in codes:
            if code not in code2name:
                continue
            val = int(_is_st_name(code2name[code]))
            path = os.path.join(V2_DIR, f'{code}.csv')
            tail = read_last_lines(path, n_lines=1) if os.path.exists(path) else []
            parts = tail[-1].split(',') if tail else []
            if len(parts) < 13 or parts[0] != target:
                continue                       # 当天无行（如已退市/无数据）→ 无处标注
            cur = _num(parts[12])
            if cur is not None and int(cur) == val:
                continue                       # 当天行 isST 已正确 → 不重写文件（保持廉价）
            df = read_v2(code)
            if df.empty:
                continue
            # 与 rebuild 路径同规矩：ST 状态向后延续、首个交易日之前按 0；当天行以今日名单为准。
            # 整列归一化成 0/1 整数，避免 read_csv 把含空值的列读成 float → 写成 '0.0' 这种混杂写法
            col = pd.to_numeric(df['isST'], errors='coerce').ffill().fillna(0).astype(int)
            col.loc[df['date'] == target] = val
            df['isST'] = col
            write_v2(code, df)
            filled += 1
        for code in changed:
            logger.info(f'[isst] 名字变化: {code} {last.get(code)} → {code2name[code]}')
        logger.info(f'[isst] update: 名单 {len(code2name)} 只，名字变化 {len(changed)} 只，'
                    f'补写当天 isST {filled} 只')

    # ==================== 内部：m15 ====================

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
                rows = []
                for it in arr:
                    day = it['day']
                    rows.append({'date': day[:10], 'time': int(day[11:13]) * 100 + int(day[14:16]),
                                 'open': float(it['open']), 'high': float(it['high']), 'low': float(it['low']),
                                 'close': float(it['close']), 'volume': int(float(it['volume'])),
                                 'amount': float(it['amount'])})
                time.sleep(random.uniform(0.02, 0.06))
                return pd.DataFrame(rows)
            except Exception as e:
                last = e
                time.sleep(random.uniform(0.5, 1.5))
        raise last

    def _m15_upsert(self, code, rows, force_merge=False):
        path = os.path.join(M15_KLINE_DIR, f'{code}.csv')
        cols = ['date', 'time', 'code', 'open', 'high', 'low', 'close', 'volume', 'amount']
        os.makedirs(M15_KLINE_DIR, exist_ok=True)   # to_csv 不会建父目录；空库首跑否则 OSError
        if not os.path.exists(path):
            rows.insert(1, 'code', code)
            rows[cols].to_csv(path, index=False)
            return True
        tail = read_last_lines(path, n_lines=1)
        last = (tail[-1].split(',') if tail else None)
        last_dt = (last[0], int(last[1])) if last and len(last) >= 2 else None
        new_rows = rows if last_dt is None else rows[[((r.date, r.time) > last_dt) for r in rows.itertuples()]]
        if len(new_rows) == 0 and not force_merge:
            return False
        if not force_merge and list(new_rows.index) == list(range(len(rows) - len(new_rows), len(rows))):
            out = new_rows.copy()   # 抓取窗与本地衔接（新区间=连续后缀）→ 直接追加
            out.insert(1, 'code', code)
            out[cols].to_csv(path, mode='a', header=False, index=False)
            return True
        old = pd.read_csv(path, dtype={'date': str})
        m = pd.concat([old, rows.assign(code=code)], ignore_index=True)
        m = m.drop_duplicates(['date', 'time'], keep='last').sort_values(['date', 'time'])
        m[cols].to_csv(path, index=False)
        return True

    # ==================== 内部：杂项 ====================

    def _load_prev_close(self):
        d = {}
        if self._prev_close_mem:
            d.update(self._prev_close_mem)
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


def _num(v):
    """CSV 字段 → float；空/nan/非数字 → None（'1' 与 '1.0' 都算 1）。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f else None


def _had_turn(path, n=60):
    """该票最近 n 行是否出现过 turn 值：区分 turn 缺失是「滞后」还是「结构性无量」。"""
    for ln in read_last_lines(path, n_lines=n):
        parts = ln.split(',')
        if len(parts) >= 13 and _num(parts[9]) is not None:
            return True
    return False


def is_latest(quiet=False):
    """v2 日线库各字段 + m15 是否都已更新到最近交易日（get_target_trade_date）。

    逐票抽末行检查（语义等同旧器 update_kline.py 的 all_latest）：末行 date 达目标日；当天真成交
    （tradestatus=1）的票 volume/amount/preclose/pctChg/turn 非空、isST 非空；当天有成交的票 m15
    末 bar 达目标日。豁免「源不提供、重跑也补不回来」的票，否则会永远报未达：退市
    （_aux/delisted.parquet）、无数据行（只有表头的空 CSV）、结构性无量（末 60 行 turn 全空，
    如 sh.689009 这类新浪/扶摇都给不出股本的标的）。
    判据是**全库**的，只回答「整库是否已齐」；某一步拿它当守卫（不关心别列）时传 quiet=True。
    """
    target = get_target_trade_date()
    if not target:
        return False
    tgt = f'{target[:4]}-{target[4:6]}-{target[6:8]}'
    delisted = set(pd.read_parquet(DELISTED_F)['code']) if os.path.exists(DELISTED_F) else set()
    lags, exempt = {}, {}
    files = sorted(f for f in os.listdir(V2_DIR) if f.endswith('.csv')) if os.path.isdir(V2_DIR) else []
    if not files:
        if not quiet:
            logger.warning(f'[is_latest] v2 库为空（{V2_DIR}）→ False')
        return False
    traded = set()
    for f in files:
        code = f[:-4]
        if code in delisted:
            continue   # 退市票：行情本就终结于退市日，以退市表豁免
        path = os.path.join(V2_DIR, f)
        tail = read_last_lines(path, n_lines=1)
        parts = tail[-1].split(',') if tail else []
        if len(parts) < 13 or not parts[0][:1].isdigit():
            exempt.setdefault('no_data', []).append(code)   # 空表/只有表头 → 源里本就没这票
            continue
        if parts[0] < tgt:
            lags.setdefault('date', []).append(code)
            continue
        if _num(parts[10]) == 1:   # 真成交行（'1' / '1.0' 都认）：量/额/前收/涨跌幅/换手必须有值
            traded.add(code)
            for name, i in (('volume', 7), ('amount', 8), ('preclose', 6), ('pctChg', 11)):
                if _num(parts[i]) is None:
                    lags.setdefault(name, []).append(code)
            if _num(parts[9]) is None:
                # turn 缺失：最近 60 行有过值=真滞后；从没有=源无股本，重跑也算不出 → 豁免
                if _had_turn(path):
                    lags.setdefault('turn', []).append(code)
                else:
                    exempt.setdefault('turn', []).append(code)
        if _num(parts[12]) is None:
            lags.setdefault('isST', []).append(code)
    if os.path.isdir(M15_KLINE_DIR):
        for code in sorted(traded):
            p = os.path.join(M15_KLINE_DIR, f'{code}.csv')
            tail = read_last_lines(p, n_lines=1) if os.path.exists(p) else []
            if not tail or tail[-1].split(',')[0] < tgt:
                lags.setdefault('m15', []).append(code)
    elif traded:
        lags['m15'] = sorted(traded)   # m15 库缺失：当日有行情的票全部计入滞后
    note = '，'.join(f'{k}={len(v)}' for k, v in sorted(exempt.items()))
    if not lags:
        if not quiet:
            logger.info(f'[is_latest] 全部字段已达 {tgt}（日线 {len(files)} 只'
                        + (f'；已豁免 {note}' if note else '') + '）')
        return True
    if not quiet:
        for k, v in sorted(lags.items()):
            logger.warning(f'[is_latest] {k} 未达 {tgt}: {len(v)} 只（样例: {v[:5]}）')
        if note:
            logger.info(f'[is_latest] 已豁免（源不提供，重跑无效）: {note}')
    return False


def main():
    ap = argparse.ArgumentParser(description='日K v2 构建器（rebuild/backfill/update）')
    ap.add_argument('--mode', required=True, choices=['rebuild', 'backfill', 'update'])
    ap.add_argument('--codes', default='', help='逗号分隔，如 600519,sz.000001（缺省=全市场沪深）')
    ap.add_argument('--start', default=None)
    ap.add_argument('--end', default=None)
    ap.add_argument('--isst-from', default='baostock', choices=['baostock', 'old'],
                    help='rebuild/backfill 的 isST 来源：baostock 回扫 或 旧库移植')
    ap.add_argument('--skip', default='', help='跳过列方法，如 m15,isst')
    ap.add_argument('--force', action='store_true')
    a = ap.parse_args()
    codes = norm_codes(a.codes) if a.codes else None
    builder = KlineV2Builder(isst_from=a.isst_from, force=a.force)
    builder.run(a.mode, codes, norm_date(a.start), norm_date(a.end),
                skip=[s for s in a.skip.split(',') if s])


if __name__ == '__main__':
    main()
