"""update_notrade —— 当前停牌名单（notrade_yet.csv）双轨更新器。

轨一 tushare（默认，新管线用）：suspend_d 官方停复牌，锚日规则——最近已收盘
    交易日当天出现在 S 清单 = 停牌中（含下一交易日"预停"记录）；notrade_date
    = 连续 S 段起点（按交易日历回溯）。
轨二 fuyao（备用/对账）：扶摇 daily-k-10d dump，在市票（stock_basic L）∩ 锚日
    无 bar = 停牌；起点按 dump 窗口（10 交易日）回溯，超窗记窗口首日。
旧法（--source daily）：本地日线尾部 tradestatus=0 段扫描（回退路径；缺点：
    退市/摘牌票不清理，会残留历史脏数据）。

用法:
  python update_notrade.py                  # tushare 轨全量重建（默认）
  python update_notrade.py --source fuyao   # fuyao 轨全量重建
  python update_notrade.py --compare        # tushare 写盘 + 与 fuyao 对账打印
"""
import argparse
import io
import json
import os

import pandas as pd
import requests
import tqdm

from AshareData.datautils.kline_scripts.tushare.tushare_kline import _pro as _ts_pro
from AshareData.paths import ASHARE_ROOT, DAILY_KLINE_TS_DIR, INFO_DIR
from AshareData.utils.exchanges_utils.a_open import get_target_trade_date, get_trade_date_list
from AshareData.utils.log_util import get_logger
from AshareData.utils.read_file_utils import read_csv_by_tail

logger = get_logger('更新停牌数据')

FUYAO_BASE = 'https://fuyao.aicubes.cn'
FUYAO_KEY_FILE = os.path.join(ASHARE_ROOT, '.keys', '.fuyao_api.json')


def _local_code(ts_code):
    """tushare '600363.SH' → 本地 'sh.600363'。"""
    c, ex = ts_code.split('.')
    return f'{ex.lower()}.{c}'


def _fuyao_key():
    """扶摇 API key（JSON 单值文件）。"""
    if not os.path.exists(FUYAO_KEY_FILE):
        raise RuntimeError(f'缺少扶摇 API key 文件: {FUYAO_KEY_FILE}')
    key = next((v for v in json.load(open(FUYAO_KEY_FILE)).values()
                if isinstance(v, str) and len(v) > 10), None)
    if not key:
        raise RuntimeError(f'扶摇 API key 文件里没有可用 key: {FUYAO_KEY_FILE}')
    return key


# 仅旧法（--source daily）使用的历史人工补充（新轨已由官方数据覆盖，不再需要）
manual_update = [
    ('sz.300029', '2026-07-11'),
    ('sz.000004', '2026-07-14'),
    ('sz.002808', '2026-07-14'),
    ('sz.002898', '2026-07-17'),
    ('sz.300391', '2026-07-17'),
    ('sz.300344', '2026-07-17')
]

class UpdateNotrade:
    """Tracks stocks that are currently suspended (tradestatus=0 at end of history)."""

    NOTRADE_CSV = f'{INFO_DIR}/notrade_yet.csv'
    COLUMNS = ['code', 'notrade_date']
    # Number of tail rows to scan per stock CSV when detecting suspension runs.
    TAIL_SCAN_ROWS = 500

    def __init__(self, force_init=False, source='tushare'):
        self._source = source
        self.daily_database = DAILY_KLINE_TS_DIR   # 旧法（--source daily）读 ts 日线
        self.notrade_dir = os.path.dirname(self.NOTRADE_CSV)
        self.df_notrade_yet = self._load_notrade_yet(force_init=force_init)
        self.changed = False

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------

    def init_notrade_yet(self, source=None):
        """全量重建 notrade_yet（source: tushare 默认 | fuyao | daily）。

        tushare：suspend_d 锚日规则（最近已收盘交易日当天在 S 清单 = 停牌中，
                  含下一交易日"预停"记录；notrade_date = 连续 S 段起点）。
        fuyao ：扶摇 daily-k-10d dump，在市票（stock_basic L）∩ 锚日无 bar = 停牌；
                起点按 dump 窗口回溯（超窗记窗口首日）。
        daily ：旧法——本地日线尾部 tradestatus=0 段扫描（回退用；退市票不清理）。
        """
        source = source or self._source
        if source == 'tushare':
            records = self._fetch_tushare()
        elif source == 'fuyao':
            records = self._fetch_fuyao()
        elif source == 'daily':
            records = self._scan_daily()
        else:
            raise ValueError(f'未知 source: {source}')

        df = pd.DataFrame(records, columns=self.COLUMNS).sort_values('code').reset_index(drop=True)
        os.makedirs(self.notrade_dir, exist_ok=True)
        df.to_csv(self.NOTRADE_CSV, index=False)
        self.df_notrade_yet = df.copy()
        self.changed = False
        logger.info(f'init_notrade_yet({source}) 完成: 共 {len(df)} 支停牌股票, 已写入 {self.NOTRADE_CSV}')
        return df

    def _scan_daily(self):
        """旧法（回退）：扫描全部日线尾部 tradestatus=0 连续段（含 manual_update 补充）。

        A stock is considered currently suspended when the last known row has
        tradestatus=0.  We walk the tail backwards to find the first consecutive
        row with tradestatus=0 — that date is recorded as notrade_date.
        """
        all_files = [f for f in os.listdir(self.daily_database) if f.endswith('.csv')]
        records = []
        for f in tqdm.tqdm(all_files, desc='扫描停牌状态'):
            code = f.rsplit('.', 1)[0]
            file_path = os.path.join(self.daily_database, f)
            notrade_date = self._detect_notrade_start(file_path)
            if notrade_date is not None:
                records.append({'code': code, 'notrade_date': notrade_date})

        #manual updates
        for code, notrade_date in manual_update:
            records.append({'code': code, 'notrade_date': notrade_date})
        return records

    def _fetch_tushare(self):
        """tushare suspend_d 轨：锚日 S 集合 = 当前停牌；notrade_date = 连续 S 段起点。"""
        pro = _ts_pro()
        anchor = pd.to_datetime(str(get_target_trade_date())).strftime('%Y-%m-%d')
        cal = [pd.to_datetime(str(d)).strftime('%Y-%m-%d') for d in get_trade_date_list()]
        t0 = pd.Timestamp(anchor)
        frames = []
        for s, e in [(t0 - pd.Timedelta(days=220), t0 - pd.Timedelta(days=20)),
                     (t0 - pd.Timedelta(days=21), t0 + pd.Timedelta(days=10))]:
            frames.append(pro.suspend_d(suspend_type='S',
                                        start_date=s.strftime('%Y%m%d'),
                                        end_date=e.strftime('%Y%m%d')))
        ev = pd.concat(frames, ignore_index=True).drop_duplicates()
        ev['td'] = pd.to_datetime(ev['trade_date']).dt.strftime('%Y-%m-%d')
        records = []
        for code, g in ev.groupby('ts_code'):
            ds = set(g['td'])
            latest = max(ds)
            if anchor not in ds and latest <= anchor:
                continue                       # 锚日无 S 且无未来预停 → 已复牌/摘牌
            cur = latest if latest > anchor else anchor
            i = cal.index(cur)
            while i - 1 >= 0 and cal[i - 1] in ds:
                i -= 1
            records.append((_local_code(code), cal[i]))
        logger.info(f'[tushare] suspend_d {len(ev)} 事件 / {ev.ts_code.nunique()} 票 → 当前停牌 {len(records)} 只')
        return records

    def _fetch_fuyao(self):
        """扶摇 dump 轨：在市票 ∩ 锚日无 bar = 停牌；起点按 dump 窗口回溯（超窗记窗口首日）。"""
        anchor = pd.to_datetime(str(get_target_trade_date())).strftime('%Y-%m-%d')
        s = requests.Session()
        s.headers['X-api-key'] = _fuyao_key()
        u = s.get(f'{FUYAO_BASE}/api/dump/market-dumps/daily-k-10d/download-url', timeout=60).json()
        url = (u.get('data') or {}).get('presigned_url') or ''
        if not url:
            raise RuntimeError(f'扶摇 dump 签名失败: {u.get("code")} {u.get("message")}')
        blob = requests.get(url, timeout=300).content
        d = pd.read_parquet(io.BytesIO(blob))
        d['date'] = (pd.to_datetime(d['date_ms'], unit='ms', utc=True)
                     .dt.tz_convert('Asia/Shanghai').dt.strftime('%Y-%m-%d'))
        days = sorted(d['date'].unique())
        eff = anchor if anchor in days else days[-1]
        if eff != anchor:
            logger.warning(f'[fuyao] dump 最新日 {eff} < 锚日 {anchor}，按 {eff} 判定')
        live = set(_ts_pro().stock_basic(exchange='', list_status='L')['ts_code'])
        bars = set(d.loc[d['date'] == eff, 'thscode'])
        ii = days.index(eff)
        records = []
        for tc in sorted(live - bars):
            ds = set(d.loc[d['thscode'] == tc, 'date'])
            i = ii
            while i - 1 >= 0 and days[i - 1] in ds:
                i -= 1
            records.append((_local_code(tc), days[i]))
        capped = sum(1 for _, dt in records if dt == days[0])
        logger.info(f'[fuyao] 在市 {len(live)} / 锚日有 bar {len(bars & live)} → 当前停牌 {len(records)} 只'
                    + (f'（{capped} 只起点超窗记窗口首日 {days[0]}）' if capped else ''))
        return records

    def compare_sources(self):
        """两轨对账：tushare（写盘为准）vs fuyao，差异打日志。"""
        df = self.init_notrade_yet(source='tushare')
        t_codes = set(df['code'])
        f_codes = {c for c, _ in self._fetch_fuyao()}
        logger.info(f'[对账] tushare {len(t_codes)} vs fuyao {len(f_codes)} | 交集 {len(t_codes & f_codes)}')
        logger.info(f'[对账] 仅 tushare（含下一交易日预停）: {sorted(t_codes - f_codes)}')
        logger.info(f'[对账] 仅 fuyao: {sorted(f_codes - t_codes)}')
        return df

    def get_notrade_yet(self):
        """Return in-memory notrade_yet table (code, notrade_date)."""
        return self.df_notrade_yet

    def set_notrade_yet(self, df=None, code=None, notrade_date=None, delete=None, query_range_days=0):
        """Update in-memory notrade_yet table with explicit parameters.

        Modes:
        - DataFrame mode: provide df (columns code/date/tradestatus).
        - Single mode: provide code and (notrade_date or delete=True).

        Extra rule:
        - If df is empty and query_range_days > 30, treat as suspended and upsert
          this code with notrade_date (fallback to today if not provided).
        """
        if df is not None:
            if not isinstance(df, pd.DataFrame):
                raise ValueError('df must be a pandas DataFrame when provided')
            if delete is not None:
                raise ValueError('delete must be None when df is provided')

            changed = False
            if not df.empty:
                changed = self._set_notrade_yet_from_df(df)
            else:
                range_days = int(query_range_days or 0)
                if range_days > 30:
                    if code is None or str(code).strip() == '':
                        logger.info('set_notrade_yet(df-empty): missing code, skip long-range suspension update')
                    else:
                        inferred_date = notrade_date
                        if inferred_date is None:
                            inferred_date = pd.to_datetime('today').strftime('%Y-%m-%d')
                        changed = self._set_notrade_yet_single(
                            code=str(code),
                            notrade_date=str(inferred_date),
                            delete=False,
                        )
                        logger.info(
                            f'set_notrade_yet(df-empty): code={code}, query_range_days={range_days}, '
                            f'inferred_notrade_date={inferred_date}, changed={changed}'
                        )

            if changed:
                self.changed = True
            logger.info(
                f'set_notrade_yet(df): rows={len(df)}, code={code}, query_range_days={query_range_days}, '
                f'changed={changed}, pending_flush={self.changed}'
            )
            return changed

        if code is None or str(code).strip() == '':
            raise ValueError('code is required when df is None')

        delete_flag = False if delete is None else bool(delete)
        if not delete_flag and notrade_date is None:
            raise ValueError('notrade_date is required when df is None and delete is not True')

        changed = self._set_notrade_yet_single(
            code=str(code),
            notrade_date=None if notrade_date is None else str(notrade_date),
            delete=delete_flag,
        )
        if changed:
            self.changed = True
        logger.info(
            f'set_notrade_yet: code={code}, notrade_date={notrade_date}, '
            f'delete={delete_flag}, changed={changed}, pending_flush={self.changed}'
        )
        return changed

    def _set_notrade_yet_single(self, code: str, notrade_date: str = None, delete: bool = False):
        """Apply single-code update and return whether data changed."""
        df = self.df_notrade_yet
        code = str(code)
        notrade_date = '' if notrade_date is None else str(notrade_date)
        mask = df['code'] == code
        changed = False

        if delete:
            if mask.any():
                self.df_notrade_yet = df.loc[~mask].reset_index(drop=True)
                logger.info(f'set_notrade_yet(delete): removed code={code}')
                return True
            logger.info(f'set_notrade_yet(delete): code not found, skip code={code}')
            return False

        if mask.any():
            old_date = str(df.loc[mask, 'notrade_date'].iloc[0])
            old_dt = pd.to_datetime(old_date, errors='coerce')
            new_dt = pd.to_datetime(notrade_date, errors='coerce')
            if pd.notna(old_dt) and pd.notna(new_dt):
                keep_date = old_dt.strftime('%Y-%m-%d') if old_dt <= new_dt else new_dt.strftime('%Y-%m-%d')
            elif pd.notna(old_dt):
                keep_date = old_dt.strftime('%Y-%m-%d')
            elif pd.notna(new_dt):
                keep_date = new_dt.strftime('%Y-%m-%d')
            else:
                keep_date = old_date if old_date <= notrade_date else notrade_date
            if old_date != keep_date:
                df.loc[mask, 'notrade_date'] = keep_date
                changed = True
        else:
            new_row = pd.DataFrame([{'code': code, 'notrade_date': notrade_date}])
            self.df_notrade_yet = pd.concat([df, new_row], ignore_index=True).reset_index(drop=True)
            return True

        self.df_notrade_yet = df.reset_index(drop=True)
        return changed

    def _set_notrade_yet_from_df(self, updates_df: pd.DataFrame):
        """Apply batch updates from DataFrame(code, date, tradestatus)."""
        required_cols = {'code', 'date', 'tradestatus'}
        if updates_df is None or updates_df.empty:
            return False
        if not required_cols.issubset(set(updates_df.columns)):
            missing = sorted(required_cols - set(updates_df.columns))
            raise ValueError(f'updates_df missing required columns: {missing}')

        work = updates_df[['code', 'date', 'tradestatus']].copy()
        work['code'] = work['code'].astype('string').fillna('').str.strip()
        work['date_dt'] = pd.to_datetime(work['date'], errors='coerce')
        work['tradestatus'] = pd.to_numeric(work['tradestatus'], errors='coerce').fillna(1).astype(int)
        work = work[(work['code'] != '') & work['date_dt'].notna()].copy()
        if work.empty:
            return False

        work = work.sort_values(['code', 'date_dt']).reset_index(drop=True)
        changed_any = False

        for code, grp in work.groupby('code', sort=False):
            grp = grp.reset_index(drop=True)
            statuses = grp['tradestatus'].tolist()
            latest_status = statuses[-1]

            if latest_status != 0:
                changed_any = self._set_notrade_yet_single(code=code, delete=True) or changed_any
                continue

            start_idx = len(grp) - 1
            for i in range(len(grp) - 2, -1, -1):
                if int(grp.iloc[i]['tradestatus']) != 0:
                    start_idx = i + 1
                    break
                start_idx = i

            start_date = grp.iloc[start_idx]['date_dt'].strftime('%Y-%m-%d')
            has_nonzero_before = bool((grp.iloc[:start_idx]['tradestatus'] != 0).any()) if start_idx > 0 else False

            if has_nonzero_before:
                df = self.df_notrade_yet
                mask = df['code'] == code
                if mask.any():
                    old_date = str(df.loc[mask, 'notrade_date'].iloc[0])
                    if old_date != start_date:
                        df.loc[mask, 'notrade_date'] = start_date
                        self.df_notrade_yet = df.reset_index(drop=True)
                        changed_any = True
                else:
                    changed_any = self._set_notrade_yet_single(code=code, notrade_date=start_date, delete=False) or changed_any
            else:
                changed_any = self._set_notrade_yet_single(code=code, notrade_date=start_date, delete=False) or changed_any

        return changed_any

    def flush_notrade_yet(self, force: bool = False):
        """Persist in-memory notrade_yet table to disk.

        Writes only when changes exist unless force=True.
        """
        if not force and not self.changed:
            logger.info('flush_notrade_yet: no changes, skip write')
            return False

        os.makedirs(self.notrade_dir, exist_ok=True)
        self.df_notrade_yet.to_csv(self.NOTRADE_CSV, index=False)
        self.changed = False
        logger.info(f'flush_notrade_yet: wrote {len(self.df_notrade_yet)} rows to {self.NOTRADE_CSV}')
        return True

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    def _load_notrade_yet(self, force_init=False):
        """Load notrade_yet from disk once during initialization.

        Behavior:
        - If NOTRADE_CSV directory does not exist, call init_notrade_yet.
        - If directory exists but file missing/empty, return empty DataFrame.
        """
        if not os.path.isdir(self.notrade_dir) or force_init:
            return self.init_notrade_yet(source=self._source)
        if not os.path.exists(self.NOTRADE_CSV) or os.path.getsize(self.NOTRADE_CSV) == 0:
            return pd.DataFrame(columns=self.COLUMNS)
        return pd.read_csv(self.NOTRADE_CSV, dtype=str)

    def _detect_notrade_start(self, file_path: str):
        """Return the start date of the trailing suspension run, or None.

        Reads up to TAIL_SCAN_ROWS rows from the tail of the CSV and walks
        backward to find the beginning of a consecutive tradestatus=0 block.
        If the last row is not suspended, returns None.
        """
        try:
            df = read_csv_by_tail(file_path, n_lines=self.TAIL_SCAN_ROWS)
        except Exception:
            return None

        if df is None or df.empty or 'tradestatus' not in df.columns or 'date' not in df.columns:
            return None

        tradestatus = pd.to_numeric(df['tradestatus'], errors='coerce').fillna(1)
        # Only interested when the very last record is suspended
        if tradestatus.iloc[-1] != 0:
            return None

        # Walk backward to find first row that is NOT 0 — suspension starts one
        # row after that (or at row 0 if all scanned rows are 0).
        first_suspended_idx = 0
        for i in range(len(tradestatus) - 2, -1, -1):
            if tradestatus.iloc[i] != 0:
                first_suspended_idx = i + 1
                break

        notrade_date = df['date'].iloc[first_suspended_idx]
        try:
            return pd.to_datetime(notrade_date).strftime('%Y-%m-%d')
        except Exception:
            return str(notrade_date)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='停牌名单更新（双轨：tushare 默认 / fuyao；daily 为旧法回退）')
    ap.add_argument('--source', default='tushare', choices=['tushare', 'fuyao', 'daily'])
    ap.add_argument('--compare', action='store_true', help='tushare 写盘并与 fuyao 对账打印')
    a = ap.parse_args()
    updater = UpdateNotrade(source=a.source)
    if a.compare:
        updater.compare_sources()
    else:
        updater.init_notrade_yet(source=a.source)