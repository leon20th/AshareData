# news_features —— 新闻快讯标签计数特征（东财 7x24 + 财联社电报）→ built_data/news_features.parquet
# 数据源: scrap_data/news_em724/*.json（条目 stockList 带 0./1. 六位码）
#         scrap_data/news_cls/*.json（条目 stock_list 带 {'StockID':'sz000100'}）
# 口径: 按 (trade_date, code) 聚合"该交易日收盘(15:00)前"的快讯；
#       时间戳 >15:00 的快讯归入下一交易日（非交易日同理）⇒ 无未来函数。
# 特征（NW_COLUMNS，8 列，全 [0,1] 量纲，仿 kpl_features 的 O(1) 计数族）:
#   nw_em   当日该股被东财 7x24 提及条数
#   nw_cls  当日该股被财联社电报提及条数
#   nw_cnt  合计提及条数
#   nw_rec  最近一条提及的收盘前时间衰减 exp(-Δt/240min)
#   nw_late 收盘前 1 小时内提及条数（临收盘消息）
#   nw_3d   近 3 个交易日提及合计（持续关注度）
#   nw_nov  新面孔：过去 5 个交易日无提及（对位 kp_nf）
#   nw_subj 所属 KPL 题材当日在财联社 subjects 里的热度（题材共振）
# 注（2026-10-06 消融）：另试过 nw_pre/nw_mid（盘前/盘中拆分，seed 不稳）、
#   nw_rel/nw_burst（相对热度/突增，负向）→ 均未入列，仅保留以上 8 列。
import json
import os
from bisect import bisect_left
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from AshareData.paths import BUILT_DATA_DIR, SCRAP_DATA_DIR
from AshareData.utils.exchanges_utils.a_open import get_trade_date_idx, get_trade_date_list
from AshareData.datautils.dataloaders.feature_build.feature_utils import get_code_idx

EM_DIR = f'{SCRAP_DATA_DIR}/news_em724'
CLS_DIR = f'{SCRAP_DATA_DIR}/news_cls'
NEWS_PARQUET = f'{BUILT_DATA_DIR}/news_features.parquet'

NW_COLUMNS = ['nw_em', 'nw_cls', 'nw_cnt', 'nw_rec', 'nw_late', 'nw_3d', 'nw_nov', 'nw_subj']

_CLOSE_HOUR = 15
_TAU_MIN = 240.0     # nw_rec 衰减尺度
_LATE_MIN = 60.0     # nw_late：收盘前 1 小时
_TZ_CN = timezone(timedelta(hours=8))


def _trade_close_ts(start: str, end: str):
    """[start,end] 内的交易日 → (dates, close_ts naive datetime 升序)。"""
    dates = [d for d in sorted(get_trade_date_list()) if start <= d <= end]
    close = [datetime.strptime(d, '%Y%m%d').replace(hour=_CLOSE_HOUR) for d in dates]
    return dates, close


def _em_codes(d: dict) -> list:
    out = []
    for s in d.get('stockList') or []:
        if len(s) == 8 and s[1] == '.' and s[0] in '01' and s[2:].isdigit():
            out.append(s[2:])
    return out


def _cls_codes(d: dict) -> list:
    out = []
    for s in d.get('stock_list') or []:
        sid = str((s or {}).get('StockID') or '')
        if len(sid) == 8 and sid[:2] in ('sh', 'sz') and sid[2:].isdigit():
            out.append(sid[2:])
    return out


def _iter_news(start: str, end: str):
    """产出 (ts naive, codes, source, subjects, text)；source ∈ {'em','cls'}。

    无个股代码但带 subjects 的条目也会产出（供题材热度统计，text 为空）。
    """
    for src, folder in (('em', EM_DIR), ('cls', CLS_DIR)):
        if not os.path.isdir(folder):
            continue
        for fn in sorted(f for f in os.listdir(folder) if f.endswith('.json') and f[:4].isdigit()):
            day = fn[:10].replace('-', '')
            if not (start <= day <= end):
                continue
            try:
                items = json.load(open(f'{folder}/{fn}', encoding='utf-8'))
            except (ValueError, OSError):
                continue
            if not isinstance(items, list):
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                if src == 'em':
                    st = it.get('showTime') or ''
                    try:
                        ts = datetime.strptime(st, '%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        continue
                    codes, subs = _em_codes(it), ()
                    text = ((it.get('title') or '') + ' ' + (it.get('summary') or '')).strip()
                else:
                    try:
                        ts = datetime.fromtimestamp(int(it.get('ctime')), tz=_TZ_CN).replace(tzinfo=None)
                    except (TypeError, ValueError, OSError):
                        continue
                    codes = _cls_codes(it)
                    subs = tuple(it.get('subjects') or [])
                    text = ((it.get('title') or '') + ' ' + (it.get('content') or '')).strip()
                if codes or subs:
                    yield ts, codes, src, subs, text


_THEME_NAMES: set | None = None
_NAME_HITS: dict = {}


def _kpl_theme_names() -> set:
    """KPL 题材名集合（实体 name_cur ∪ 别名 zs_name；剔泛词/单字）。"""
    global _THEME_NAMES
    if _THEME_NAMES is None:
        names: set = set()
        for f, col in (('kpl_theme_entities.parquet', 'name_cur'),
                       ('kpl_theme_aliases.parquet', 'zs_name')):
            try:
                names |= set(pd.read_parquet(f'{BUILT_DATA_DIR}/{f}', columns=[col])[col].astype(str))
            except (OSError, KeyError):
                pass
        _THEME_NAMES = {n for n in names if len(n) >= 2 and n not in ('其他', '其他概念')}
    return _THEME_NAMES


def _names_in(subject: str) -> tuple:
    """subject 命中的 KPL 题材名（带缓存；subjects 词表有限）。"""
    hit = _NAME_HITS.get(subject)
    if hit is None:
        hit = tuple(n for n in _kpl_theme_names() if n in subject)
        _NAME_HITS[subject] = hit
    return hit


def _subj_heat_table(subj_heat) -> pd.DataFrame:
    """(date, code) → 该股所属 KPL 题材当日的 subjects 热度（多题材取最大）。"""
    empty = pd.DataFrame({'date': pd.Series(dtype=str), 'code': pd.Series(dtype=str),
                          'subj': pd.Series(dtype=float)})
    if not subj_heat:
        return empty
    try:
        m = pd.read_parquet(f'{BUILT_DATA_DIR}/kpl_theme_members.parquet',
                            columns=['date', 'code', 'entity_id'])
        ent = pd.read_parquet(f'{BUILT_DATA_DIR}/kpl_theme_entities.parquet',
                              columns=['entity_id', 'name_cur'])
    except (OSError, KeyError):
        return empty
    m = m.copy()
    m['date'] = m.date.astype(str)
    m['code'] = m.code.astype(str)
    m = m.merge(ent, on='entity_id', how='left')
    m['subj'] = [subj_heat.get((d, n), 0) for d, n in zip(m.date, m.name_cur.fillna(''))]
    return m[m.subj > 0].groupby(['date', 'code'])['subj'].max().reset_index()


def build_news_features(force_rebuild: bool = False, start: str = '20200101',
                        end: str | None = None) -> pd.DataFrame:
    """新闻快讯 → news_features.parquet。返回特征表（trade_date_idx, code_idx, nw_*）。"""
    if end is None:
        end = datetime.now().strftime('%Y%m%d')
    if not force_rebuild and os.path.exists(NEWS_PARQUET):
        cur = pd.read_parquet(NEWS_PARQUET)
        print(f'[news] 已存在 {NEWS_PARQUET}（{len(cur)} 行）；force_rebuild=True 可重算')
        return cur

    dates, close = _trade_close_ts(start, end)
    if not dates:
        raise RuntimeError(f'[{start},{end}] 内无交易日')
    d2i = {d: i for i, d in enumerate(dates)}

    # (trade_date, code) → [em, cls, late, min_delta]
    acc: dict[tuple[str, str], list] = defaultdict(lambda: [0, 0, 0, np.inf])
    subj_heat: dict[tuple[str, str], int] = defaultdict(int)     # (date, 题材名) → 当日被提及条数
    n_item = 0
    for ts, codes, src, subs, _text in _iter_news(start, end):
        i = bisect_left(close, ts)
        if i >= len(close):
            continue
        d = dates[i]
        dm = (close[i] - ts).total_seconds() / 60.0
        for s in subs:
            for n in _names_in(s):
                subj_heat[(d, n)] += 1
        for c in codes:
            a = acc[(d, c)]
            a[0 if src == 'em' else 1] += 1
            if dm <= _LATE_MIN:
                a[2] += 1
            if dm < a[3]:
                a[3] = dm
            n_item += 1

    rows = [(d, c, a[0], a[1], a[2], a[3]) for (d, c), a in acc.items()]
    df = pd.DataFrame(rows, columns=['date', 'code', 'em', 'cls', 'late', 'last_min'])
    df['cnt'] = df.em + df.cls

    # ---- 跨日：近 3 日合计 / 新面孔（全局交易日索引差 ≈ 交易日间隔）
    df['tdi'] = df.date.map(lambda d: d2i[d])
    df = df.sort_values(['code', 'tdi']).reset_index(drop=True)
    sum3 = np.zeros(len(df), dtype=np.float32)
    nov = np.ones(len(df), dtype=np.float32)
    for _, idx in df.groupby('code', sort=False).indices.items():
        idx = np.sort(idx)
        t, c = df.tdi.values[idx], df.cnt.values[idx]
        for j in range(len(idx)):
            lo3 = np.searchsorted(t, t[j] - 2, side='left')
            sum3[idx[j]] = c[lo3:j + 1].sum()
            nov[idx[j]] = 1.0 if j == 0 or (t[j] - t[j - 1]) > 5 else 0.0
    df['sum3'], df['nov'] = sum3, nov

    # ---- 题材热度（KPL 题材 × CLS subjects）
    df = df.merge(_subj_heat_table(subj_heat), on=['date', 'code'], how='left')
    df['subj'] = df['subj'].fillna(0.0) if 'subj' in df.columns else 0.0

    # ---- 归一化（全 [0,1]，与 kp_* 同风格）
    df['nw_em'] = np.log1p(df.em) / np.log1p(20.0)
    df['nw_cls'] = np.log1p(df.cls) / np.log1p(20.0)
    df['nw_cnt'] = np.log1p(df.cnt) / np.log1p(40.0)
    df['nw_rec'] = np.exp(-df.last_min / _TAU_MIN)
    df['nw_late'] = np.log1p(df.late) / np.log1p(10.0)
    df['nw_3d'] = np.log1p(df.sum3) / np.log1p(100.0)
    df['nw_nov'] = df.nov
    df['nw_subj'] = np.log1p(df.subj) / np.log1p(50.0)

    # ---- 全局索引映射
    t2i = {d: get_trade_date_idx(d) for d in dates}
    c2i = dict(zip(df.code.unique(), get_code_idx(list(df.code.unique()))))
    df['trade_date_idx'] = df.date.map(t2i)
    df['code_idx'] = df.code.map(c2i)
    K = df[(df.trade_date_idx >= 0) & (df.code_idx >= 0)].copy()
    print(f'[news] {n_item} 条提及 → {len(df)} (date,code) 行；映射 {len(K)} 行')
    K = K[['trade_date_idx', 'code_idx'] + NW_COLUMNS]
    K['trade_date_idx'] = K.trade_date_idx.astype('int64')
    K['code_idx'] = K.code_idx.astype('int64')
    K = K.sort_values(['trade_date_idx', 'code_idx']).reset_index(drop=True)
    K.to_parquet(NEWS_PARQUET, index=False)
    print(f'[news] → {NEWS_PARQUET}（{len(K)} 行，{len(NW_COLUMNS)} 列）')
    return K


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--start', default='20200101')
    ap.add_argument('--end', default=None)
    a = ap.parse_args()
    K = build_news_features(force_rebuild=a.force, start=a.start, end=a.end)
    print(K.describe().to_string())
