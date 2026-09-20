# kpl_features —— 开盘啦事件结构特征（co 日内共现 / chain 跨日连边）→ built_data
# 流程: scrap_data/kaipanla/*.json → kpl_events.parquet（原始事件，增量）
#        → kp_* 8 列（co: 组规模/组内板数分位/首板占比；chain: 轮龄/轮首日/间隔/5日再现/新面孔）
#        → kpl_edges.parquet（题材-个股边表，供图模型）+ 合并进 event_feat.parquet（ev_* 列原样保留）
# 口径: 2019+ 涨停复盘事件; 剔除 zscode=0("其他"); 遗留码段(885/881/880/882)按题材名归一到 801 主码;
#        episode = 同题材相邻出演间隔 ≤3 交易日为同一轮
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

from AshareData.paths import BUILT_DATA_DIR, SCRAP_DATA_DIR
from AshareData.datautils.dataloaders.feature_build.feature_utils import (
    get_code_idx, get_trade_date_idx)

KPL_RAW_DIR = f'{SCRAP_DATA_DIR}/kaipanla'
KPL_EVENTS_PARQUET = f'{BUILT_DATA_DIR}/kpl_events.parquet'
KPL_EDGES_PARQUET = f'{BUILT_DATA_DIR}/kpl_edges.parquet'
EVENT_FEAT_PARQUET = f'{BUILT_DATA_DIR}/event_feat.parquet'

EVENT_COLUMNS = ['date', 'code', 'zscode', 'zs_name', 'segment', 'industry', 'board_txt', 'boards', 'reason_txt']
KP_COLUMNS = ['kp_sz', 'kp_rk', 'kp_fb', 'kp_age', 'kp_new', 'kp_gap', 'kp_rev', 'kp_nf']


def _parse_json(fp: str) -> list:
    """单个 kpl JSON → 事件行。StockList 定长字段: [0]code [9]板数文本 [10]板数 [11]行业标签 [16]细分题材 [17]原因长文"""
    d8 = os.path.basename(fp)[:10].replace('-', '')
    rows = []
    for g in json.load(open(fp, encoding='utf-8')).get('list') or []:
        zs = (g.get('ZSName') or '').strip()
        for s in (g.get('StockList') or []):
            try:
                boards = int(float(s[10]))
            except (ValueError, IndexError):
                boards = 0
            rows.append((
                d8,
                str(s[0]),
                str(g.get('ZSCode')).strip(),
                zs,
                str(s[16]).strip() if len(s) > 16 else '',
                str(s[11]).strip() if len(s) > 11 else '',
                str(s[9]).strip() if len(s) > 9 else '',
                boards,
                str(s[17]).strip()[:400] if len(s) > 17 else '',
            ))
    return rows


def build_kpl_events(update: bool = False, force_rebuild: bool = False):
    """原始 JSON → kpl_events.parquet（增量：只解析不在表内的日期；force_rebuild 全量重来）。"""
    existing = None
    if not force_rebuild and os.path.exists(KPL_EVENTS_PARQUET):
        existing = pd.read_parquet(KPL_EVENTS_PARQUET)
        existing['date'] = existing.date.astype(str)
    if not update and not force_rebuild:
        return existing, 0
    files = sorted(f for f in os.listdir(KPL_RAW_DIR) if f.endswith('.json')) if os.path.isdir(KPL_RAW_DIR) else []
    if not files:
        print(f'[kpl] 无原始数据 {KPL_RAW_DIR}')
        return existing, 0
    have = set(existing.date.unique()) if existing is not None else set()
    todo = [f for f in files if f[:10].replace('-', '') not in have]
    if not todo:
        return existing, 0
    frames = [existing] if existing is not None else []
    frames += [pd.DataFrame(_parse_json(f'{KPL_RAW_DIR}/{f}'), columns=EVENT_COLUMNS) for f in todo]
    combined = (pd.concat(frames, ignore_index=True)
                .drop_duplicates(['date', 'code', 'zscode'], keep='last')
                .sort_values(['date', 'code', 'zscode']).reset_index(drop=True))
    combined.to_parquet(KPL_EVENTS_PARQUET, index=False)
    print(f'[kpl] events +{len(todo)}天 → {len(combined)}行 → kpl_events.parquet')
    return combined, len(todo)


def _compute_kp(E: pd.DataFrame):
    """事件表 → (kp 特征 [trade_date_idx, code_idx, kp_*], 边表)。"""
    E = E[E.date >= '20190101'].copy()
    E = E[E.zscode != '0'].copy()
    E['date'] = E.date.astype(int)
    E['code'] = E.code.astype(str)
    E['zscode'] = E.zscode.astype(str)

    # 遗留码按题材名归一到 801 主码
    canon = E[E.zscode.str.startswith('801')].groupby('zs_name')['zscode'].agg(lambda s: s.mode().iat[0])
    lg = ~E.zscode.str.startswith('801')
    E.loc[lg, 'zscode'] = E.loc[lg, 'zs_name'].map(canon).fillna(E.loc[lg, 'zscode'])

    dates = np.array(sorted(E.date.unique()))
    d2i = {d: i for i, d in enumerate(dates)}
    E['di'] = E.date.map(d2i)

    # ---- co: 组规模 / 组内板数分位 / 首板占比
    grp = E.groupby(['date', 'zscode'])
    E['sz'] = grp['code'].transform('size')
    rmin = grp['boards'].rank(method='min')
    E['rk'] = ((rmin - 1) / (E.sz - 1)).where(E.sz > 1, 0.0)
    E['fb'] = grp['boards'].transform(lambda b: (b == 1).mean())

    # ---- chain: episode 轮龄 / 轮首日 / 间隔 / 新面孔占比
    day_sets = {}
    for (zc, dt), cs in E.groupby(['zscode', 'date'])['code'].apply(frozenset).items():
        day_sets.setdefault(zc, []).append((dt, cs))
    age_of, new_of, gap_of, nf_of = {}, {}, {}, {}
    for zc, lst in day_sets.items():
        prev_d = prev_cs = None
        age = -1
        for dt, cs in lst:
            di = d2i[dt]
            age = age + 1 if (prev_d is not None and di - d2i[prev_d] <= 3) else 0
            age_of[(zc, dt)] = age
            new_of[(zc, dt)] = 1.0 if age == 0 else 0.0
            gap_of[(zc, dt)] = 1.0 if prev_d is None else min(np.log1p(di - d2i[prev_d]) / np.log1p(60.0), 1.0)
            if prev_d is not None and di - d2i[prev_d] <= 5 and prev_cs:
                nf_of[(zc, dt)] = 1.0 - len(cs & prev_cs) / len(cs)
            else:
                nf_of[(zc, dt)] = 1.0
            prev_d, prev_cs = dt, cs
    key = list(zip(E.zscode, E.date))
    E['age'] = [age_of[k] for k in key]
    E['new'] = [new_of[k] for k in key]
    E['gap'] = [gap_of[k] for k in key]
    E['nf'] = [nf_of[k] for k in key]

    # ---- rev: 个股-题材 5 日内再现次数
    rev = np.zeros(len(E), dtype=np.float32)
    hist = defaultdict(list)
    for i in np.lexsort((E.di.values, E.code.values, E.zscode.values)):
        zc, cd, d = E.zscode.iat[i], E.code.iat[i], int(E.di.iat[i])
        h = hist[(zc, cd)]
        cnt = 0
        for p in reversed(h):
            if d - p <= 5:
                cnt += 1
            else:
                break
        rev[i] = min(cnt, 5) / 5.0
        h.append(d)
    E['rev'] = rev

    E['kp_sz'] = np.log1p(E.sz) / np.log1p(200.0)
    E['kp_rk'] = E.rk.astype(float)
    E['kp_fb'] = E.fb.astype(float)
    E['kp_age'] = np.minimum(E.age, 20) / 20.0
    E['kp_new'] = E.new
    E['kp_gap'] = E.gap
    E['kp_rev'] = E.rev
    E['kp_nf'] = E.nf

    # ---- join (trade_date_idx, code_idx)：原生全局索引，未映射行剔除
    t2i = dict(zip(dates, get_trade_date_idx(list(dates))))
    c2i = dict(zip(E.code.unique(), get_code_idx(list(E.code.unique()))))
    E['tdi'] = E.date.map(t2i)
    E['cidx'] = E.code.map(c2i)
    K = E[(E.tdi >= 0) & (E.cidx >= 0)]
    print(f'[kpl] 特征 {len(K)}/{len(E)} 行映射 ({len(K) / len(E):.1%})')

    edges = E[['date', 'zscode', 'zs_name', 'code', 'boards', 'sz']].copy()
    edges['date'] = edges.date.astype(str)
    edges = edges.sort_values(['date', 'zscode', 'boards', 'code']).reset_index(drop=True)
    K = K[['tdi', 'cidx'] + KP_COLUMNS].rename(columns={'tdi': 'trade_date_idx', 'cidx': 'code_idx'})
    K['trade_date_idx'] = K.trade_date_idx.astype('int64')
    K['code_idx'] = K.code_idx.astype('int64')
    return K, edges


def _merge_event_feat(K: pd.DataFrame) -> pd.DataFrame:
    """kp 列并入 event_feat.parquet：ev_* 列原样保留，kp 列整体替换；无事件日补 0。"""
    feat = pd.read_parquet(EVENT_FEAT_PARQUET) if os.path.exists(EVENT_FEAT_PARQUET) else None
    ev_cols = [c for c in feat.columns if c.startswith('ev_')] if feat is not None else []
    if feat is None:
        out = K.copy()
    else:
        base = feat.drop(columns=[c for c in KP_COLUMNS if c in feat.columns]).copy()
        base['trade_date_idx'] = base.trade_date_idx.astype('int64')
        base['code_idx'] = base.code_idx.astype('int64')
        out = base.merge(K, on=['trade_date_idx', 'code_idx'], how='outer')
    for c in ev_cols + KP_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0).astype(np.float32)
    out = out[['trade_date_idx', 'code_idx'] + ev_cols + KP_COLUMNS]
    out = out.sort_values(['trade_date_idx', 'code_idx']).reset_index(drop=True)
    out.to_parquet(EVENT_FEAT_PARQUET, index=False)
    return out


def build_kpl_features(update: bool = False, force_rebuild: bool = False):
    """增量构建开盘啦结构特征并合并进 event_feat。无新数据直接返回已有（force_rebuild 全量重来）。"""
    events, added = build_kpl_events(update=update, force_rebuild=force_rebuild)
    if events is None:
        return None
    feat = pd.read_parquet(EVENT_FEAT_PARQUET) if os.path.exists(EVENT_FEAT_PARQUET) else None
    has_kp = feat is not None and all(c in feat.columns for c in KP_COLUMNS)
    if added == 0 and not force_rebuild and has_kp:
        print('[kpl] 无新数据')
        return feat
    K, edges = _compute_kp(events)
    edges.to_parquet(KPL_EDGES_PARQUET, index=False)
    merged = _merge_event_feat(K)
    print(f'[kpl] edges {len(edges)}行 → kpl_edges.parquet | event_feat {merged.shape} | ' +
          f'kp 非零率 ' + str({c: round(float((merged[c] > 0).mean()), 3) for c in KP_COLUMNS}))
    return merged


if __name__ == '__main__':
    build_kpl_features(update=True)
