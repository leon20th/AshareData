"""KPL 题材实体化（2026-09-25）：zscode 锚点清洗 → 实体/别名/成员/复核 四件套。

背景（全量审计结论）：
  · 801 段码稳定，但存在 ①一码多名（改名，122 干净/15 重叠）②一名多码（迁移，
    8 个纯 801 全部为时期不相交的迁移 + 34 个涉遗留码）③遗留段 885/881/880/882
    （2018-2019，退役）④新段 803/810（2025-09 起，活）⑤复合组（逗号拼接，1.3%）
    ⑥'0' 其他组（15%，非题材，排除）。

规则（确定性、全自动、可全量重建；冲突只进 review 队列、不阻塞）：
  1) 复合组按逗号对齐拆 (name, code) 对（名字数≠码数 → review）
  2) 图连通分量 = 实体：节点=(name,code)；连边 = 共享 name（迁移，无条件）/ 共享 code
     且名字相关（改名：互含/财务家族/共≥3字）；不相关的同码相邻期 = KPL 回收码
     给新题材 → 拆开 + review('code_reuse')。
     实体表：entity_id（稳定序号，按 首现日+最早码 排序）| anchor_code=最新码 |
     name_cur=最新名 | first/last | 码数/名数/行数
  3) 同名跨码（迁移）/ 同码改名 → 自动并入同一实体；时期重叠 → review 标记
  4) 复合拆对若 (name,code) 从未在单码行出现过 → review 标记（pair_new）
  5) 成员表全展开：一行 → 组内全部 (entity) 关系；primary = 单码行本主题 或
     复合行六级匹配（名字==细分 / 剥括号 / 括号内容 / 原因头"父(子)" / 细分映射表
     kpl_theme_segment_map.csv / 行级映射表 kpl_theme_row_map.csv，AI 离线生成）；
     '0' 组 entity_id=-1。

输出（built_data/）：
  kpl_theme_entities.parquet / kpl_theme_aliases.parquet /
  kpl_theme_members.parquet / kpl_theme_review.csv

用法：python -m AshareData.datautils.dataloaders.feature_build.kpl_entities
（全量重建秒级；新增数据后直接重跑即可，entity_id 在历史不变时逐位稳定）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from AshareData.paths import BUILT_DATA_DIR
from AshareData.datautils.dataloaders.feature_build.feature_utils import (
    get_code_idx,
    get_trade_date_idx,
)

EVENTS = f'{BUILT_DATA_DIR}/kpl_events.parquet'
OUT_E = f'{BUILT_DATA_DIR}/kpl_theme_entities.parquet'
OUT_A = f'{BUILT_DATA_DIR}/kpl_theme_aliases.parquet'
OUT_M = f'{BUILT_DATA_DIR}/kpl_theme_members.parquet'
OUT_R = f'{BUILT_DATA_DIR}/kpl_theme_review.csv'
# 人工覆写（可选）：列 [code, name_a, name_b, note]；同码下强制合并 name_a~name_b
MANUAL = f'{BUILT_DATA_DIR}/kpl_theme_manual.csv'
# 细分→父级 映射表（kpl_segment_map_ai 生成/可人工改）：列 [segment, parent, source, note]
SEG_MAP = f'{BUILT_DATA_DIR}/kpl_theme_segment_map.csv'
# 行级映射表（原因级回退，细分分不了时用）：列 [date, code, parent, source, note]
ROW_MAP = f'{BUILT_DATA_DIR}/kpl_theme_row_map.csv'


def _norm(s: str) -> str:
    return str(s).strip().replace('　', '').casefold()


import re as _re

_FIN = ('业绩', '预增', '预减', '扭亏', '年报', '半年报', '季报', '送转', '高送转', '增长')
_GENERIC = ('概念', '板块', '指数', '龙头', '产业链', '制造')


def _related(a: str, b: str) -> bool:
    """两个名字是否指向同一题材（改名守卫）：
    ① 互相包含（含剥括号后再试）；② 同为财务业绩家族；③ 共享≥2字非泛词子串。
    误并的保护方向：宁可拆开（保守），不给假连续性；拆错的进 review 或人工覆写。"""
    na, nb = _norm(a), _norm(b)
    ra = _re.sub(r'[（(].*?[)）]', '', na)
    rb = _re.sub(r'[（(].*?[)）]', '', nb)
    for x, y in ((na, nb), (ra, nb), (na, rb), (ra, rb)):
        if x and y and (x in y or y in x):
            return True
    if any(t in na for t in _FIN) and any(t in nb for t in _FIN):
        return True
    s, l = (na, nb) if len(na) <= len(nb) else (nb, na)
    for i in range(len(s) - 1):
        g2 = s[i:i + 2]
        if g2 not in _GENERIC and g2 in l:
            return True
    return False


_PARENT_RE = _re.compile(r'^([^；;+，,（()）]{1,12})[（(]([^）()]{1,15})[）)]')


def _strip_paren(s: str) -> str:
    return _re.sub(r'[（(][^（）()]*[）)]', '', str(s)).strip()


def _paren_content(s: str) -> str:
    m = _re.search(r'[（(]([^（）()]*)[）)]', str(s))
    return m.group(1).strip() if m else ''


def _load_segmap() -> dict:
    """细分映射表 → {细分(归一): 父级名(归一)}；空父级（AI 判定无归属）跳过。"""
    import os
    if not os.path.exists(SEG_MAP):
        return {}
    try:
        mf = pd.read_csv(SEG_MAP)
        return {_norm(r.segment): _norm(r.parent) for _, r in mf.iterrows()
                if str(r.parent).strip() and str(r.parent) != 'nan'}
    except Exception as e:
        print(f'[kpl-ent] 细分映射表读取失败: {e}')
        return {}


def _load_rowmap() -> dict:
    """行级映射表 → {(date, code6): 父级名(归一)}；空父级（AI 判定无归属）跳过。"""
    import os
    if not os.path.exists(ROW_MAP):
        return {}
    try:
        mf = pd.read_csv(ROW_MAP, dtype=str)
        return {(str(r.date).strip(), str(r.code).strip().zfill(6)): _norm(r.parent)
                for _, r in mf.iterrows()
                if str(r.parent).strip() and str(r.parent) != 'nan'}
    except Exception as e:
        print(f'[kpl-ent] 行映射表读取失败: {e}')
        return {}


def _pick_parent(ns: list, seg: str, reason: str, segmap: dict,
                 row_pick: str = '') -> int:
    """复合行成员 → 所属组内名字下标（-1 = 未命中）。
    规则依次：①名字==细分 ②剥括号相等 ③名字括号内容==细分 ④原因头"父(子)"子==细分
    ⑤细分映射表 ⑥行级映射表（原因级回退，AI 逐行判定；⑤未中才用⑥）。"""
    sn = _norm(seg)
    if not sn or sn == 'nan':
        return -1
    for i, n in enumerate(ns):
        if _norm(n) == sn:
            return i
    for i, n in enumerate(ns):
        if _norm(_strip_paren(n)) == sn or _norm(_paren_content(n)) == sn:
            return i
    m = _PARENT_RE.match(str(reason or ''))
    if m and _norm(m.group(2)) == sn:
        p = _norm(m.group(1))
        for i, n in enumerate(ns):
            if _norm(n) == p or _norm(_strip_paren(n)) == p:
                return i
    p = segmap.get(sn) or row_pick
    if p:
        for i, n in enumerate(ns):
            if _norm(n) == p or _norm(_strip_paren(n)) == p:
                return i
    return -1


class _DSU:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def build() -> None:
    df = pd.read_parquet(EVENTS)
    df['date'] = df.date.astype(str)
    df['zs_name'] = df.zs_name.astype(str)
    df['zscode'] = df.zscode.astype(str)
    review = []
    segmap = _load_segmap()
    rowmap = _load_rowmap()

    # ---- 1) 行 → 原子 (name, code) 对 ----
    recs = []          # (date, code, name, code_e, via, primary)
    miss = []          # 复合行未命中细分（归一化），供统计
    for d, c, zs, zc, seg, rt in zip(df.date, df.code.astype(str), df.zs_name,
                                     df.zscode, df.segment.astype(str),
                                     df.reason_txt.fillna('').astype(str)):
        if zc == '0':
            recs.append((d, c, None, None, 'other', False))
            continue
        if ',' in zc:
            ns = [x.strip() for x in zs.split(',')]
            cs = [x.strip() for x in zc.split(',')]
            if len(ns) != len(cs):
                review.append({'kind': 'badsplit', 'key': f'{d}|{c}', 'val': zc, 'note': zs})
                recs.append((d, c, None, None, 'badsplit', False))
                continue
            pi = _pick_parent(ns, seg, rt, segmap,
                              rowmap.get((d, str(c).zfill(6)), ''))
            if pi < 0 and _norm(seg) not in ('', 'nan'):
                miss.append(_norm(seg))
            for i, (n, cc) in enumerate(zip(ns, cs)):
                recs.append((d, c, n, cc, 'split', i == pi))
        else:
            # 单码行：本主题即 primary（segment 另存于 events，不参与归属）
            recs.append((d, c, zs.strip(), zc, 'single', True))
    P = pd.DataFrame(recs, columns=['date', 'code', 'name', 'code_e', 'via', 'primary'])

    # ---- 2) 节点时期统计（单码+拆解全部计入时间线）----
    atom = P[P.name.notna()].copy()
    st = (atom.groupby(['name', 'code_e'], as_index=False)
          .agg(n=('date', 'size'), first=('date', 'min'), last=('date', 'max')))
    nodes = list(zip(st.name, st.code_e))
    idx = {nd: i for i, nd in enumerate(nodes)}
    dsu = _DSU(len(nodes))

    # 同 name 的节点并（迁移，无条件）；同 code 的相邻相邻时期仅在“名字相关”时并
    # （改名）；不相关的同码期 = KPL 回收码给新题材 → 拆开 + review
    for name, g in st.groupby('name'):
        ids = [idx[(name, c)] for c in g.code_e]
        for a in ids[1:]:
            dsu.union(ids[0], a)
    for code, g in st.groupby('code_e'):
        if len(g) < 2:
            continue
        gs = list(g.sort_values('first').iterrows())
        for (_, r1), (_, r2) in zip(gs[:-1], gs[1:]):
            if _related(r1['name'], r2['name']):
                dsu.union(idx[(r1['name'], code)], idx[(r2['name'], code)])
            else:
                review.append({'kind': 'code_reuse', 'key': code,
                               'val': f"{r1['name']}({r1['first']}~{r1['last']})",
                               'note': f"→ {r2['name']}({r2['first']}~{r2['last']}) 不相关，已拆"})

    # 人工覆写（可选，存在才读）：强制同码名字合并（如 白酒↔酿酒）
    import os
    if os.path.exists(MANUAL):
        mf = pd.read_csv(MANUAL)
        for _, r in mf.iterrows():
            ka, kb = (str(r.name_a), str(r.code)), (str(r.name_b), str(r.code))
            if ka in idx and kb in idx:
                dsu.union(idx[ka], idx[kb])
                print(f"[kpl-ent] 人工覆写合并: {r.code} {r.name_a} ↔ {r.name_b}")
            else:
                review.append({'kind': 'manual_miss', 'key': f"{r.code}|{r.name_a}|{r.name_b}",
                               'val': '', 'note': '人工覆写目标不在节点表'})

    # 重叠告警（并行同名 / 同码异名重叠）
    for name, g in st.groupby('name'):
        if len(g) > 1:
            gs = g.sort_values('first')
            for (_, r1), (_, r2) in zip(list(gs.iterrows())[:-1], list(gs.iterrows())[1:]):
                if r2['first'] <= r1['last']:
                    review.append({'kind': 'name_overlap', 'key': name,
                                   'val': f"{r1['code_e']}({r1['first']}~{r1['last']})",
                                   'note': f"vs {r2['code_e']}({r2['first']}~{r2['last']})"})
    for code, g in st.groupby('code_e'):
        if len(g) > 1:
            gs = g.sort_values('first')
            for (_, r1), (_, r2) in zip(list(gs.iterrows())[:-1], list(gs.iterrows())[1:]):
                if r2['first'] <= r1['last']:
                    review.append({'kind': 'code_overlap', 'key': code,
                                   'val': f"{r1['name']}({r1['first']}~{r1['last']})",
                                   'note': f"vs {r2['name']}({r2['first']}~{r2['last']})"})

    # 复合拆对未见对 → review
    singles_pairs = set(zip(atom[atom.via == 'single'].name, atom[atom.via == 'single'].code_e))
    split_pairs = atom[atom.via == 'split'][['name', 'code_e']].drop_duplicates()
    for n, c in zip(split_pairs.name, split_pairs.code_e):
        if (n, c) not in singles_pairs:
            review.append({'kind': 'pair_unseen', 'key': f'{n}|{c}', 'val': '',
                           'note': '复合组拆出的 (名,码) 未在单码行出现过'})

    # ---- 3) 实体表 ----
    comp = {}
    for i, nd in enumerate(nodes):
        comp.setdefault(dsu.find(i), []).append(i)
    ents = []
    for root, ids in comp.items():
        g = st.iloc[ids]
        first, last = g['first'].min(), g['last'].max()
        anchor = g.sort_values(['last', 'n'], ascending=False).iloc[0]
        ents.append({'first': first, 'last': last, 'anchor_code': anchor.code_e,
                     'name_cur': anchor['name'], 'n_codes': g.code_e.nunique(),
                     'n_names': g['name'].nunique(), 'n_rows': int(g.n.sum()),
                     'root': root})
    ents = pd.DataFrame(ents).sort_values(['first', 'anchor_code']).reset_index(drop=True)
    ents.insert(0, 'entity_id', ents.index + 1)
    node2e = {}
    for root, ids in comp.items():
        eid = int(ents[ents.root == root].entity_id.iat[0])
        for i in ids:
            node2e[nodes[i]] = eid
    ents = ents.drop(columns=['root'])
    ents.to_parquet(OUT_E, index=False)

    # ---- 别名表 ----
    al = st.copy()
    al['entity_id'] = [node2e[(n, c)] for n, c in zip(al.name, al.code_e)]
    al = al.merge(ents[['entity_id', 'anchor_code', 'name_cur']], on='entity_id')
    al['is_anchor'] = al.code_e == al.anchor_code
    al['is_name_cur'] = al['name'] == al.name_cur
    al = al.rename(columns={'name': 'zs_name', 'code_e': 'zscode', 'n': 'n_rows'})
    al.sort_values(['entity_id', 'first']).to_parquet(OUT_A, index=False)

    # ---- 4) 成员表（全展开 + primary + tdi/cidx 映射）----
    M = P.copy()
    M['entity_id'] = [node2e.get((n, c), -1) if n is not None else -1
                      for n, c in zip(M.name, M.code_e)]
    dates = sorted(df.date.unique())
    d2i = dict(zip(dates, get_trade_date_idx(list(dates))))
    codes_u = list(df.code.astype(str).unique())
    c2i = dict(zip(codes_u, get_code_idx(codes_u)))
    M['date'] = M.date.astype(str)
    M['trade_date_idx'] = M.date.map(d2i).fillna(-1).astype('int64')
    M['code_idx'] = M.code.map(c2i).fillna(-1).astype('int64')
    M = M[['date', 'code', 'code_idx', 'trade_date_idx', 'entity_id', 'primary', 'via']]
    M.to_parquet(OUT_M, index=False)

    rev = pd.DataFrame(review).drop_duplicates() if review else pd.DataFrame(
        columns=['kind', 'key', 'val', 'note'])
    rev.to_csv(OUT_R, index=False)

    # ---- 5) 统计 + 回归案例（301335 宠物经济 轮次修复）----
    print(f"[kpl-ent] 实体 {len(ents)} 个 | 行 {len(M)} | 单码 {(P.via=='single').sum()} "
          f"| 拆解 {(P.via=='split').sum()} | other {(P.via=='other').sum()}")
    print(f"[kpl-ent] 多码实体(迁移链) {(ents.n_codes>1).sum()} | 多名码(改名) "
          f"{(ents.n_names>1).sum()} | review 行: {len(rev)}")
    sp = P[P.via == 'split']
    n_ev = sp.drop_duplicates(['date', 'code']).shape[0]
    hit_ev = sp[sp.primary].drop_duplicates(['date', 'code']).shape[0]
    print(f"[kpl-ent] 复合事件 {n_ev} 例 | 归属命中 {hit_ev} ({hit_ev/max(1,n_ev):.1%})"
          f" | 未命中细分 {len(set(miss))} 种（可跑 kpl_segment_map_ai 补映射）")
    if len(rev):
        print(rev.kind.value_counts().to_string())

    # 回归：301335 与 宠物经济(801243) 实体，重算轮龄
    eid = int(al[(al.zscode == '801243')].entity_id.iat[0])
    mm = M[(M.entity_id == eid)]
    days = sorted(mm.date.unique())
    di = {d: i for i, d in enumerate(days)}
    age, prev = {}, None
    for d in days:
        age[d] = age[prev] + 1 if (prev is not None and di[d] - di[prev] <= 3) else 0
        prev = d
    rows301 = M[(M.code == '301335') & (M.entity_id == eid)].drop_duplicates('date')
    print(f"[回归] 301335 × 宠物经济(实体{eid}, 锚{al[al.entity_id==eid].anchor_code.iat[0]}) "
          f"轮龄（修后）:")
    for d in sorted(rows301.date):
        print(f"    {d}: age={age[d]}  (旧 kp_age 口径见备注)")


if __name__ == '__main__':
    build()
