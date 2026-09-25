"""KPL 细分/行级 归属映射 AI 生成器（2026-09-25，两阶段）。

背景：复合组（如"文化传媒,酿酒,零售,食品饮料,旅游,免税,NMN"）拆解时，部分成员的
细分与组内名字都匹配不上（如 代糖概念、数据中心、卫星导航）——kpl_entities 的
四级确定性规则（名字/剥括号/括号内容/原因头"父(子)"）无法归属。本脚本分两阶段：
  阶段1（细分级）：残余细分 → DeepSeek 判定所属板块 → kpl_theme_segment_map.csv
  阶段2（行级回退）：阶段1仍分不了的行 → 按"该行原因文本"逐条判定（如第二业务
      曲线与候选板块吻合则归其）→ kpl_theme_row_map.csv（列 [date, code, parent,
      source, note]）
两表均可人工修改；kpl_entities.build() 将其作为第五/六级匹配（只读表、确定性）。
本脚本已接入每日构建链（build_all_feature：kpl_entities 之后自动增量；无新数据秒退；
API 不可用自动跳过、不阻断构建，未覆盖项下次运行重试）——日常无需手动运行，仅在
表需人工修正或补跑时直接执行本模块即可（只问新细分/新行；判定无归属的记空父级防
反复询问，要重问删掉该行即可）。

用法：python -m AshareData.datautils.dataloaders.feature_build.kpl_segment_map_ai
"""
import json
import os
from collections import Counter

import pandas as pd
from openai import OpenAI

from AshareData.datautils.dataloaders.feature_build.kpl_entities import (
    EVENTS, SEG_MAP, ROW_MAP, _norm, _pick_parent, _load_segmap, _load_rowmap)

BATCH = 20
MODEL = 'deepseek-flash'   # API 正式名（deepseek-v4-flash 为兼容别名；v4.1-flash 不存在）
SYSTEM = ('你是A股题材分类助手。输入是一组条目：每条有"细分"（一个细分题材）、"候选板块"列表，'
          '部分条目附"原因片段"（个别样本的涨停原因文本，仅供参考）。判断每个细分题材最合适归属的'
          '哪个候选板块。要求：板块必须从该条的候选里原样选一个；若候选中确无合适归属则填 null。'
          '只输出 JSON 对象（键=细分原文，值=板块名或null），不要解释、不要代码块。')
SYSTEM2 = ('你是A股题材分类助手。输入条目：一只股票某日涨停的"细分"标签、"候选板块"列表和'
           '"原因片段"（当日涨停原因文本，可能含多条并列催化线）。判断该股被归入该行时对应'
           '哪个候选板块：①细分标签合适则选它；②细分标签不在候选中，则看原因片段中与候选'
           '板块吻合的其他催化线（如第二业务曲线），选对应板块；③原因与所有候选均无关联则'
           '填 null。板块必须从该条候选中原样选一个。只输出 JSON 对象（键=条目 id 原文，'
           '值=板块名或null），不要解释、不要代码块。')


def _have() -> set:
    """映射表中已处理过的细分（含空父级），归一化。"""
    if not os.path.exists(SEG_MAP):
        return set()
    return {_norm(s) for s in pd.read_csv(SEG_MAP).segment.astype(str)}


def _pending() -> dict:
    """复合行残余细分 → {'p': 候选父级计数器, 'r': 原因样本≤2}；确定性规则已命中的跳过。"""
    df = pd.read_parquet(EVENTS)
    have, out = _have(), {}
    for d, c, zs, zc, seg, rt in zip(df.date.astype(str), df.code.astype(str),
                                     df.zs_name.astype(str), df.zscode.astype(str),
                                     df.segment.fillna('').astype(str),
                                     df.reason_txt.fillna('').astype(str)):
        ns = [x.strip() for x in zs.split(',')]
        cs = [x.strip() for x in zc.split(',')]
        if (',' not in zc or len(ns) != len(cs) or not seg.strip()
                or _norm(seg) in have or _pick_parent(ns, seg, rt, {}) >= 0):
            continue
        rec = out.setdefault(seg, {'p': Counter(), 'r': [], 'n': 0})
        rec['n'] += 1
        for n in ns:
            rec['p'][n] += 1
        t = str(rt).strip()
        if t and len(rec['r']) < 2 and t[:26] not in [x[:26] for x in rec['r']]:
            rec['r'].append(t[:220])
    return out


def _parse(txt: str) -> dict:
    """容错解析响应：支持 {细分: 板块} 或 [{细分, 板块}, ...] 两种形态。"""
    out = {}
    for a, b in (('[', ']'), ('{', '}')):
        i, j = txt.find(a), txt.rfind(b)
        if i < 0 or j <= i:
            continue
        try:
            obj = json.loads(txt[i:j + 1])
        except Exception:
            continue
        if isinstance(obj, dict):
            out.update(obj)
        elif isinstance(obj, list):
            for it in obj:
                if not isinstance(it, dict):
                    continue
                key = it.get('id', it.get('细分'))
                if key is None:
                    continue
                v = next((x for k, x in it.items()
                          if k not in ('id', '细分') and isinstance(x, str)), None)
                out[str(key).strip()] = v
        if out:
            return out
    return out


def _ask(client, items: list) -> dict:
    payload = [{'细分': s, '候选板块': [n for n, _ in r['p'].most_common()],
                '原因片段': r['r']} for s, r in items]
    r = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=4000,
        extra_body={'thinking': {'type': 'disabled'}},   # 分类任务无需推理，防预算被吃掉
        messages=[{'role': 'system', 'content': SYSTEM},
                  {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}])
    return _parse((r.choices[0].message.content or '').strip())


def _have_rows() -> set:
    """行级映射表中已处理过的 (date, code6)，防反复询问。"""
    if not os.path.exists(ROW_MAP):
        return set()
    mf = pd.read_csv(ROW_MAP, dtype=str)
    return {(str(r.date).strip(), str(r.code).strip().zfill(6))
            for _, r in mf.iterrows()}


def _pending_rows(segmap: dict, rowmap: dict) -> list:
    """所有规则+两表后仍无归属的复合行 → 阶段2 任务列表。"""
    df = pd.read_parquet(EVENTS)
    have, out = _have_rows(), []
    for d, c, zs, zc, seg, rt in zip(df.date.astype(str), df.code.astype(str),
                                     df.zs_name.astype(str), df.zscode.astype(str),
                                     df.segment.fillna('').astype(str),
                                     df.reason_txt.fillna('').astype(str)):
        ns = [x.strip() for x in zs.split(',')]
        cs = [x.strip() for x in zc.split(',')]
        key = (str(d), str(c).zfill(6))
        if (',' not in zc or len(ns) != len(cs) or key in have
                or _pick_parent(ns, seg, rt, segmap, rowmap.get(key, '')) >= 0):
            continue
        out.append({'id': f'{key[0]}|{key[1]}', 'date': key[0], 'code': key[1],
                    'seg': seg, 'ns': ns, 'rt': str(rt)[:380]})
    return out


def _ask_rows(client, items: list) -> dict:
    payload = [{'id': it['id'], '细分': it['seg'], '候选板块': it['ns'],
                '原因片段': it['rt']} for it in items]
    r = client.chat.completions.create(
        model=MODEL, temperature=0, max_tokens=4000,
        extra_body={'thinking': {'type': 'disabled'}},
        messages=[{'role': 'system', 'content': SYSTEM2},
                  {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False)}])
    return _parse((r.choices[0].message.content or '').strip())


def update() -> int:
    """两阶段增量更新映射表；返回新写入行数（0 = 无变化）。

    接入每日构建链（build_all_feature.py）；API/网络不可用时仅打印并返回 0，
    不阻断构建——未覆盖的细分/行会在下次运行自动重试。"""
    try:
        from env_setting import DEEPSEEK_API_KEY
        client = OpenAI(api_key=DEEPSEEK_API_KEY['regime'],
                        base_url='https://api.deepseek.com', timeout=120)
    except Exception as e:
        print(f'[kpl-segmap] 客户端不可用，跳过: {e}')
        return 0
    n = 0

    # ---- 阶段 1：细分级映射（细分 → 候选板块之一）----
    pending = _pending()
    if not pending:
        print('[kpl-segmap] 阶段1：无待判定细分')
    else:
        print(f'[kpl-segmap] 阶段1：待判定细分 {len(pending)} 个')
        items = sorted(pending.items(), key=lambda kv: -sum(kv[1]['p'].values()))
        rows = []
        for i in range(0, len(items), BATCH):
            batch = items[i:i + BATCH]
            try:
                got = {str(k).strip(): v for k, v in _ask(client, batch).items()}
            except Exception as e:
                print(f'[kpl-segmap] 阶段1 批次 {i // BATCH + 1} 失败: {e}')
                continue
            for s, cc in batch:
                if s not in got:                      # 未解析：不写，下次重问
                    continue
                v = str(got[s]).strip() if got[s] is not None else ''
                if v and v not in cc['p']:            # 必须选自候选，否则记 null
                    print(f'[kpl-segmap] 阶段1 越界候选忽略: {s} → {v!r}')
                    v = ''
                rows.append({'segment': s, 'parent': v, 'source': 'ai',
                             'note': f"rows={cc['n']}"})
            miss = [s for s, _ in batch if s not in got]
            print(f'[kpl-segmap] 阶段1 批次 {i // BATCH + 1}: {len(batch)} 个 → 判定 '
                  f'{len(batch) - len(miss)}' + (f'，未解析 {miss}' if miss else ''))
        if rows:
            new = pd.DataFrame(rows)
            old = pd.read_csv(SEG_MAP) if os.path.exists(SEG_MAP) else None
            out = (pd.concat([old, new], ignore_index=True) if old is not None else new)
            out = out.drop_duplicates('segment', keep='last').sort_values('segment')
            out.to_csv(SEG_MAP, index=False)
            n += len(new)
            print(f'[kpl-segmap] 阶段1 写入 {SEG_MAP}: +{len(new)} 行（共 {len(out)}）')

    # ---- 阶段 2：行级回退（按原因文本逐行判定归属）----
    prows = _pending_rows(_load_segmap(), _load_rowmap())
    if not prows:
        print('[kpl-segmap] 阶段2：无待判定行')
    else:
        print(f'[kpl-segmap] 阶段2：待判定行 {len(prows)} 条')
        prows.sort(key=lambda x: x['date'])
        rrows = []
        for i in range(0, len(prows), BATCH):
            batch = prows[i:i + BATCH]
            try:
                got = {str(k).strip(): v for k, v in _ask_rows(client, batch).items()}
            except Exception as e:
                print(f'[kpl-segmap] 阶段2 批次 {i // BATCH + 1} 失败: {e}')
                continue
            for it in batch:
                if it['id'] not in got:                   # 未解析：不写，下次重问
                    continue
                v = str(got[it['id']]).strip() if got[it['id']] is not None else ''
                if v and v not in it['ns']:               # 必须选自候选，否则记 null
                    print(f"[kpl-segmap] 阶段2 越界候选忽略: {it['id']} → {v!r}")
                    v = ''
                rrows.append({'date': it['date'], 'code': it['code'], 'parent': v,
                              'source': 'ai', 'note': f"细分={it['seg']}"})
            miss = [it['id'] for it in batch if it['id'] not in got]
            print(f'[kpl-segmap] 阶段2 批次 {i // BATCH + 1}: {len(batch)} 条 → 判定 '
                  f'{len(batch) - len(miss)}' + (f'，未解析 {miss}' if miss else ''))
        if rrows:
            new = pd.DataFrame(rrows)
            old = pd.read_csv(ROW_MAP, dtype=str) if os.path.exists(ROW_MAP) else None
            out = (pd.concat([old, new], ignore_index=True) if old is not None else new)
            out = out.drop_duplicates(['date', 'code'], keep='last').sort_values(['date', 'code'])
            out.to_csv(ROW_MAP, index=False)
            n += len(new)
            print(f'[kpl-segmap] 阶段2 写入 {ROW_MAP}: +{len(new)} 行（共 {len(out)}）')
            for r in rrows:
                print(f"    {r['date']} {r['code']} → {r['parent'] or '(null)'}")
    return n


def main() -> None:
    update()


if __name__ == '__main__':
    main()
