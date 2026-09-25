# -*- coding: utf-8 -*-
# scrap_news_em724 —— 东财 7x24 全球财经快讯抓取（np-weblist 接口，免登录、无签名）
# 数据范围: 2016 年起；条目自带 stockList（个股 0./1.xxxxxx 与板块 90.BKxxxx）
# 产物: scrap_data/news_em724/YYYY-MM-DD.json（按 code 去重）+ _state.json（断点）
# 机制: sortEnd=微秒时间戳链式回溯，可直跳任意历史时刻；首跑回填到 --start，之后跑增量
# 注意: 节流保守（默认 0.5s+抖动）；连续失败自动退避重试；中断后重跑即续传
# 用法:
#   python AshareData/datautils/regime_scripts/scrap_news_em724.py --start 2020-01-01
#   python AshareData/datautils/regime_scripts/scrap_news_em724.py --max-req 3   # 冒烟
import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta, timezone

import requests

from AshareData.paths import SCRAP_DATA_DIR
from AshareData.utils.exchanges_utils.a_open import get_target_trade_date
from AshareData.utils.log_util import get_logger

logger = get_logger('东财7x24抓取')

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36')
API = 'https://np-weblist.eastmoney.com/comm/web/getFastNewsList'
OUT_DIR = f'{SCRAP_DATA_DIR}/news_em724'
STATE_F = f'{OUT_DIR}/_state.json'
PAGE = 200
TZ = timezone(timedelta(hours=8))


def fetch_page(sort_end):
    """取一页（sort_end 空=从最新开始）；返回 (条目list新→旧, 下一页sortEnd µs)。"""
    par = {'client': 'web', 'biz': 'web_724', 'fastColumn': '102', 'sortEnd': str(sort_end or ''),
           'pageSize': str(PAGE), 'req_trace': str(int(time.time() * 1000))}
    r = requests.get(API, params=par, timeout=25,
                     headers={'User-Agent': UA, 'Referer': 'https://kuaixun.eastmoney.com/'})
    j = r.json()
    if r.status_code != 200 or str(j.get('code')) != '1':
        raise RuntimeError(f'HTTP {r.status_code} code={j.get("code")} msg={j.get("message")}')
    d = j['data']
    return d.get('fastNewsList') or [], int(d['sortEnd'])


def item_ts(it):
    return int(datetime.strptime(it['showTime'], '%Y-%m-%d %H:%M:%S').replace(tzinfo=TZ).timestamp())


def save_day(day, items):
    """写入/合并单日文件（按 code 去重）；返回新增条数。"""
    f = f'{OUT_DIR}/{day}.json'
    old = json.load(open(f, encoding='utf-8')) if os.path.exists(f) else []
    merged = {it['code']: it for it in old}
    n_new = sum(1 for it in items if it['code'] not in merged)
    for it in items:
        merged.setdefault(it['code'], it)
    json.dump(list(merged.values()), open(f, 'w', encoding='utf-8'), ensure_ascii=False)
    return n_new


def rebuild_state_from_files(start):
    """_state.json 丢失兜底：按日文件覆盖重建游标（避免整段重爬）。返回 None=无可重建。"""
    days = sorted(f[:-5] for f in os.listdir(OUT_DIR) if re.fullmatch(r'\d{4}-\d{2}-\d{2}\.json', f))
    if not days:
        return None

    def edge(day, last):
        vals = [item_ts(it) for it in json.load(open(f'{OUT_DIR}/{day}.json', encoding='utf-8')) if it.get('showTime')]
        return (max(vals) if last else min(vals)) if vals else None

    newest, oldest = edge(days[-1], True), edge(days[0], False)
    if newest is None or oldest is None:
        return None
    d0, d1 = datetime.fromisoformat(days[0]), datetime.fromisoformat(days[-1])
    holes = sorted({(d0 + timedelta(days=i)).strftime('%Y-%m-%d') for i in range((d1 - d0).days + 1)} - set(days))
    if holes:
        logger.warning(f'本地文件存在 {len(holes)} 天空洞: {holes[:8]}{" ..." if len(holes) > 8 else ""}')
    covered = days[0] <= start
    logger.info(f'无 _state.json → 从本地文件重建: 覆盖 {days[0]}~{days[-1]}，{"已达下界" if covered else "未达下界，续传回填"}')
    return {'newest_us': newest * 1_000_000, 'oldest_us': (oldest - 1) * 1_000_000, 'done': covered, 'floor': days[0] if covered else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2020-01-01', help='回填下界（仅回填/续传用）')
    ap.add_argument('--max-req', type=int, default=0, help='最多请求数（冒烟用，0=不限）')
    ap.add_argument('--sleep', type=float, default=0.5, help='请求间隔秒（另加 0~50%% 抖动）')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    st = json.load(open(STATE_F, encoding='utf-8')) if os.path.exists(STATE_F) else rebuild_state_from_files(args.start)
    if st is None:
        st = {'newest_us': 0, 'oldest_us': None, 'done': False, 'floor': None}
    start_ts = int(datetime.fromisoformat(args.start).replace(tzinfo=TZ).timestamp())
    if st['done'] and args.start < (st['floor'] or '9999-99-99'):
        st['done'] = False
        logger.info(f"重开回填: floor {st['floor']} → {args.start}")
    incremental = st['done']
    cursor = None if incremental else st['oldest_us']
    stop_ts = st['newest_us'] // 1_000_000 if incremental else start_ts
    req = n_new = 0
    while True:
        if args.max_req and req >= args.max_req:
            logger.info(f'达到 --max-req={args.max_req}，正常退出（状态已存，重跑续传）')
            break
        items = next_us = None
        for attempt in range(6):
            try:
                items, next_us = fetch_page(cursor)
                break
            except Exception as e:
                wait = [10, 30, 60, 120, 300][min(attempt, 4)]
                logger.error(f'请求失败({attempt + 1}/6): {e} → {wait}s 后重试')
                time.sleep(wait)
        if items is None:
            json.dump(st, open(STATE_F, 'w', encoding='utf-8'))
            logger.error('连续失败，中止（断点已存，可直接重跑续传）')
            sys.exit(2)
        req += 1
        if not items:
            st['done'] = True
            st['floor'] = args.start
            logger.info(f'空页 → 完成（floor={args.start}）')
            break
        hit_end, buckets = False, {}
        for it in items:  # 新→旧
            ts = item_ts(it)
            if (incremental and ts <= stop_ts) or ((not incremental) and ts < start_ts):
                hit_end = True
                break
            buckets.setdefault(it['showTime'][:10], []).append(
                {k: it.get(k) for k in ('code', 'title', 'summary', 'showTime', 'stockList', 'realSort')})
        for day, its in buckets.items():
            n_new += save_day(day, its)
        st['newest_us'] = max(st['newest_us'], max(item_ts(it) for it in items) * 1_000_000)
        if not incremental:
            if cursor is not None and next_us >= cursor:
                logger.warning(f'sortEnd 未推进({next_us})，停止防死循环')
                st['done'] = True
                st['floor'] = args.start
                break
            cursor = next_us
            st['oldest_us'] = cursor
        if hit_end:
            st['done'] = True
            st['floor'] = args.start
            logger.info(f'到达下界 → 完成（floor={args.start}）')
            break
        json.dump(st, open(STATE_F, 'w', encoding='utf-8'))
        if req % 20 == 0:
            reached = datetime.fromtimestamp((next_us or 0) / 1_000_000, TZ).strftime('%Y-%m-%d')
            logger.info(f'req={req} 进度至 {reached} 累计新增={n_new}')
        time.sleep(args.sleep + random.uniform(0, args.sleep * 0.5))
    json.dump(st, open(STATE_F, 'w', encoding='utf-8'))
    logger.info(f'完成: req={req} 新增条目={n_new} done={st["done"]} → {OUT_DIR}')


def is_latest():
    """本地数据是否已覆盖到最近交易日（get_target_trade_date）。无参。"""
    target = get_target_trade_date()
    if not target or not os.path.isdir(OUT_DIR):
        return False
    days = [f[:-5] for f in os.listdir(OUT_DIR) if re.fullmatch(r'\d{4}-\d{2}-\d{2}\.json', f)]
    return bool(days) and max(days) >= f'{target[:4]}-{target[4:6]}-{target[6:8]}'


if __name__ == '__main__':
    main()
