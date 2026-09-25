# -*- coding: utf-8 -*-
# scrap_news_cls —— 财联社「电报」快讯抓取（v1/roll 接口，sha1·md5 签名，免登录）
# 数据范围: 2015 年起；条目带 subjects（话题）与稀疏 stock_list（个股）
# 产物: scrap_data/news_cls/YYYY-MM-DD.json（裁剪版条目，按 id 去重）+ _state.json（断点）
# 机制: last_time=秒级时间戳链式回溯，可直跳任意历史时刻；首跑回填到 --start，之后跑增量
# 注意: 节流保守（默认 0.5s+抖动）；连续失败自动退避重试；中断后重跑即续传
# 用法:
#   python AshareData/datautils/regime_scripts/scrap_news_cls.py --start 2020-01-01
#   python AshareData/datautils/regime_scripts/scrap_news_cls.py --max-req 3   # 冒烟
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.parse
from datetime import datetime, timedelta, timezone

import requests

from AshareData.paths import SCRAP_DATA_DIR
from AshareData.utils.exchanges_utils.a_open import get_target_trade_date
from AshareData.utils.log_util import get_logger

logger = get_logger('财联社抓取')

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36')
API = 'https://www.cls.cn/v1/roll/get_roll_list'
OUT_DIR = f'{SCRAP_DATA_DIR}/news_cls'
STATE_F = f'{OUT_DIR}/_state.json'
RN = 50
TZ = timezone(timedelta(hours=8))


def fetch_page(last_time):
    """取一页（last_time 空=从最新开始，否则取该时刻之前）；返回条目list（新→旧）。"""
    par = {'app': 'CailianpressWeb', 'category': '', 'last_time': str(last_time or int(time.time())),
           'os': 'web', 'refresh_type': '1', 'rn': str(RN), 'sv': '8.4.6'}
    par['sign'] = hashlib.md5(hashlib.sha1(urllib.parse.urlencode(par).encode()).hexdigest().encode()).hexdigest()
    r = requests.get(API, params=par, timeout=25,
                     headers={'User-Agent': UA, 'Referer': 'https://www.cls.cn/telegraph'})
    j = r.json()
    if r.status_code != 200 or j.get('errno') != 0:
        raise RuntimeError(f'HTTP {r.status_code} errno={j.get("errno")} msg={j.get("msg")}')
    return (j.get('data') or {}).get('roll_data') or []


def slim(it):
    """裁剪：保留文本/时间/标签字段，去掉图片/分享/评论等大字段。"""
    o = {k: it.get(k) for k in ('id', 'ctime', 'level', 'title', 'brief', 'content', 'category')}
    o['stock_list'] = [{'StockID': s.get('StockID'), 'name': s.get('name')} for s in it.get('stock_list') or []]
    o['subjects'] = [s.get('subject_name') for s in it.get('subjects') or []]
    o['plate_list'] = it.get('plate_list') or []
    return o


def save_day(day, items):
    """写入/合并单日文件（按 id 去重）；返回新增条数。"""
    f = f'{OUT_DIR}/{day}.json'
    old = json.load(open(f, encoding='utf-8')) if os.path.exists(f) else []
    merged = {it['id']: it for it in old}
    n_new = sum(1 for it in items if it['id'] not in merged)
    for it in items:
        merged.setdefault(it['id'], it)
    json.dump(list(merged.values()), open(f, 'w', encoding='utf-8'), ensure_ascii=False)
    return n_new


def rebuild_state_from_files(start):
    """_state.json 丢失兜底：按日文件覆盖重建游标（避免整段重爬）。返回 None=无可重建。"""
    days = sorted(f[:-5] for f in os.listdir(OUT_DIR) if re.fullmatch(r'\d{4}-\d{2}-\d{2}\.json', f))
    if not days:
        return None

    def edge(day, last):
        vals = [int(it['ctime']) for it in json.load(open(f'{OUT_DIR}/{day}.json', encoding='utf-8')) if it.get('ctime')]
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
    return {'newest_ts': newest, 'oldest_ts': oldest - 1, 'done': covered, 'floor': days[0] if covered else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2020-01-01', help='回填下界（仅回填/续传用）')
    ap.add_argument('--max-req', type=int, default=0, help='最多请求数（冒烟用，0=不限）')
    ap.add_argument('--sleep', type=float, default=0.5, help='请求间隔秒（另加 0~50%% 抖动）')
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    st = json.load(open(STATE_F, encoding='utf-8')) if os.path.exists(STATE_F) else rebuild_state_from_files(args.start)
    if st is None:
        st = {'newest_ts': 0, 'oldest_ts': None, 'done': False, 'floor': None}
    start_ts = int(datetime.fromisoformat(args.start).replace(tzinfo=TZ).timestamp())
    if st['done'] and args.start < (st['floor'] or '9999-99-99'):
        st['done'] = False
        logger.info(f"重开回填: floor {st['floor']} → {args.start}")
    incremental = st['done']
    cursor = None if incremental else st['oldest_ts']
    stop_ts = st['newest_ts'] if incremental else start_ts
    req = n_new = 0
    while True:
        if args.max_req and req >= args.max_req:
            logger.info(f'达到 --max-req={args.max_req}，正常退出（状态已存，重跑续传）')
            break
        items = None
        for attempt in range(6):
            try:
                items = fetch_page(cursor)
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
            ts = int(it['ctime'])
            if (incremental and ts <= stop_ts) or ((not incremental) and ts < start_ts):
                hit_end = True
                break
            buckets.setdefault(datetime.fromtimestamp(ts, TZ).strftime('%Y-%m-%d'), []).append(slim(it))
        for day, its in buckets.items():
            n_new += save_day(day, its)
        st['newest_ts'] = max(st['newest_ts'], max(int(it['ctime']) for it in items))
        if not incremental:
            cursor = min(int(it['ctime']) for it in items) - 1
            st['oldest_ts'] = cursor
        if hit_end:
            st['done'] = True
            st['floor'] = args.start
            logger.info(f'到达下界 → 完成（floor={args.start}）')
            break
        json.dump(st, open(STATE_F, 'w', encoding='utf-8'))
        if req % 20 == 0:
            reached = datetime.fromtimestamp(cursor or 0, TZ).strftime('%Y-%m-%d %H:%M')
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
