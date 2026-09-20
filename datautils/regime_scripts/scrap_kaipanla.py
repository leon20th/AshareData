# scrap_kaipanla —— 开盘啦「涨停原因题材榜」抓取（历史接口 apphis.longhuvip.com，免登录、无配额）
# 数据范围: 2018-04-03 起（更早只有汇总无题材明细）
# 产物: scrap_data/kaipanla/YYYY-MM-DD.json（原始返回 + _actual_date/_fallback 标记；空数据不落盘）
# 用法:
#   python AshareData/datautils/regime_scripts/scrap_kaipanla.py --update               # 增量补齐到最近交易日
#   python AshareData/datautils/regime_scripts/scrap_kaipanla.py --date 2020-06-30
#   python AshareData/datautils/regime_scripts/scrap_kaipanla.py --start 2018-04-01 --end 2026-09-17
#   加 --force 可重抓已有文件；--sleep 控制请求间隔（默认 2.5s）
import argparse
import json
import os
import random
import sys
import time
import uuid
from datetime import date as _date
from datetime import timedelta

import requests

from AshareData.paths import SCRAP_DATA_DIR
from AshareData.utils.exchanges_utils import a_open
from AshareData.utils.log_util import get_logger

logger = get_logger('开盘啦抓取')

UA = 'Dalvik/2.1.0 (Linux; U; Android 12; ALN-AL00 Build/W528JS)'  # 非 Dalvik UA 会被拒绝
API = 'https://apphis.longhuvip.com/w1/api/index.php'
KPL_BEGIN = '2018-04-03'  # 首个有题材明细的数据日（更早只回汇总，空数据不会落盘）
KPL_DIR = f'{SCRAP_DATA_DIR}/kaipanla'


def fetch_day(d):
    """抓取单日历史题材榜；返回 dict（含 list/nums/date）。"""
    body = {'a': 'GetPlateInfo_w38', 'st': '100', 'c': 'HisLimitResumption',
            'PhoneOSNew': '1', 'DeviceID': str(uuid.uuid4()), 'VerSion': '6.2.20.2',
            'Index': '0', 'Date': d, 'apiv': 'w47', 'Red': '0'}
    r = requests.post(API, data=body, timeout=25, headers={
        'User-Agent': UA, 'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'})
    if r.status_code != 200:
        raise RuntimeError(f'HTTP {r.status_code}')
    j = r.json()
    if not isinstance(j, dict) or 'list' not in j:
        raise RuntimeError(f'异常响应: {str(j)[:150]}')
    return j


def saved_dates():
    return {f[:10] for f in os.listdir(KPL_DIR)} if os.path.isdir(KPL_DIR) else set()


def resolve_dates(args):
    """显式单日/区间，或默认增量：日历内 [KPL_BEGIN, 最近交易日] 减去已落盘日。"""
    cal = set(a_open.get_trade_date_list())
    if args.date:
        return [args.date]
    if args.start:
        end = args.end or _date.today().isoformat()
        days = []
        d = _date.fromisoformat(args.start)
        d1 = _date.fromisoformat(end)
        while d <= d1:
            days.append(d.isoformat())
            d += timedelta(days=1)
        return [x for x in days if x.replace('-', '') in cal]
    target = a_open.get_target_trade_date()  # 收盘后=今天，盘中=上一交易日
    target = f'{target[:4]}-{target[4:6]}-{target[6:]}'
    have = saved_dates()
    days = []
    d = _date.fromisoformat(KPL_BEGIN)
    d1 = _date.fromisoformat(target)
    while d <= d1:
        s = d.isoformat()
        if s.replace('-', '') in cal and s not in have:
            days.append(s)
        d += timedelta(days=1)
    return days


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', help='单日 YYYY-MM-DD')
    ap.add_argument('--start', help='区间起始（缺省结束=今天，走日历过滤）')
    ap.add_argument('--end', help='区间结束')
    ap.add_argument('--update', action='store_true', help='增量补齐（无 --date/--start 时的默认行为）')
    ap.add_argument('--force', action='store_true', help='已有文件也重抓')
    ap.add_argument('--sleep', type=float, default=2.5, help='请求间隔秒（默认 2.5）')
    args = ap.parse_args()

    os.makedirs(KPL_DIR, exist_ok=True)
    dates = resolve_dates(args)
    todo = [d for d in dates if args.force or not os.path.exists(f'{KPL_DIR}/{d}.json')]
    logger.info(f'计划抓取 {len(todo)}/{len(dates)} 天（跳过已存在）')
    ok = skip = fail = empty = 0
    for d in dates:
        out_f = f'{KPL_DIR}/{d}.json'
        if os.path.exists(out_f) and not args.force:
            skip += 1
            continue
        try:
            j = fetch_day(d)
            actual = j.get('date') or j.get('Day')
            j['_actual_date'] = actual
            j['_fallback'] = actual != d  # 非交易日会回退到最近交易日
            n_stock = sum(len(t.get('StockList') or []) for t in j.get('list') or [])
            if n_stock == 0:
                empty += 1
                logger.warning(f'[{d}] 空数据(0题材0个股) 不落盘')
            else:
                with open(out_f, 'w', encoding='utf-8') as f:
                    json.dump(j, f, ensure_ascii=False)
                ok += 1
                logger.info(f'[{d}] OK 题材 {len(j.get("list") or [])} 个股 {n_stock}' +
                            (f' (回退->{actual})' if j['_fallback'] else ''))
        except Exception as e:
            fail += 1
            logger.error(f'[{d}] 失败 {type(e).__name__}: {e}')
        time.sleep(args.sleep + random.uniform(0, args.sleep * 0.4))
    logger.info(f'完成: ok={ok} skip={skip} fail={fail} empty={empty} → {KPL_DIR}')
    if fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
