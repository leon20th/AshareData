# -*- coding: utf-8 -*-
# scrap_cninfo —— 巨潮资讯网「公告」全市场抓取（per-stock 通道，免登录、无签名）
# 数据范围: 2020-01-01 起（实测可回溯至 ≤2010）；沪深全市场（北交所需另行并入公司表）
# 产物: scrap_data/cninfo/stock/<code>.json   单票公告全文（slim，按 announcementId 去重）
#       scrap_data/cninfo/_orgids.json       code→orgId 表（增量补）
#       scrap_data/cninfo/_state_stock.json  断点（每票条数 + 最新日期）
#       scrap_data/cninfo/_heartbeat.json    看门狗心跳
#
# 为什么是 per-stock（而不是按日翻页）:
#   按「日」查 (seDate=D~D) 会被服务端硬顶 **3000 条/次**：pageNum>100 会静默回落第 1 页，
#   而日粒度上找不到任何完备的二次切分维度 ——
#     · pageSize 硬顶 30（传 int/str/500 都只回 30）
#     · column 完全无效（szse/sse/bj 返回逐字节相同）
#     · plate 只认 sz/sh/bj；cyb/kcb/代码前缀/中文名一律被忽略
#     · category 有效类目只覆盖约 64%，其余无类目可查
#     · seDate 虽支持秒级，但 97% 公告的 announcementTime 被归零到 00:00:00，切不动
#   单只股票 6.8 年公告量仅数百条（如 600519=493 条/17 页），永不触顶 → 完备。
#
# 频控: 接口有 WAF 路径级限流 —— 超 ~9 req/s 会被 403 封锁整条约 9 分钟。
#       默认 --workers 5 --sleep 0.7（约 5.5~7 req/s）；撞 403 则**全局暂停** 10/20/40 分钟续爬。
#
# 用法:
#   python AshareData/datautils/regime_scripts/scrap_cninfo.py --build-orgids   # 只建/补 orgId 表
#   python AshareData/datautils/regime_scripts/scrap_cninfo.py --start 2020-01-01   # 全量回扫
#   python AshareData/datautils/regime_scripts/scrap_cninfo.py --incremental       # 每日增量
#   python AshareData/datautils/regime_scripts/scrap_cninfo.py --limit 20          # 冒烟
#   python AshareData/datautils/regime_scripts/scrap_cninfo.py --latest            # 只问答状态
import argparse
import glob
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone

import requests

from AshareData.paths import (DAILY_KLINE_DIR, DAILY_KLINE_TS_DIR, DAILY_KLINE_V2_DIR,
                             SCRAP_DATA_DIR)
from AshareData.utils.exchanges_utils.a_open import get_target_trade_date
from AshareData.utils.log_util import get_logger

logger = get_logger('巨潮公告抓取')

UA = ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
      '(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36')
API = 'http://www.cninfo.com.cn/new/hisAnnouncement/query'
SEARCH_API = 'http://www.cninfo.com.cn/new/information/topSearch/query'
OUT_DIR = f'{SCRAP_DATA_DIR}/cninfo'
STOCK_DIR = f'{OUT_DIR}/stock'
ORG_F = f'{OUT_DIR}/_orgids.json'
STATE_F = f'{OUT_DIR}/_state_stock.json'
HEART_F = f'{OUT_DIR}/_heartbeat.json'
TZ = timezone(timedelta(hours=8))
PAGE_SIZE = 30            # 服务端硬上限
# 日K库候选目录（双管线：tushare → daily_kline_ts；mix → daily_kline_v2；另留 daily_kline 老库）。
# 默认取并集：哪条管线有库就用哪个。多抓的只是冗余、成本可忽略；只认一条会在换管线时
# **静默失效**（universe 变空 → “无待抓”退出）。
KLINE_DIRS = (DAILY_KLINE_TS_DIR, DAILY_KLINE_V2_DIR, DAILY_KLINE_DIR)
DAY_F = re.compile(r'\d{4}-\d{2}-\d{2}\.json')
CODE_RE = re.compile(r'\d{6}')
COLS = ('announcementId', 'secCode', 'secName', 'announcementTitle', 'shortTitle',
        'announcementTime', 'announcementType', 'announcementTypeName',
        'adjunctUrl', 'adjunctSize', 'adjunctType', 'orgId', 'important', 'batchNum')
HDR = {'Origin': 'http://www.cninfo.com.cn',
       'Referer': 'http://www.cninfo.com.cn/new/commonUrl?url=disclosure/list/notice',
       'X-Requested-With': 'XMLHttpRequest',
       'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8'}

# —— 全局频控（多线程共享）——
_plock = threading.Lock()
_paused_until = [0.0]
_ban_n = [0]
BAN_BACKOFF = [600, 1200, 2400]     # 10 / 20 / 40 分钟
_last_act = [time.time()]           # 最近一次「有动静」的时刻（看门狗据此判卡死）
_req_n = [0]
_MAXREQ = [0]
_tls = threading.local()


class _Stop(Exception):
    """达到 --max-req 的哨兵。"""


def _session():
    s = getattr(_tls, 's', None)
    if s is None:
        s = requests.Session()
        s.headers.update({'User-Agent': UA, **HDR})
        _tls.s = s
    return s


def _touch():
    _last_act[0] = time.time()


def _wait_pause():
    while True:
        with _plock:
            remain = _paused_until[0] - time.time()
        if remain <= 0:
            return
        time.sleep(min(remain, 5))


def _on_banned():
    with _plock:
        wait = BAN_BACKOFF[min(_ban_n[0], len(BAN_BACKOFF) - 1)]
        _ban_n[0] += 1
        _paused_until[0] = max(_paused_until[0], time.time() + wait)
        n = _ban_n[0]
    logger.warning(f'⚠ WAF 403（第 {n} 次）→ 全局暂停 {wait // 60} 分钟，之后自动续爬')


def _post(url, data, sleep):
    """带节流 + 403 全局暂停 + 短退避的一发请求；返回 (json, 是否成功)。"""
    for attempt in range(20):
        _wait_pause()
        with _plock:
            if _MAXREQ[0] and _req_n[0] >= _MAXREQ[0]:
                raise _Stop()
            _req_n[0] += 1
        try:
            r = _session().post(url, data=data, timeout=30)
            _touch()
        except Exception as e:
            _touch()
            wait = min(5 * (attempt + 1), 60)
            logger.warning(f'网络异常({type(e).__name__}: {e}) → {wait}s 重试')
            time.sleep(wait)
            continue
        finally:
            time.sleep(sleep + random.uniform(0, sleep * 0.3))
        if r.status_code == 403:
            _on_banned()
            continue
        if r.status_code != 200 or not r.content:
            wait = min(5 * (attempt + 1), 60)
            logger.warning(f'HTTP {r.status_code} → {wait}s 重试')
            time.sleep(wait)
            continue
        try:
            return r.json(), True
        except Exception:
            logger.warning('非 JSON 响应 → 重试')
            time.sleep(5)
    return None, False


def _slim(a):
    o = {k: a.get(k) for k in COLS}
    for k in ('announcementTitle', 'shortTitle'):
        if o.get(k):
            o[k] = re.sub(r'<[^>]+>', '', o[k]).strip()
    return o


def _day_of(ms):
    return datetime.fromtimestamp(int(ms) / 1000, TZ).strftime('%Y-%m-%d')


# ---------- orgId 表 ----------

def universe():
    """公司全集 = **有日K的票**（只用仓库内可复现的来源，覆盖全部日K管线）。

    口径就是「有日K才抓」：没日K的票既算不出特征、也对不上标签，抓它的公告没意义。
    刻意**不扫** scrap_data/cninfo 下已有的产物：那会让“抓到什么”反过来决定“该抓什么”，
    新机器上跑出来的全集跟本机不一致（不可复现）。
    B 股（200xxx 深B / 900xxx 沪B）不在 A 股日K库，自然被排除 —— 上游
    `tushare_kline.update_meta()` 也显式去了它们（900xxx.SH / 200xxx.SZ）。
    """
    codes = set()
    for d in KLINE_DIRS:
        for p in glob.glob(f'{d}/*.*.csv'):
            m = os.path.basename(p).split('.')
            if len(m) >= 2 and CODE_RE.fullmatch(m[1]):
                codes.add(m[1])
    return sorted(codes)


def seed_orgids():
    """从已抓的 per-stock 产物里白捡 orgId（每行都带该字段，同一只票相同）。

    仅当 `_orgids.json` 缺失时当快路径用；**不参与**公司全集的定义（否则产物会反过来
    决定输入，新机器上不可复现）。每只票只取第一条记录的 orgId。
    """
    out = {}
    for p in glob.glob(f'{STOCK_DIR}/*.json'):
        try:
            rows = json.load(open(p, encoding='utf-8'))
        except Exception:
            continue
        code, o = os.path.basename(p)[:-5], (rows[0].get('orgId') if rows else None)
        if o:
            out[code] = o
    return out


def harvest_orgids(codes, sleep, workers, org):
    """对缺失的代码逐个 topSearch 取 orgId；边补边落盘。返回本次新增的 code→orgId。"""
    got, fail = {}, 0

    def one(code):
        j, ok = _post(SEARCH_API, {'keyWord': code, 'maxNum': 10}, sleep)
        if not ok or not isinstance(j, list):
            return None, code
        for it in j:
            if it.get('code') == code and it.get('orgId'):
                return it['orgId'], code
        return None, code

    logger.info(f'orgId 缺失 {len(codes)} 个 → 开始 topSearch 补全')
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, (o, code) in enumerate(ex.map(one, codes), 1):
                if o:
                    got[code] = o
                    org[code] = o
                else:
                    fail += 1
                if i % 250 == 0:
                    dump_json(ORG_F, org)        # 边补边存，中途被杀不白跑
                    logger.info(f'  orgId 进度 {i}/{len(codes)} 成功={len(got)} 失败={fail}')
    except _Stop:
        logger.info('达到 --max-req，orgId 补全提前结束')
    logger.info(f'orgId 补全结束: 新增 {len(got)}，失败 {fail}')
    return got


# ---------- 抓取 ----------

def _probe(code, orgid, lo, hi, page, sleep):
    j, ok = _post(API, {'pageNum': page, 'pageSize': PAGE_SIZE, 'column': 'szse',
                        'tabName': 'fulltext', 'plate': '', 'stock': f'{code},{orgid}',
                        'searchkey': '', 'secid': '', 'category': '', 'trade': '',
                        'seDate': f'{lo}~{hi}', 'sortName': '', 'sortType': '',
                        'isHLtitle': 'false'}, sleep)
    if not ok:
        raise RuntimeError(f'{code} {lo}~{hi} p{page} 重试耗尽')
    return j


def fetch_range(code, orgid, lo, hi, sleep, depth=0):
    """拉一只票在 [lo,hi] 内的全部公告；返回 (rows, nreq)。

    3000 条/查询是服务端硬顶，故 total 超限时按**日期区间二分递归** ——
    日期是真分区，递归后必然完备（per-stock 之外的第二道保险）。
    """
    j = _probe(code, orgid, lo, hi, 1, sleep)
    nreq = 1
    total = int(j.get('totalAnnouncement') or 0)
    if total > 100 * PAGE_SIZE:
        a, b = date.fromisoformat(lo), date.fromisoformat(hi)
        if a < b:
            m = a + timedelta(days=(b - a).days // 2)
            if depth == 0:
                logger.info(f'{code} {lo}~{hi} 共 {total} 条 > 3000 → 按日期二分递归')
            r1, q1 = fetch_range(code, orgid, lo, m.strftime('%Y-%m-%d'), sleep, depth + 1)
            r2, q2 = fetch_range(code, orgid, (m + timedelta(days=1)).strftime('%Y-%m-%d'),
                                 hi, sleep, depth + 1)
            return r1 + r2, nreq + q1 + q2
        logger.warning(f'{code} {lo}~{hi} total={total} > 3000 且无法再细分（单日）')
    rows = list(j.get('announcements') or [])
    first_id = rows[0]['announcementId'] if rows else None
    for p in range(2, -(-total // PAGE_SIZE) + 1):
        jj = _probe(code, orgid, lo, hi, p, sleep)
        nreq += 1
        ann = jj.get('announcements') or []
        if not ann:
            break
        if p > 100 and ann[0]['announcementId'] == first_id:
            logger.warning(f'{code} {lo}~{hi} pageNum>100 回落首页 → 停止翻页')
            break
        rows += ann
    return rows, nreq


def fetch_stock(code, orgid, lo, sleep):
    """拉一只票从 lo 到今天的公告（内部按需递归细分）。"""
    return fetch_range(code, orgid, lo, date.today().strftime('%Y-%m-%d'), sleep)


def save_stock(code, rows):
    """合并写入单票文件（按 announcementId 去重，原子发布）；返回新增条数。"""
    f = f'{STOCK_DIR}/{code}.json'
    old = json.load(open(f, encoding='utf-8')) if os.path.exists(f) else []
    merged = {str(x['announcementId']): x for x in old}
    n_new = 0
    for r in rows:
        s = _slim(r)
        k = str(s['announcementId'])
        if k not in merged:
            merged[k] = s
            n_new += 1
    tmp = f + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(list(merged.values()), fh, ensure_ascii=False)
    os.replace(tmp, f)
    return n_new, len(merged)


# ---------- 状态 / 心跳 ----------

def load_json(f, default):
    try:
        return json.load(open(f, encoding='utf-8'))
    except Exception:
        return default


def dump_json(f, obj):
    tmp = f + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, ensure_ascii=False)
    os.replace(tmp, f)


def heartbeat_loop(meta):
    while not meta.get('stop'):
        dump_json(HEART_F, {'t': _last_act[0], 'paused_until': _paused_until[0],
                            'req': _req_n[0], 'stocks_done': meta.get('done', 0),
                            'stocks_total': meta.get('total', 0),
                            'state': meta.get('state', 'running'),
                            'pid': os.getpid(), 'updated': time.time()})
        time.sleep(10)


def is_latest():
    """整库是否已覆盖到最近交易日（读 _state_stock.json 的 newest）。"""
    target = get_target_trade_date()
    st = load_json(STATE_F, None)
    if not target or not st:
        return False
    newest = st.get('newest')
    tgt = f'{target[:4]}-{target[4:6]}-{target[6:8]}'
    return bool(newest) and newest >= tgt


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--start', default='2020-01-01', help='全量回扫下界')
    ap.add_argument('--since', default=None, help='只取该日之后的公告（体检/定点补漏）')
    ap.add_argument('--incremental', action='store_true',
                    help='增量模式：每票只从上次已见的最新日期起抓（每日链用）')
    ap.add_argument('--workers', type=int, default=5, help='并发票数（默认 5）')
    ap.add_argument('--sleep', type=float, default=0.7, help='每请求间隔秒（默认 0.7）')
    ap.add_argument('--max-req', type=int, default=0, help='最多请求数（冒烟）')
    ap.add_argument('--limit', type=int, default=0, help='只处理前 N 只（冒烟）')
    ap.add_argument('--build-orgids', action='store_true', help='只建/补 orgId 表后退出')
    ap.add_argument('--latest', action='store_true', help='只问答状态，不抓数据')
    args = ap.parse_args()

    if args.latest:
        logger.info(f'is_latest = {is_latest()}')
        sys.exit(0 if is_latest() else 1)

    os.makedirs(STOCK_DIR, exist_ok=True)
    _MAXREQ[0] = args.max_req
    meta = {'done': 0, 'total': 0, 'state': 'starting', 'stop': False}
    dump_json(HEART_F, {'state': 'starting', 't': time.time(), 'stocks_done': 0,
                        'stocks_total': 0, 'paused_until': 0, 'req': 0,
                        'pid': os.getpid(), 'updated': time.time()})
    threading.Thread(target=heartbeat_loop, args=(meta,), daemon=True).start()
    st = load_json(STATE_F, {}) or {}
    st.setdefault('stocks', {})
    org = load_json(ORG_F, {}) or {}

    codes = universe()
    if args.limit:
        codes = codes[:args.limit]
    codes = [c for c in codes if not c.startswith('_')]
    logger.info(f'公司全集 {len(codes)} 只（日K目录: '
                f'{[os.path.basename(d) for d in KLINE_DIRS if os.path.isdir(d)]}）；'
                f'已有 orgId {len(org)}；已完成 {len(st["stocks"])} 只；'
                f'并发 {args.workers} × {args.sleep}s')

    # 1) orgId 表：先从历史日文件白捡，再 topSearch 补缺
    if not org:
        org = seed_orgids()
        if org:
            logger.info(f'从历史日文件白捡 orgId {len(org)} 个')
            dump_json(ORG_F, org)
    missing = [c for c in codes if c not in org]
    if missing:
        org.update(harvest_orgids(missing, args.sleep, args.workers, org))
        dump_json(ORG_F, org)
    if args.build_orgids:
        logger.info(f'--build-orgids 完成：共 {len(org)} 个 → {ORG_F}')
        sys.exit(0)

    todo = [c for c in codes if c in org and (args.incremental or c not in st['stocks'])]
    if args.since:
        todo = [c for c in codes if c in org]
    logger.info(f'待抓 {len(todo)} 只（增量={args.incremental}）')
    if not todo:
        logger.info('无待抓，退出')
        dump_json(HEART_F, {'state': 'partial' if (args.limit or args.since) else 'done',
                            't': time.time(), 'stocks_done': len(st['stocks']),
                            'stocks_total': len(codes), 'paused_until': 0, 'req': _req_n[0],
                            'pid': os.getpid(), 'updated': time.time()})
        sys.exit(0)

    meta.update({'done': len(st['stocks']), 'total': len(codes), 'state': 'running'})

    t0, n_ok, n_rows, n_fail = time.time(), 0, 0, 0
    req0 = _req_n[0]
    consecutive_fail = [0]

    def job(code):
        if args.since:
            lo = args.since
        elif args.incremental:
            prev = st['stocks'].get(code, {}).get('newest')
            lo = (date.fromisoformat(prev) - timedelta(days=1)).strftime('%Y-%m-%d') \
                if prev else args.start
        else:
            lo = args.start
        rows, nreq = fetch_stock(code, org[code], lo, args.sleep)
        n_new, n_all = save_stock(code, rows)
        newest = max((_day_of(x['announcementTime']) for x in rows), default=None)
        return code, n_new, n_all, newest, nreq

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(job, c): c for c in todo}
            for fut in as_completed(futs):
                code = futs[fut]
                try:
                    code, n_new, n_all, newest, nreq = fut.result()
                except _Stop:
                    meta['state'] = 'maxreq'
                    continue
                except Exception as e:
                    n_fail += 1
                    consecutive_fail[0] += 1
                    logger.error(f'{code} 失败: {type(e).__name__}: {e}')
                    if consecutive_fail[0] >= 20:
                        raise RuntimeError(f'连续失败 {consecutive_fail[0]} 只，主动退出交给看门狗')
                    continue
                consecutive_fail[0] = 0
                prev = st['stocks'].get(code, {})
                st['stocks'][code] = {'n': n_all, 'newest': newest or prev.get('newest')}
                n_ok += 1
                n_rows += n_new
                meta['done'] = len(st['stocks'])
                if n_ok % 25 == 0 or n_ok == 1:
                    el = time.time() - t0
                    dr = _req_n[0] - req0
                    eta = (len(todo) - n_ok) * el / n_ok
                    logger.info(f'[{n_ok}/{len(todo)}] {code} 新增{n_new} 累计{n_all} | '
                                f'总计 {n_rows:,} 行 req={dr:,} {dr / el:.1f} req/s '
                                f'已用 {el / 60:.0f}min ETA {eta / 60:.0f}min')
                    dump_json(STATE_F, st)
    except KeyboardInterrupt:
        logger.warning('中断 → 保存断点（重跑即续传）')
    except Exception as e:
        logger.error(f'异常中止: {type(e).__name__}: {e}')
        dump_json(STATE_F, st)
        meta['stop'] = True
        meta['state'] = 'failed'
        dump_json(HEART_F, {'state': 'failed', 't': time.time(), 'stocks_done': len(st['stocks']),
                            'stocks_total': len(codes), 'paused_until': 0, 'req': _req_n[0],
                            'pid': os.getpid(), 'updated': time.time()})
        sys.exit(1)

    st['newest'] = max((v.get('newest') or '' for v in st['stocks'].values()), default=None)
    done = len(codes) <= len(st['stocks'])
    dump_json(STATE_F, st)
    meta['stop'] = True
    if meta['state'] == 'maxreq':
        pass
    elif done and not (args.limit or args.since):
        meta['state'] = 'done'
    else:
        meta['state'] = 'partial'    # --limit/--since 冒烟跑不可冒充 done，否则看门狗会误退
    dump_json(HEART_F, {'state': meta['state'], 't': time.time(),
                        'stocks_done': len(st['stocks']), 'stocks_total': len(codes),
                        'paused_until': 0, 'req': _req_n[0], 'pid': os.getpid(),
                        'updated': time.time()})
    el = time.time() - t0
    logger.info(f'结束: 成功 {n_ok} 只 / 失败 {n_fail} 只 / 新增 {n_rows:,} 条；'
                f'请求 {_req_n[0] - req0:,} 次，用时 {el / 60:.1f}min；'
                f'覆盖 {len(st["stocks"])}/{len(codes)} 只，state={meta["state"]}')
    if meta['state'] == 'maxreq':
        sys.exit(0)
    # 未覆盖全部 → 非 0 退出，交给看门狗重启续爬
    sys.exit(0 if done else 1)


if __name__ == '__main__':
    main()
