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
#       增量下界 = 上次已见最新日 − N 天（--overlap-days，默认 7）：**往回覆盖一段**。
#       announcementId 去重 ⇒ 重复抓幂等，只多花请求。这是防「迟到公告」的唯一可行手段：
#       秒级下界不可用 —— 97% 的 announcementTime 被归零到 00:00:00，而它们正是训练主群，
#       用时间戳当下界会让这批再也不会被查到（制造缺口）。
#       --incremental 走**双通道**：
#         ① 按日全市场查（快速路径，≈12× 省请求）：逐日 1 个探测拿 totalAnnouncement，
#            再跨日并发翻页抓完并分发回各票。实测 8 天窗口 = 227 请求 / 1.2 min；
#            同期按票需 5,448 请求 / 13~16 min。（按日查单请求更重：延迟 ~0.76s vs 按票
#            ~0.14s ⇒ 5 线程 ×0.7s 节流下只能到 3.2 req/s，但请求数少两个量级仍赢。）
#            探测到的 total 顺带就是**缺口审计基准**（该日全市场条数 vs 库内条数）。
#         ② 按票（完备但慢）：**超限日回退**用。按日查被硬顶 3,000 条（30×100，超出静默
#            回落第 1 页），实测 11.2% 的自然日 > 3,000（最忙 28,390 条/天），且日粒度上
#            无可靠二次切分维度（column/plate/category 实测无效）⇒ 那些天按日查不完备。
#            窗口 > DAY_MODE_MAX_DAYS(60) 天时也整体回退按票；而「有票无断点」（新上市/
#            冷库需全史）**不阻塞**按日路径 —— 那几只单独交按票跑即可。
#            **宁可慢也不能不完备**：闸门不满足即整体回退。
#       要报「整库无缺口」需定期全量重扫审计（同样幂等）：--start 2020-01-01
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
DAY_CAP = PAGE_SIZE * 100  # 3000：按日全市场查的硬顶（pageNum>100 静默回落第 1 页）
DAY_MODE_MAX_DAYS = 60     # 追更窗口超过这么多天就不走按日模式（逐日探测+回退代价失控）
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


def _incr_lo(prev, overlap_days, start):
    """增量下界 = 上次已见最新日的**前 overlap_days 天**（迟到公告的回填窗口）。

    为何不做到秒：`announcementTime` 有 97% 被归零到 00:00:00，而这些**正是训练主群**
    （归零识别出 89.91%）。若拿「上次最大时间戳」当下界，所有时间戳 ≤ 它的归零公告
    就再也不会被查到 ⇒ 秒级下界会**制造**缺口，而不是消除它。
    真正能防漏的是「往回覆盖一段」+ `announcementId` 去重（幂等，只多花请求）。

    prev 为 None（无断点/新挂票）⇒ 返回 start（全量下界）；不会退到 start 之前。
    """
    if not prev:
        return start
    lo = (date.fromisoformat(prev)
          - timedelta(days=max(int(overlap_days), 0))).strftime('%Y-%m-%d')
    return max(lo, start)


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


# ---------- 按日全市场通道（增量快速路径）----------
# 为何存在：按票增量即使「零新增」也要 5,448 个请求（≈13 min @7.1 req/s）；
# 而按日全市场只需 Σ ceil(当日条数/30) + 每日 1 探测 ⇒ 最近 7 天约 301 请求 ≈ 42 s（~18×）。
# 为何只是「快速路径」而不能取代按票：按日查被硬顶 3,000 条（30×100，超出静默回落第 1 页），
# 而实测 11.2% 的自然日 > 3,000（最忙 28,390 条/天），且日粒度上无可靠二次切分维度
# （column/plate/category 实测无效、pageSize 硬顶 30、97% 时间戳归零）⇒ 那些天**不完备**。
# 故：超限日一律回退按票（按票的窗口是那一天，每票约 1 请求）。
def _probe_day(day, page, sleep):
    """按日**全市场**查（不带 stock 过滤）。seDate 同样只用到日粒度。"""
    j, ok = _post(API, {'pageNum': page, 'pageSize': PAGE_SIZE, 'column': 'szse',
                        'tabName': 'fulltext', 'plate': '', 'stock': '',
                        'searchkey': '', 'secid': '', 'category': '', 'trade': '',
                        'seDate': f'{day}~{day}', 'sortName': '', 'sortType': '',
                        'isHLtitle': 'false'}, sleep)
    if not ok:
        raise RuntimeError(f'day {day} p{page} 重试耗尽')
    return j


def probe_day(day, sleep):
    """1 个请求：该日全市场条数 + 首页行。

    total 顺便就是**缺口审计的基准**：与库内该日条数一比即知有没有丢。
    """
    j = _probe_day(day, 1, sleep)
    return int(j.get('totalAnnouncement') or 0), list(j.get('announcements') or [])


def code_of(a):
    """行 → 6 位代码（按日查返回 secCode='002475'，与 universe() 口径一致）。"""
    c = str(a.get('secCode') or '').strip()
    return c.zfill(6) if c else ''


def save_day_rows(rows, cset):
    """按日行 → 按 secCode 分发回各票文件（announcementId 去重）。

    返回 ({code: (n_new, n_all)}, 网格外行数)。网格外 = 北交/退市等不在「有日K」全集里的票。
    """
    by, off = {}, 0
    for r in rows:
        c = code_of(r)
        if c and c in cset:
            by.setdefault(c, []).append(r)
        else:
            off += 1
    return {c: save_stock(c, rs) for c, rs in by.items()}, off


def run_day_mode(codes, st, meta, args):
    """按日抓增量窗口。返回 (ok, leftover)：

      ok=True  ⇒ 已按日覆盖（leftover = 仍需交按票的票，通常是缺断点的新票/冷库）
      ok=False ⇒ 必须**整体**回退按票

    两道闸门，任一不满足即整体回退（宁可慢也不能不完备）：
      1) 窗口天数 ≤ DAY_MODE_MAX_DAYS
      2) 窗口内**每一天**全市场条数 ≤ DAY_CAP（超限日无法完备切分）
    缺断点的票（新上市/冷库）**不阻塞**按日路径：它们需要全史，单独交按票跑。
    本模式不需要 orgId（不带 stock 过滤）。
    """
    leftover = [c for c in codes if c not in st['stocks']]      # 需全史 ⇒ 交按票
    known = [c for c in codes if c in st['stocks']]
    if not known:
        return False, list(codes)
    prev_days = [st['stocks'][c].get('newest') for c in known
                 if st['stocks'][c].get('newest')]
    if not prev_days and not st.get('covered_through'):
        logger.info('按日模式跳过：无历史断点')
        return False, list(codes)
    # 窗口基准：按日全市场查覆盖到哪天（covered_through）；没有就退回「全局最新公告日」
    basis = st.get('covered_through') or max(prev_days)
    lo = _incr_lo(basis, args.overlap_days, args.start)
    d0, d1 = date.fromisoformat(lo), date.today()
    days = [(d0 + timedelta(days=i)).isoformat() for i in range((d1 - d0).days + 1)]
    if len(days) > DAY_MODE_MAX_DAYS:
        logger.info(f'按日模式跳过：窗口 {len(days)} 天 > {DAY_MODE_MAX_DAYS}')
        return False, list(codes)

    t0, req0 = time.time(), _req_n[0]
    totals, first = {}, {}
    try:                                     # 逐日探测（并发）：1 请求/天，拿 total + 首页
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(probe_day, d, args.sleep): d for d in days}
            for f in as_completed(futs):
                totals[futs[f]], first[futs[f]] = f.result()
    except Exception as e:
        logger.error(f'按日探测失败({type(e).__name__}: {e}) → 回退按票')
        return False, list(codes)
    over = [d for d in days if totals[d] > DAY_CAP]
    logger.info(f'按日模式：窗口 {days[0]}~{days[-1]}（{len(days)} 天）；'
                f'每日条数 {[totals[d] for d in days]}')
    if over:
        logger.info(f'按日模式放弃：{len(over)} 天超 {DAY_CAP} 条硬顶（无法完备切分）→ 回退按票')
        return False, list(codes)

    # 所有 (日, 页>1) 一次性并发（跨日并发 ⇒ 消除“每天等批尾”的停顿；频控在 _post 内每线程生效）。
    # 收到就 _slim（丢掉 announcementContent 等大字段）⇒ 窗口拉到几十天也不爆内存。
    tasks = [(d, p) for d in days
             for p in range(2, -(-totals[d] // PAGE_SIZE) + 1)]
    extra = {d: [] for d in days}
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_probe_day, d, p, args.sleep): d for d, p in tasks}
            for f in as_completed(futs):
                extra[futs[f]] += [_slim(r) for r in (f.result().get('announcements') or [])]
    except Exception as e:
        logger.error(f'按日抓取失败({type(e).__name__}: {e}) → 回退按票')
        return False, list(codes)

    cset, n_new, n_off = set(codes), 0, 0
    nall, seen = {}, {}
    for d in days:
        rows = [_slim(r) for r in first[d]] + extra[d]
        nreq = 1 + len(range(2, -(-totals[d] // PAGE_SIZE) + 1))
        info, off = save_day_rows(rows, cset)
        got = sum(v[0] for v in info.values())
        n_new, n_off = n_new + got, n_off + off
        for r in rows:                        # 该票「已见最新公告日」（不存行、不占内存）
            c = code_of(r)
            if c in cset and r.get('announcementTime'):
                dd = _day_of(r['announcementTime'])
                if dd > seen.get(c, ''):
                    seen[c] = dd
        for c, (_nn, na) in info.items():
            nall[c] = na          # 该票最新总条数（供断点记录）
        logger.info(f'  按日 {d}：{totals[d]:,d} 条 / {nreq} 请求 → 新增 {got}'
                    f'（{len(info)} 只；网格外 {off}）')
    last = days[-1]
    # 断点语义与按票一致：newest = 该票**已见最新公告日**（数据驱动，供 is_latest/看门狗用）；
    # 「全市场覆盖到哪天」另记 covered_through（下轮窗口据此算），不拿它顶替 newest
    # （否则 is_latest 永远为真、看门狗误判）。
    for c in known:                          # leftover 的票不动状态（由按票路径写）
        prev = st['stocks'].get(c, {})
        st['stocks'][c] = {'n': nall.get(c, prev.get('n', 0)),
                           'newest': seen.get(c) or prev.get('newest')}
    st['covered_through'] = last
    st['newest'] = max((v.get('newest') or '' for v in st['stocks'].values()), default=None)
    dump_json(STATE_F, st)
    dr, el = _req_n[0] - req0, time.time() - t0
    logger.info(f'按日模式完成：新增 {n_new:,d} 行（网格外 {n_off:,d}）；'
                f'{dr:,d} 请求 / {el / 60:.1f} min（{dr / max(el, 1e-9):.1f} req/s）')
    dump_json(HEART_F, {'state': 'done', 't': time.time(), 'stocks_done': len(st['stocks']),
                        'stocks_total': len(codes), 'paused_until': 0, 'req': _req_n[0],
                        'pid': os.getpid(), 'updated': time.time()})
    return True, leftover


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
                    help='增量模式：每票从上次已见的最新日期起抓（每日链用）')
    ap.add_argument('--overlap-days', type=int, default=7,
                    help='增量回溯天数：下界 = 上次已见最新日 − N 天（默认 7）。'
                         '防迟到/延迟可见的公告；announcementId 去重 ⇒ 幂等，只多花请求。'
                         '无法靠秒级下界代替：97%% 的 announcementTime 被归零到 00:00:00')
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

    # ---- 增量快速路径：按日全市场查（~18× 省请求）；闸门不满足就整体回退按票 ----
    if args.incremental and not args.since and not args.limit:
        try:
            ok, left = run_day_mode(codes, st, meta, args)
        except Exception as e:
            ok, left = False, list(codes)
            logger.error(f'按日模式异常({type(e).__name__}: {e})')
        if ok:
            if not left:
                return
            logger.info(f'按日已覆盖增量窗口；剩 {len(left)} 只缺断点/冷票交按票（需全史）')
            todo = [c for c in left if c in org]
            if not todo:
                return
        else:
            logger.info('回退按票模式（完备但慢）')

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
            lo = _incr_lo(st['stocks'].get(code, {}).get('newest'),
                          args.overlap_days, args.start)
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
