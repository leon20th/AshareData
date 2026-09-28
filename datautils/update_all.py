# -*- coding: utf-8 -*-
"""update_all —— 依次更新所有依赖数据（日K（双管线可选）/ 共用 m15 / 停牌名单 / 新闻双源 / 榜单四源 / 特征构建）。

日K 双管线（--daily 切换；默认 tushare）：
  tushare    daily_kline_ts（tushare 官方接口，收费）；停牌名单轨 = tushare suspend_d
  mix        daily_kline_v2（扶摇 dump + baostock + 新浪，免费）；停牌名单轨 = fuyao dump 推断
m15 两管线共用一套（m15_kline_ts；新浪+baostock；kline_scripts/m15/update_m15.py；
含"源坏自动跳过缺口"）。特征构建读口随管线（daily_kline_ts / daily_kline_v2）。

顺序：
  1) 日K         选定管线（缺口判定 + 最小请求，可中断续传）
  2) m15         共用库增量（新浪窗内 + baostock 老段；源缺自动跳过）
  3) 停牌名单    随管线选轨（tushare suspend_d / fuyao dump）
  4) 新闻        财联社 / 东财7x24（增量；空目录自动从 2020-01-01 回填）
  5) 榜单        开盘啦 / 涨跌停（问财）/ 龙虎榜（增量补齐缺失日期）
  6) 特征构建    base_feature（自增量；读口随 --daily）

约定：
  - 空库一律从 2020-01-01 起；各步幂等，可随时中断重跑（缺口/断点驱动）
  - 每步失败自动重试（--retry，默认 2 次重试／退避 20s・40s）；最终失败不阻塞后续，
    结束时 [ALERT] 告警 + 状态写入 .cache/update_all_status.log + 非 0 退出
  - 输出：各步**原样透出**——子进程继承终端，脚本自己的日志/进度条怎么打就怎么显示，
    本脚本不捕获、不整理、不落盘，只在每步前后各打一行边界标记 + 末尾一张汇总表
用法:
  python AshareData/datautils/update_all.py                  # 全流程（默认 tushare 管线）
  python AshareData/datautils/update_all.py --daily mix      # mix_source 管线（免费源）
  python AshareData/datautils/update_all.py --skip feature   # 跳过指定步骤
  python AshareData/datautils/update_all.py --retry 1        # 每步仅重试 1 次
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
ASHARE = os.path.dirname(HERE)
ROOT = os.path.dirname(ASHARE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AshareData.paths import DAILY_KLINE_TS_DIR, M15_KLINE_TS_DIR

PY = sys.executable
D = f'{ASHARE}/datautils'
TSK = f'{D}/kline_scripts/tushare/tushare_kline.py'
MIXK = f'{D}/kline_scripts/mix_source/quick_kline.py'
M15K = f'{D}/kline_scripts/m15/update_m15.py'
FLOOR = '2020-01-01'
ALERT_F = f'{ASHARE}/.cache/update_all_status.log'   # 每步结果存档（终端不显示）


def _count(d):
    return len([f for f in os.listdir(d) if f.endswith('.csv')]) if os.path.isdir(d) else 0


def plan_steps(daily='tushare'):
    ts_n, m15_n = _count(DAILY_KLINE_TS_DIR), _count(M15_KLINE_TS_DIR)
    if daily == 'tushare':
        # tushare_kline 自己判断每票每字段的缺口与手段，这里不再传模式/区间
        daily_step = ('v2', '日K·tushare（daily_kline_ts）', [PY, '-u', TSK])
        notrade_src = 'tushare'
    else:
        # quick_kline 同理；--skip m15 避免它写老 m15 库（m15 由共用模块负责）
        daily_step = ('v2', '日K·mix_source（daily_kline_v2：扶摇+baostock）', [PY, '-u', MIXK, '--skip', 'm15'])
        notrade_src = 'fuyao'
    steps = [
        daily_step,
        ('m15', 'm15（共用·新浪+baostock）', [PY, '-u', M15K, '--daily-ref', daily]),
        ('notrade', f'停牌名单·{notrade_src}', [PY, '-u', f'{D}/kline_scripts/update_notrade.py', '--source', notrade_src]),
        ('cls', '新闻·财联社', [PY, '-u', f'{D}/regime_scripts/scrap_news_cls.py']),
        ('em724', '新闻·东财7x24', [PY, '-u', f'{D}/regime_scripts/scrap_news_em724.py']),
        ('kaipanla', '榜单·开盘啦', [PY, '-u', f'{D}/regime_scripts/scrap_kaipanla.py', '--start', FLOOR]),
        ('dietingban', '榜单·跌停板（问财）', [PY, '-u', f'{D}/regime_scripts/scrap_dietingban.py']),
        ('zhangtingban', '榜单·涨停板（问财）', [PY, '-u', f'{D}/regime_scripts/scrap_zhangtingban.py']),
        ('longhu', '榜单·龙虎榜（同花顺）', [PY, '-u', f'{D}/regime_scripts/scrap_longhu.py']),
        ('feature', '特征构建 base_feature', [PY, '-u', f'{D}/dataloaders/feature_build/build_all_feature.py', '--daily-source', daily]),
    ]
    info = f'管线={daily} | 库状态: ts={ts_n} 只, m15={m15_n} 只'
    return steps, info


def _run_step(cmd, env, desc, attempts, idx, total):
    """前台跑一步（子进程继承终端，输出原样透出），失败退避重试；返回 (rc, 已试次数, 用时)。"""
    print('=' * 72, flush=True)
    print(f'[{idx}/{total}] {desc}', flush=True)
    print('=' * 72, flush=True)
    t0 = time.time()
    rc, tried = -1, 0
    for a in range(1, attempts + 1):
        tried = a
        if a > 1:
            print(f'[{idx}/{total}] 第 {a}/{attempts} 次尝试…', flush=True)
        rc = subprocess.call(cmd, cwd=ROOT, env=env)
        if rc == 0:
            break
        if a < attempts:
            print(f'[{idx}/{total}] 失败(rc={rc}) → {20 * a}s 后重试', flush=True)
            time.sleep(20 * a)
    dt = time.time() - t0
    mark = '完成' if rc == 0 else f'失败 rc={rc}'
    print(f'[{idx}/{total}] {mark}  用时 {dt:.0f}s', flush=True)
    return rc, tried, dt


def _log_status(line):
    os.makedirs(os.path.dirname(ALERT_F), exist_ok=True)
    with open(ALERT_F, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def main():
    ap = argparse.ArgumentParser(description='依次更新所有依赖数据（空库自动从 2020-01-01 起步）')
    ap.add_argument('--skip', default='',
                    help='跳过步骤，逗号分隔: v2,m15,notrade,cls,em724,kaipanla,dietingban,zhangtingban,longhu,feature')
    ap.add_argument('--retry', type=int, default=2, help='每步失败重试次数（默认 2，即最多 3 次尝试）')
    ap.add_argument('--daily', default='tushare', choices=['tushare', 'mix'],
                    help='日K 来源管线：tushare（默认，收费源）/ mix（fuyao+baostock 免费源）')
    a = ap.parse_args()
    skip = {s for s in a.skip.split(',') if s}
    steps, info = plan_steps(a.daily)
    known = {k for k, _, _ in steps}
    unknown = skip - known
    if unknown:
        print(f'[update_all] 警告: 未知步骤名 {sorted(unknown)}（可用: {sorted(known)}）')
    steps = [s for s in steps if s[0] not in skip]
    env = {**os.environ, 'PYTHONPATH': ROOT + os.pathsep + os.environ.get('PYTHONPATH', ''),
           'ASHARE_DAILY_SOURCE': a.daily}   # 供 feature_calc 选择日K读口（tushare|mix）
    print(f'[update_all] {info}；共 {len(steps)} 步：{[k for k, _, _ in steps]}', flush=True)
    _log_status(f'==== {datetime.now():%F %T} run start（{len(steps)} 步） ====')
    results, t_all = [], time.time()
    for i, (key, name, cmd) in enumerate(steps, 1):
        rc, tried, dt = _run_step(cmd, env, name, max(1, a.retry + 1), i, len(steps))
        results.append((key, name, rc, dt, tried))
        status = 'OK' if rc == 0 else f'FAILED rc={rc} x{tried}'
        _log_status(f'{datetime.now():%F %T} {status:>15}  {key:<14}{dt:>8.0f}s  {name}')
        if rc != 0:
            print(f'[ALERT] 步骤失败: {name}（尝试 {tried} 次仍失败 rc={rc}）', flush=True)
    print('\n===== update_all 汇总 =====')
    for key, name, rc, dt, tried in results:
        mark = '✔' if rc == 0 else '✘'
        tail = '' if rc == 0 else f'  [尝试 {tried} 次]'
        print(f'  {mark} {key:<14}{dt:>8.0f}s  {name}{tail}')
    print(f'  总计 {time.time() - t_all:.0f}s；状态记录: {ALERT_F}')
    failed = [r for r in results if r[2] != 0]
    if failed:
        print(f'\n{"!" * 20} [ALERT] {len(failed)} 个步骤失败: {[r[0] for r in failed]} {"!" * 20}')
        sys.exit(1)


if __name__ == '__main__':
    main()
