import json
from AshareData.paths import META_DIR
from datetime import datetime, time as dt_time

trade_date_list = json.load(open(f"{META_DIR}/trade_date_list.json", "r"))
trade_date_dict = {}
for idx, _t_date in enumerate(trade_date_list):
    trade_date_dict[_t_date] = idx

def update_a_open():
    """更新交易日历（baostock query_trade_dates：全史 + 已发布计划日期，单次全量重取）。

    覆盖 [1990-01-01, 明年年末]；未发布段自动为空，年末重跑即随年度计划前移。
    """
    import baostock as bs
    lg = bs.login()
    if lg.error_code != '0':
        raise RuntimeError(f'baostock 登录失败: {lg.error_msg}')
    try:
        rs = bs.query_trade_dates(start_date='1990-01-01',
                                  end_date=f'{datetime.now().year + 1}-12-31')
        days = []
        while rs.next():
            row = rs.get_row_data()
            if row[1] == '1':
                days.append(row[0].replace('-', ''))
    finally:
        bs.logout()
    if len(days) < 8000:   # 宁可报错也不覆盖文件
        raise RuntimeError(f'交易日历异常偏少: {len(days)} 条，已中止写入')
    json.dump(days, open(f"{META_DIR}/trade_date_list.json", "w"), ensure_ascii=False)
    ahead = len([d for d in days if d > datetime.now().strftime('%Y%m%d')])
    print(f'交易日历已更新: {len(days)} 条（{days[0]} → {days[-1]}，前瞻 {ahead} 个交易日）')
    if ahead < 15 and datetime.now().month >= 11:
        print('[WARN] 前瞻交易日不足 15 个，下一年度计划可能尚未发布，请稍后重跑')


def is_a_share_open_today(date=None):
    trade_date_list = json.load(open(f"{META_DIR}/trade_date_list.json", "r"))
    if date is None:
        date = datetime.now().strftime("%Y%m%d")
    if date in trade_date_list:
        return True, f"{date} 是A股交易日，A股市场开盘"
    else:
        return False, f"{date} 不是A股交易日，A股市场休市"        

def get_trade_date_list():
    return trade_date_list

def get_trade_date_idx(date):
    date = str(date).strip().replace("-", "")
    if not date.isdigit():
        return -2
    if date not in trade_date_dict:
        return -1
    return trade_date_dict[date]

def get_target_trade_date(end_date=None):
    trade_dates = sorted(get_trade_date_list())
    if not trade_dates:
        return None

    now = datetime.now()
    today_str = now.strftime('%Y%m%d')
    bound = (end_date or now.strftime('%Y-%m-%d')).replace('-', '')

    if bound == today_str and today_str in trade_dates and now.time() < dt_time(15, 0):
        valid_dates = [d for d in trade_dates if d < today_str]
    else:
        valid_dates = [d for d in trade_dates if d <= bound]

    if not valid_dates:
        return None
    return valid_dates[-1]

def get_next_trade_date(start_date=None):
    trade_dates = sorted(get_trade_date_list())
    if not trade_dates:
        return None

    now = datetime.now()
    today_str = now.strftime('%Y%m%d')
    bound = (start_date or now.strftime('%Y-%m-%d')).replace('-', '')

    if bound == today_str and today_str in trade_dates and now.time() < dt_time(15, 0):
        valid_dates = [d for d in trade_dates if d >= today_str]
    else:
        valid_dates = [d for d in trade_dates if d > bound]

    if not valid_dates:
        return None
    return valid_dates[0]

if __name__ == "__main__":
    print(get_trade_date_idx('2026-08-24'))