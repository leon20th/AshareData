import json
from AshareData.paths import META_DIR
from datetime import datetime, time as dt_time

trade_date_list = json.load(open(f"{META_DIR}/trade_date_list.json", "r"))
trade_date_dict = {}
for idx, _t_date in enumerate(trade_date_list):
    trade_date_dict[_t_date] = idx

def update_a_open():
    # update every year
    import akshare as ak
    # 获取历史交易日历
    trade_date_df = ak.tool_trade_date_hist_sina()
    trade_date_list = trade_date_df["trade_date"].astype(str).tolist()
    trade_date_list = [t.replace("-", "") for t in trade_date_list]
    json.dump(trade_date_list, open(f"{META_DIR}/trade_date_list.json", "w"), ensure_ascii=False)


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