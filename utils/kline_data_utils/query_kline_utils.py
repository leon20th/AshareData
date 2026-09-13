import pandas as pd
import baostock as bs

class QueryKlineUtils:
    def __init__(self):
        self.login = False

    def login_bs(self):
        if not self.login:
            bs.login()
            self.login = True

    def query_history(self, code, start_date='2020-01-01', end_date='2026-01-21', frequency="d", adjustflag="2"):
        self.login_bs()
        fields = ['date', 'code', 'open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'tradestatus', 'pctChg', 'isST']
        numic_fields = ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount', 'turn', 'pctChg', 'isST']
        if frequency in ['5', '15', '30', '60']:
            unsupported_fields = ['preclose', 'turn', 'tradestatus', 'pctChg', 'isST']
            fields = [f for f in fields if f not in unsupported_fields]
            fields.insert(1, 'time')
        try:
            rs = bs.query_history_k_data_plus(code,
                ",".join(fields),
                start_date=start_date, end_date=end_date,
                frequency=frequency, adjustflag=adjustflag)
            data_list = []
            if rs.error_code != '0':
                msg = f"code={code}, 获取失败: {rs.error_msg}, 结束日期={end_date}"
                return None, msg
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            result = pd.DataFrame(data_list, columns=rs.fields)
            # 日期区间
            date_range = f"{result['date'].min()} to {result['date'].max()}" if not result.empty else "no data"
            msg = f"code={code}, 获取成功: {date_range}"
            if frequency in ['5', '15', '30', '60']:
                result['time'] = result['time'].astype(str).str.slice(8, 12)
            for field in numic_fields:
                if field in result.columns:
                    result[field] = pd.to_numeric(result[field], errors='coerce')
            return result, msg
        except Exception as e:
            msg = f'code={code}, 获取失败: {e}, 结束日期={end_date}'
            return None, msg


if __name__ == "__main__":
    import time
    query_utils = QueryKlineUtils()
    code = 'sz.300344'
    start_date = '2026-07-17'
    end_date = '2026-07-17'
    frequency = '15'
    df, msg = query_utils.query_history(code, start_date=start_date, end_date=end_date, frequency=frequency)
    print(msg)
    if df is not None:
        print(df.head())
    
    1 / 0
    st = time.time()
    for i in range(100):
        df1, msg = query_utils.query_history(code, start_date='2026-06-24', end_date='2026-06-24', frequency='d')
        df2, msg = query_utils.query_history(code, start_date='2026-06-24', end_date='2026-06-24', frequency='15')
    total_time = time.time() - st
    print(f"Total time for two queries: {total_time:.2f} seconds")
    print(df1.head())
    print(df2.head())
    