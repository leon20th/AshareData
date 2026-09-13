import os
import sys
import argparse
import json
import pandas as pd
import tqdm

from AshareData.paths import DAILY_KLINE_DIR, M15_KLINE_DIR
from AshareData.utils.log_util import get_logger
from tqdm.contrib.logging import logging_redirect_tqdm
from AshareData.utils.exchanges_utils.a_open import get_target_trade_date
from AshareData.utils.read_file_utils import get_first_last_line_from_csv
from AshareData.utils.kline_data_utils.query_kline_utils import QueryKlineUtils
from AshareData.datautils.kline_scripts.update_notrade import UpdateNotrade

logger = get_logger('更新k线数据')

class UpdateKline:
    def __init__(self):
        self.daily_database = DAILY_KLINE_DIR
        self.m15_database = M15_KLINE_DIR
        self.frequency_map = {
            'd': self.daily_database,
            '15': self.m15_database,
        }
        for freq, path in self.frequency_map.items():
            os.makedirs(path, exist_ok=True)
        logger.info(f'日线数据目录: {self.daily_database}')
        logger.info(f'15分钟线数据目录: {self.m15_database}')
        self.query_utils = QueryKlineUtils()
        self.update_notrade = UpdateNotrade(force_init=True)
    
    # 析构函数 bs.logout()
    def __del__(self):
        pass

    @staticmethod
    def _format_trade_date(trade_date):
        if not trade_date:
            return None
        try:
            return pd.to_datetime(trade_date).strftime('%Y-%m-%d')
        except Exception:
            return None

    @staticmethod
    def _get_latest_date_str(base_last_date, df_new):
        latest_date = base_last_date
        if df_new is not None and not df_new.empty and 'date' in df_new.columns:
            max_date = pd.to_datetime(df_new['date'], errors='coerce').max()
            if pd.notna(max_date):
                latest_date = max_date
        return latest_date.strftime('%Y%m%d') if pd.notna(latest_date) else None

    @staticmethod
    def _is_reached_date(latest_date_str, target_date_str):
        if not latest_date_str or not target_date_str:
            return False
        latest_date = pd.to_datetime(latest_date_str, format='%Y%m%d', errors='coerce')
        target_date = pd.to_datetime(target_date_str, format='%Y%m%d', errors='coerce')
        if pd.isna(latest_date) or pd.isna(target_date):
            return False
        return latest_date >= target_date

    def update_kline_daily(self, end_date=None, codes=None, frequencies=['d', '15']):
        codes = codes or []
        codes = set(map(str, codes))
        end_date = pd.to_datetime('today').strftime('%Y-%m-%d') if end_date is None else end_date
        target_trade_date = get_target_trade_date(end_date=end_date)
        query_end_date = self._format_trade_date(target_trade_date) or end_date
        validate_target_date = self._format_trade_date(query_end_date)
        validate_target_date_str = pd.to_datetime(validate_target_date).strftime('%Y%m%d') if validate_target_date else None
        # 获取复权行情数据：adjustflag为3表示后复权, 2表示前复权
        all_files = [f for f in os.listdir(self.daily_database) if f.endswith('.csv') and (not codes or f.rsplit(".", 1)[0] in codes)]
        results = {
            'target_trade_date': target_trade_date,
            'validate_target_date': validate_target_date_str,
            'all_latest': True,
            'details': []
        }

        with tqdm.tqdm(total=len(all_files)) as pbar, logging_redirect_tqdm():
            for f in all_files:
                code = f.rsplit('.', 1)[0]
                for frequency in frequencies:
                    code_result = {
                        'code': code,
                        'frequency': frequency,
                        'query_ok': True,
                        'write_ok': True,
                        'update_status': True,
                        'is_latest': True,
                        'last_date': None,
                    }
                    logger.info(f'开始更新 code={code} frequency={frequency} 的数据')
                    file_path = f'{self.frequency_map[frequency]}/{f}'
                    default_last_date = pd.to_datetime('2019-12-31')
                    last_df, last_msg = get_first_last_line_from_csv(file_path)
                    if last_msg:
                        logger.info(f'code={code}, 读取本地文件失败: {last_msg}, 结束日期={end_date}')
                        last_date = default_last_date
                    elif last_df is None or last_df.empty or 'date' not in last_df.columns:
                        last_date = default_last_date
                    else:
                        last_date = pd.to_datetime(last_df['date'].iloc[-1], errors='coerce')
                        if pd.isna(last_date):
                            logger.info(f'code={code}, 本地最后日期解析失败, 结束日期={end_date}')
                            last_date = default_last_date
                    
                    start_date = (last_date + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
                    if pd.to_datetime(start_date) > pd.to_datetime(query_end_date):
                        logger.info(f'code={code}, 无需更新, 数据最新日期={last_date}, 查询截止={query_end_date}')
                        df_new = None
                    else:
                        df_new, msg = self.query_utils.query_history(code, start_date=start_date, end_date=query_end_date, frequency=frequency)
                        logger.info(msg)
                        code_result['query_ok'] = df_new is not None
                        if df_new is not None and not df_new.empty:
                            if 'date' in df_new.columns and 'time' in df_new.columns:
                                df_new = df_new.drop_duplicates(subset=['date', 'time'], keep='last').reset_index(drop=True)
                            elif 'date' in df_new.columns:
                                df_new = df_new.drop_duplicates(subset=['date'], keep='last').reset_index(drop=True)
                            file_exists = os.path.exists(file_path) and os.path.getsize(file_path) > 0
                            try:
                                df_new.to_csv(file_path, mode='a', index=False, header=not file_exists)
                                code_result['write_ok'] = True
                                logger.info(f'code={code}, 更新成功, 新增记录数={len(df_new)}, 结束日期={end_date}')
                            except Exception as e:
                                code_result['write_ok'] = False
                                logger.info(f'code={code}, 写入失败: {e}, 结束日期={end_date}')
                        else:
                            logger.info(f'code={code}, 无新增数据, 结束日期={end_date}')

                        # Daily bars include tradestatus; update suspension state in memory.
                        if frequency == 'd' and df_new is not None:
                            try:
                                query_range_days = max(
                                    0,
                                    (pd.to_datetime(query_end_date) - pd.to_datetime(start_date)).days + 1,
                                )
                                if df_new.empty:
                                    self.update_notrade.set_notrade_yet(
                                        df=df_new,
                                        code=code,
                                        notrade_date=start_date,
                                        query_range_days=query_range_days,
                                    )
                                else:
                                    self.update_notrade.set_notrade_yet(
                                        df=df_new,
                                        query_range_days=query_range_days,
                                    )
                            except Exception as e:
                                logger.info(f'code={code}, 更新停牌状态失败: {e}, 结束日期={end_date}')

                    latest_date_str = self._get_latest_date_str(last_date, df_new)
                    code_result['last_date'] = latest_date_str
                    code_result['update_status'] = bool(code_result['query_ok'] and code_result['write_ok'])

                    if validate_target_date_str:
                        code_result['is_latest'] = bool(
                            code_result['update_status'] and
                            self._is_reached_date(latest_date_str, validate_target_date_str)
                        )
                    else:
                        code_result['is_latest'] = bool(code_result['update_status'])

                    results['details'].append(code_result)
                    if not code_result['is_latest']:
                        results['all_latest'] = False

                pbar.update(1)

        failed_updates = [x for x in results['details'] if not x['update_status']]
        not_latest = [x for x in results['details'] if not x['is_latest']]

        notrade_df = self.update_notrade.get_notrade_yet()
        notrade_codes = set()
        if notrade_df is not None and not notrade_df.empty and 'code' in notrade_df.columns:
            notrade_codes = set(notrade_df['code'].astype(str).tolist())

        not_latest_non_notrade = [x for x in not_latest if x.get('code') not in notrade_codes]
        logger.info(
            f"更新校验完成: target_trade_date={target_trade_date}, "
            f"validate_target_date={validate_target_date_str}, 未达指定日期数量={len(not_latest)}, "
            f"未达指定日期数量且未停牌数量={len(not_latest_non_notrade)}, 更新失败数量={len(failed_updates)}"
        )

        if len(not_latest_non_notrade) > 0:
            preview = []
            for item in not_latest_non_notrade[:30]:
                preview.append(
                    f"{item.get('code')}(freq={item.get('frequency')},last={item.get('last_date')})"
                )
            logger.info(f"未达且未停牌股票样例(最多30个): {', '.join(preview)}")

        # Persist notrade table once after all per-code updates.
        try:
            self.update_notrade.flush_notrade_yet()
        except Exception as e:
            logger.info(f'落盘停牌状态失败: {e}')

        results['not_latest_non_notrade'] = list(set(x['code'] for x in not_latest_non_notrade))

        return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='更新日线K线数据')
    parser.add_argument('--end-date', default=None, help='更新截止日期，格式 YYYY-MM-DD')
    parser.add_argument('--codes', default='', help='逗号分隔股票代码，例如 000001,600519')
    parser.add_argument('--result-file', default='', help='可选：写入结果json文件路径')
    args = parser.parse_args()

    codes = [c.strip() for c in args.codes.split(',') if c.strip()]
    updater = UpdateKline()
    result = updater.update_kline_daily(end_date=args.end_date, codes=codes)
    failed_codes = sorted({x['code'] for x in result.get('details', []) if not x.get('is_latest', False)})
    if args.result_file:
        try:
            with open(args.result_file, 'w', encoding='utf-8') as fp:
                json.dump(
                    {
                        'all_latest': bool(result.get('all_latest', False)),
                        'failed_codes': failed_codes,
                    },
                    fp,
                    ensure_ascii=False,
                )
        except Exception as e:
            logger.info(f'写入结果文件失败: {args.result_file}, err={e}')

    sys.exit(1 if result['not_latest_non_notrade'] else 0)