"""定点重抓损坏/不全的跌停板文件"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))

from AshareData.datautils.regime_scripts.scrap_dietingban import ScrapDietingban
from AshareData.utils.log_util import setup_logging, get_logger

setup_logging()
logger = get_logger('跌停板重抓')

DATES_TO_RESCRAPE = ['20201030', '20250124', '20250311', '20250312', '20250616']

if __name__ == '__main__':
    sw = ScrapDietingban(date=DATES_TO_RESCRAPE[-1])
    for date in DATES_TO_RESCRAPE:
        logger.info(f'重抓日期: {date}')
        try:
            result = sw.get_content_with_selenium(date=date, shutdown_driver=False)
            if result:
                logger.info(f'✅ {date} 完成 -> {result}')
            else:
                logger.error(f'❌ {date} 失败')
        except Exception as e:
            logger.error(f'❌ {date} 异常: {e}')
    if sw.driver:
        sw.driver.quit()
    logger.info('全部完成')
