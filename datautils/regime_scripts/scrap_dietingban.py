from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import base64
import re
import os
import pandas as pd
from datetime import datetime
from AshareData.utils.log_util import setup_logging, get_logger
from AshareData.datautils.regime_scripts.scrap import Scrap
from AshareData.utils.tsh_utils.login_util import login_wencai
import json
from openpyxl import Workbook, load_workbook
from io import StringIO
import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from env_setting import ROOT
import traceback
from bs4 import BeautifulSoup
import re

setup_logging()
logger = get_logger('跌停板爬取')

class ScrapDietingban(Scrap):
    def __init__(self, date=''):
        self.url = 'https://www.iwencai.com/unifiedwap/home/index'
        if not date:
            date = datetime.now().strftime('%Y%m%d')
        self.date = date
        self.result_path = f'{ROOT}/AshareData/dataset/scrap_data/dietingban'
        os.makedirs(self.result_path, exist_ok=True)
        self.driver = None

    def get_login_driver(self):
        driver = self.get_driver()
        driver.get(self.url)
        # 等待页面加载
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        # 登录
        ret = login_wencai(driver)
        if not ret:
            driver.quit()
            return None
        return driver

    def get_result(self, query, only_detect=False):
        # 等待下载
        result = ''
        wt = 60 if not only_detect else 3
        for i in range(int(wt // 3)): # 最多等待一分钟
            for dl in os.listdir(self.result_path):
                if query[:20] in dl:
                    result = f'{self.result_path}/{dl}'
                    if os.path.isfile(result):
                        return result
            time.sleep(3)
        if not only_detect:
            logger.error(f'下载数据失败, query={query}')
        else:
            logger.info(f'没有先前下载的数据，query={query}')
        return None

    def parse_headers(self, element):
        """提取表头列名，排除序号和复选框"""
        # 获取元素的HTML
        html = element.get_attribute("outerHTML")

        # 使用BeautifulSoup解析
        soup = BeautifulSoup(html, 'html.parser')

        headers = []
        # 查找所有li元素
        for li in soup.find_all('li'):
            # 获取li的class
            li_class = li.get('class', [])

            # 检查是否是复选框列（通过class判断）
            if 'checkbox-box' in li_class:
                continue

            # 查找cell-box
            cell_box = li.find('div', class_='cell-box')
            if cell_box:
                # 查找所有span
                spans = cell_box.find_all('span')
                if spans:
                    # 第一个span是列名
                    column_name = spans[0].get_text(strip=True)
                    # 排除序号
                    if column_name == '序号':
                        continue
                    # 添加到结果
                    headers.append(column_name)
        return headers


    def get_content_with_selenium(self, query='跌停板', date='', shutdown_driver=True):
        """
        使用Selenium获取动态加载的内容
        """
        if not date:
            date = self.date
        query = f'{date}' + query
        pre_ret = self.get_result(query, only_detect=True)
        if pre_ret:
            return pre_ret
        if self.driver is None:
            self.driver = self.get_login_driver()
        driver = self.driver
        for i in range(3):
            try:
                # 查询跌停板
                
                driver.get(f'https://www.iwencai.com/unifiedwap/result?w={query}')

                # 等待页面加载
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

                # 手动抓取导出
                cc_tables = None
                while True:
                    pager = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, '.pcwencai-pagination-wrap'))
                    )

                    table_content = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, '.iwc-table-scroll'))
                    )
                    tables = pd.read_html(StringIO(table_content.get_attribute('outerHTML')))[0]
                    tables = tables.iloc[:, 2:]

                    fix_table = driver.find_element(By.CSS_SELECTOR, '.iwc-table-fixed')
                    fix_headers = fix_table.find_element(By.CSS_SELECTOR, '.iwc-table-header')
                    sc_headers = table_content.find_element(By.CSS_SELECTOR, '.iwc-table-header-ul.clearfix')
                    headers = self.parse_headers(fix_headers) + self.parse_headers(sc_headers)
                    tables.columns = headers
                    if cc_tables is None:
                        cc_tables = tables
                    else:
                        cc_tables = pd.concat([cc_tables, tables], axis=0, ignore_index=True)
                    has_next_btn = pager.find_elements(By.LINK_TEXT, '下页')
                    if not has_next_btn:
                        break
                    next_btn = pager.find_element(By.LINK_TEXT, '下页')
                    next_btn_pt = next_btn.find_element(By.XPATH, "parent::*")
                    if next_btn_pt.get_attribute('class') == 'disabled':
                        break
                    else:
                        last_text = table_content.find_element(By.CSS_SELECTOR, "table tbody tr:first-child").text
                        next_btn.click()
                        wait = WebDriverWait(table_content, 10)
                        wait.until(
                            lambda table_content: table_content.find_element(By.CSS_SELECTOR, "table tbody tr:first-child").text != last_text
                        )
                cc_tables['股票代码'] = cc_tables['股票代码'].apply(lambda x: '%.6d' % x)
                cc_tables.to_excel(f'{self.result_path}/{query}.xlsx', index=False)

                # 直接点击数据导出
                #table_exp = WebDriverWait(driver, 10).until(
                #    EC.presence_of_element_located((By.CSS_SELECTOR, ".icon.download"))
                #)
                #driver.execute_script("arguments[0].click();", table_exp)
                
                result = self.get_result(query)
                if not result:
                    logger.error(f'下载数据失败, query={query}')
                    continue
                return result
            except Exception as e:
                logger.error(f'获取数据失败, query={query}, error={e}')
                logger.error(traceback.format_exc())
                continue
        if shutdown_driver:
            driver.quit()


if __name__ == '__main__':
    import pandas as pd
    from AshareData.utils.exchanges_utils.a_open import get_trade_date_list, get_target_trade_date

    all_dates = get_trade_date_list()
    end_date = get_target_trade_date()
    sw = ScrapDietingban(date=end_date)
    output = sw.result_path

    had_scrap_date = [f.split('跌停板')[0] for f in os.listdir(output) if f.endswith('.xlsx')]
    last_date = max(had_scrap_date) if had_scrap_date else '20200101'
    last_date_idx = all_dates.index(last_date) if last_date in all_dates else -1
    begin_date = all_dates[last_date_idx-5] if last_date_idx >= 0 else '20200101'

    dates = [d for d in all_dates if d > begin_date and d <= end_date]
    logger.info(f'需要爬取的日期列表: {dates}')
    # 遍历A列
    with tqdm.tqdm(total=len(dates)) as pbar, logging_redirect_tqdm():
        for date in dates:
            sw.get_content_with_selenium(date=date, shutdown_driver=False)
            logger.info(f'完成日期: {date}')
            pbar.update(1)
    if sw.driver:
        sw.driver.quit()