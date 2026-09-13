
import io
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
import traceback
from PIL import Image
from env_setting import ROOT
from tqdm.contrib.logging import logging_redirect_tqdm

setup_logging()
logger = get_logger('涨停分析')

class ScrapZTFX(Scrap):
    def __init__(self, date='', stocks=[], scrap_continue=True, only_kline=False):
        self.url = 'https://www.iwencai.com/unifiedwap/home/index'
        if not date:
            date = datetime.now().strftime('%Y%m%d')
        self.date = date
        self.stocks = stocks
        self.scrap_continue = scrap_continue
        self.only_kline = only_kline
        self.kline_root = f'{ROOT}/AshareData/dataset/scrap_data/wencai/stock_daily_kline'
        self.detail_root = f'{ROOT}/AshareData/dataset/scrap_data/wencai/details'

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

    def get_kline(self, driver, url, code, kline_path):
        for i in range(3):
            try:
                driver.get(url)
                kline2 = WebDriverWait(driver, 10).until(
                    EC.any_of(
                        EC.presence_of_element_located((By.CLASS_NAME, "jgy_kline2_page")),
                        EC.presence_of_element_located((By.CLASS_NAME, "kline2"))
                    )
                )
                break
            except:
                if i == 2:
                    logger.error(f'Get intro info fail, code={code}')
                    return None
        k_tab_items = WebDriverWait(kline2, 10).until(
            EC.any_of(
                EC.presence_of_element_located((By.CLASS_NAME, "jgy_kline_tab")),
                EC.presence_of_element_located((By.CLASS_NAME, "kline2_tab"))
            )
        )
        k_tab_items = kline2.find_elements(By.CSS_SELECTOR, ".select_box, .kline2_tab_item, .kline2_tab_item.kline2_select_item")
        for k_t in k_tab_items:
            driver.execute_script("arguments[0].click();", k_t)
            kline_chart = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "kline_chart"))
            )
            time.sleep(1)
            # 截图
            mw = 720
            kline2_screenshot_bytes = kline2.screenshot_as_png
            image = Image.open(io.BytesIO(kline2_screenshot_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            iw, ih = image.size
            if iw > mw:
                image = image.crop((0, 0, mw, ih))
            image.save(f'{kline_path}/{code}_{k_t.text}.png')

    def get_content_with_selenium(self):
        """
        使用Selenium获取动态加载的内容
        """
        driver = self.get_login_driver()
        # 创建当天文件夹
        # k线
        kline_path = f'{self.kline_root}/{self.date}'
        # detail
        detail_path = f'{self.detail_root}/{self.date}'
        os.makedirs(kline_path, exist_ok=True)
        os.makedirs(detail_path, exist_ok=True)
        cnt = 0
        with tqdm.tqdm(total=len(self.stocks)) as pbar, logging_redirect_tqdm():
            for code in self.stocks:
                try:
                    info_result = f'{detail_path}/{code}_detail.json'
                    if os.path.isfile(info_result) and self.scrap_continue:
                        logger.info(f'skip: code={code}')
                        continue
                    logger.info(f'processing: code={code}')
                    url = f'https://www.iwencai.com/unifiedwap/result?w={code}&querytype=stock'

                    self.get_kline(driver, url, code, kline_path)
                    logger.info('kline done.')

                    if self.only_kline:
                        continue

                    # 简介和看点
                    intros = []
                    cont_ele_d = WebDriverWait(driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, ".jgy_item_box.container_outer"))
                    )
                    cont_eles = driver.find_elements(By.CSS_SELECTOR, ".jgy_item_box.container_outer")
                    for cont_ele in cont_eles:
                        try:
                            if driver.find_elements(By.CLASS_NAME, 'title-text'):
                                title_text = driver.find_element(By.CLASS_NAME, 'title-text').text
                                if '简介和看点' in title_text:
                                    conts = cont_ele.find_elements(By.CLASS_NAME, 'container_outer')
                                    for cont in conts:
                                        intros.append(cont.text)
                                    break
                        except Exception as e:
                            logger.error(f'find introduce error: {traceback.format_exc()}')
                    logger.info('intro done.')
                    # 所属概念
                    cc_table = None
                    try:
                        WebDriverWait(driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, ".jgy_item_box.tableV1_outer"))
                        )
                    except Exception as e:
                        logger.error('no concept....')
                    table_eles = driver.find_elements(By.CSS_SELECTOR, '.jgy_item_box.tableV1_outer')
                    for table_ele in table_eles:
                        try:
                            if not driver.find_elements(By.CLASS_NAME, 'title-text'):
                                continue
                            title_text = table_ele.find_element(By.CLASS_NAME, 'title-text').text
                            if '所属概念' in title_text:
                                all_page_items = table_ele.find_elements(By.CSS_SELECTOR, "li.page_item")[1:]
                                idx = 0
                                while True:
                                    table_content = WebDriverWait(table_ele, 10).until(
                                        EC.presence_of_element_located((By.CSS_SELECTOR, '.jgy_tb_table'))
                                    )
                                    tables = pd.read_html(StringIO(table_ele.get_attribute('outerHTML')))
                                    table = pd.concat(tables, axis=1)
                                    if cc_table is None:
                                        cc_table = table
                                    else:
                                        cc_table = pd.concat([cc_table, table], axis=0, ignore_index=True)
                                    idx += 1
                                    if idx < len(all_page_items):
                                        pi = all_page_items[idx]
                                        driver.execute_script("arguments[0].click();", pi)
                                        time.sleep(1)
                                    else:
                                        break
                        except Exception as e:
                            logger.error(f'Got stock concept error: {traceback.format_exc()}')
                    logger.info('concept done.')
                    # 涨停分析
                    url = f'https://www.iwencai.com/unifiedwap/result?w={code}涨停分析&querytype=stock'
                    for i in range(3):
                        try:
                            driver.get(url)
                            zt_ele_d = WebDriverWait(driver, 10).until(
                                EC.presence_of_element_located((By.CSS_SELECTOR, ".jgy_item_box.all-components-box"))
                            )
                            break
                        except:
                            if i == 2:
                                logger.error(f'Get zt info fail, code={code}')
                    zt_eles = driver.find_elements(By.CSS_SELECTOR, ".jgy_item_box.all-components-box")
                    zt_reason = ''
                    for idx, zt_ele in enumerate(zt_eles):
                        try:
                            if driver.find_elements(By.CLASS_NAME, 'title-text'):
                                title_text = driver.find_element(By.CLASS_NAME, 'title-text').text
                                if '涨停揭秘' == title_text and \
                                        zt_ele.find_elements(By.CSS_SELECTOR, "[class='news_list1_slider_box news_list1_slider_box_1']"):
                                    zt_reason = zt_ele.find_element(By.CSS_SELECTOR, "[class='news_list1_slider_box news_list1_slider_box_1']").text
                                    break
                        except Exception as e:
                            logger.error(f'find zt reason error: {e}')
                    logger.info('zhangting reason done.')
                    # 涨停封单占成交、连板次数、涨停成交额 TODO
                    # url = f'https://www.iwencai.com/unifiedwap/result?w={code}涨停封单占成交、连板次数、涨停成交额&querytype=stock'

                    concept = cc_table.to_dict() if cc_table is not None else {}
                    logger.info(f'zhangtingfenxi, stock={code}, intro={len(intros)}, zt_reason={len(zt_reason)}, concept={len(concept)}')
                    info = {
                        'intros': intros,
                        'zt_reason': zt_reason,
                        'concept': concept
                    }
                    info_result = f'{detail_path}/{code}_detail.json'
                    json.dump(info, open(info_result, 'w'), indent=4, ensure_ascii=False)
                    logger.info(f'scrap done, code={code}')
                    cnt += 1
                except Exception as e:
                    logger.error(f"Error during scraping code={code} detail: {traceback.format_exc()}")
                finally:
                    pbar.update(1)
        driver.quit()
        return cnt

def get_zhangting_stocks(date=None, result=''):
    codes = []
    if not date:
        date = datetime.now().strftime('%Y%m%d')
    query = f'{date}涨停板'
    if not result:
        result = f'{ROOT}/business/zhangting_analysis_report/scrap_data/zhangtingban/{query}.xlsx'
    if not os.path.isfile(result):
        logger.error('下载涨停板数据失败')
    else:
        logger.info(f'读取涨停数据: {result}')
        # 查询每个股票的详细信息
        wb_read = load_workbook(result)
        ws_read = wb_read.active
        headers = [cell for cell in ws_read[1]]
        code_col = ''
        for h in headers:
            if h.value and '股票简称' in h.value:
                code_col = h.column_letter
                break
        for cell in ws_read[code_col]:
            if cell.row == 1:
                continue
            if not cell.value:
                continue
            codes.append(cell.value)
        logger.info(f'涨停板数量：{len(codes)}')
    return codes

def get_unusual_active_stock(date=None):
    if not date:
        date = datetime.now().strftime('%Y%m%d')
    date_str = date[4:6] + '月' + date[6:8] + '日'
    path = f'{ROOT}/AshareData/dataset/scrap_data/jiucai/articles/{date}_jiucai_articles.json'
    if not os.path.isfile(path):
        logger.info('异动分析文档不存在')
    articles = []
    with open(path, 'r') as f:
        for line in f:
            articles.append(json.loads(line))
    stocks = []
    for atc in articles:
        title = atc['title']
        if '股票异动解析' not in title:
            logger.info(f'跳过文章: {title}')
            continue
        if date_str not in title or not title.startswith(date_str):
            logger.info(f'跳过无日期文章: {title}')
            continue
        stock = title[6:-6]
        stocks.append(stock)
    return stocks

if __name__ == '__main__':
    date = '20260303'
    zt_stocks = set(get_zhangting_stocks(date))
    at_stocks = set(get_unusual_active_stock(date))

    all_stocks = list(zt_stocks.union(at_stocks))
    logger.info(f'涨停数量: {len(zt_stocks)}, 异动数量: {len(at_stocks)}, 合并数量: {len(all_stocks)}')
    sw = ScrapZTFX(date=date, stocks=all_stocks, scrap_continue=False)
    sw.get_content_with_selenium()