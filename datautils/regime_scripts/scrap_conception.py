"""逐只股票爬取所属概念板块，覆盖更新 conception.json。

用法:
    python scrap_conception.py

流程与 scrap_zhangtingban.py 完全一致：
  get_all_codes → 逐只查询 '{code}概念' → Selenium 翻页 → 覆盖写入 json
产物 7 天内有效，超过 7 天自动重新爬取。

输出格式:
  {
    "601869": [
      {"concept_name": "光纤概念", "scrape_date": "20260806", "inclusion_date": "20240101"},
      ...
    ],
    ...
  }
"""

from __future__ import annotations

import json
import os
import traceback
from datetime import datetime, timedelta

import tqdm
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm.contrib.logging import logging_redirect_tqdm

from env_setting import ROOT
from AshareData.datautils.regime_scripts.scrap import Scrap
from AshareData.utils.exchanges_utils.stock_utils import get_all_codes
from AshareData.utils.log_util import get_logger, setup_logging
from AshareData.utils.tsh_utils.login_util import login_wencai

setup_logging()
logger = get_logger('概念爬取')

RESULT_PATH = f'{ROOT}/AshareData/dataset/scrap_data/conception'
OUTPUT_FILE = f'{RESULT_PATH}/conception.json'
MAX_AGE_DAYS = 7


class ScrapConception(Scrap):

    def __init__(self):
        self.url = 'https://www.iwencai.com/unifiedwap/home/index'
        os.makedirs(RESULT_PATH, exist_ok=True)
        self.driver = None

    # ------------------------------------------------------------------ #
    #  Cache: 7 天内有效
    # ------------------------------------------------------------------ #

    def get_result(self):
        """如果产物在 7 天内，直接返回路径；否则返回 None。"""
        if not os.path.isfile(OUTPUT_FILE):
            return None
        mtime = datetime.fromtimestamp(os.path.getmtime(OUTPUT_FILE))
        if datetime.now() - mtime < timedelta(days=MAX_AGE_DAYS):
            logger.info(f'产物仍在有效期内({MAX_AGE_DAYS}天)，跳过爬取: {OUTPUT_FILE}')
            return OUTPUT_FILE
        logger.info(f'产物已过期(超过{MAX_AGE_DAYS}天)，重新爬取')
        return None

    # ------------------------------------------------------------------ #
    #  Login (same as ScrapZhangtingban)
    # ------------------------------------------------------------------ #

    def get_login_driver(self):
        driver = self.get_driver()
        driver.get(self.url)
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        ret = login_wencai(driver)
        if not ret:
            driver.quit()
            return None
        return driver

    # ------------------------------------------------------------------ #
    #  Helpers (same as ScrapZhangtingban)
    # ------------------------------------------------------------------ #

    def parse_headers(self, element):
        """提取表头列名，排除序号和复选框。"""
        html = element.get_attribute('outerHTML')
        soup = BeautifulSoup(html, 'html.parser')
        headers = []
        for li in soup.find_all('li'):
            li_class = li.get('class', [])
            if 'checkbox-box' in li_class:
                continue
            cell_box = li.find('div', class_='cell-box')
            if cell_box:
                spans = cell_box.find_all('span')
                if spans:
                    column_name = spans[0].get_text(strip=True)
                    if column_name == '序号':
                        continue
                    headers.append(column_name)
        return headers

    def _find_next_page_button(self, pager):
        """兼容不同分页样式，返回可点击的下一页按钮。"""
        next_xpath = (
            ".//*[self::a or self::button or self::span]"
            "[contains(normalize-space(.), '下页') or "
            "contains(normalize-space(.), '下一页') or "
            "contains(normalize-space(.), '下 一 页') or "
            "normalize-space(.)='>']"
        )
        candidates = pager.find_elements(By.XPATH, next_xpath)
        for ele in candidates:
            cls = (ele.get_attribute('class') or '').lower()
            parent_cls = (ele.find_element(By.XPATH, 'parent::*').get_attribute('class') or '').lower()
            if 'disabled' in cls or 'disabled' in parent_cls:
                continue
            if ele.is_displayed() and ele.is_enabled():
                return ele

        css_candidates = [
            '.pagination-next:not(.disabled)',
            '.next:not(.disabled)',
            '.btn-next:not(.disabled)',
            'li.next:not(.disabled) a',
        ]
        for selector in css_candidates:
            elements = pager.find_elements(By.CSS_SELECTOR, selector)
            for ele in elements:
                if ele.is_displayed() and ele.is_enabled():
                    return ele
        return None

    # ------------------------------------------------------------------ #
    #  单只股票概念抓取
    # ------------------------------------------------------------------ #

    def _scrape_one(self, driver, code: str) -> list[dict]:
        """查询单只股票的概念列表。

        iwencai 概念页面布局是两个并排 <table>：
          左表: 概念名称
          右表: 概念解析 | 纳入日期
        按行索引配对提取。
        """
        bare_code = code.split('.')[-1] if '.' in code else code
        query = f'{bare_code}概念'
        scrape_date = datetime.now().strftime('%Y%m%d')
        concepts: list[dict] = []

        for attempt in range(3):
            try:
                driver.get(f'https://www.iwencai.com/unifiedwap/result?w={query}')

                # 主动等待：概念名称表格出现且有数据行（tr 数量 > 1，说明表头+数据已渲染）
                WebDriverWait(driver, 30).until(
                    lambda d: len(d.find_elements(By.CSS_SELECTOR, 'table tr')) > 1
                )

                tables = driver.find_elements(By.CSS_SELECTOR, 'table')
                if len(tables) < 2:
                    logger.warning(f'{code}: 页面表格不足2个({len(tables)})，跳过')
                    return concepts

                # 左表: 概念名称（tr 直接在 table 下，第一行是表头，跳过）
                name_rows = tables[0].find_elements(By.CSS_SELECTOR, 'tr')[1:]
                # 右表: 概念解析 + 纳入日期
                detail_rows = tables[1].find_elements(By.CSS_SELECTOR, 'tr')[1:]

                for i, name_row in enumerate(name_rows):
                    name_cells = name_row.find_elements(By.TAG_NAME, 'td')
                    if not name_cells:
                        continue
                    concept_name = name_cells[-1].text.strip()
                    if not concept_name:
                        continue

                    inclusion_date = ''
                    description = ''
                    if i < len(detail_rows):
                        detail_cells = detail_rows[i].find_elements(By.TAG_NAME, 'td')
                        # 右表结构: [概念解析, 纳入日期, 空列]
                        if len(detail_cells) >= 2:
                            description = detail_cells[0].text.strip()
                            inclusion_date = detail_cells[1].text.strip()

                    concepts.append({
                        'concept_name': concept_name,
                        'description': description,
                        'scrape_date': scrape_date,
                        'inclusion_date': inclusion_date,
                    })

                return concepts

            except Exception as e:
                if attempt < 2:
                    logger.warning(f'获取 {code} 概念失败(第{attempt+1}次), 重试: {type(e).__name__}: {e}')
                else:
                    logger.error(f'获取 {code} 概念失败(3次): {type(e).__name__}: {e}')
                    logger.error(traceback.format_exc())

        return concepts

    # ------------------------------------------------------------------ #
    #  主流程
    # ------------------------------------------------------------------ #

    def run(self):
        """爬取全A股所属概念，覆盖写入 conception.json。产物 7 天内有效。"""
        cached = self.get_result()
        if cached:
            return cached

        codes = get_all_codes(bare=False)
        #codes = ['sh.601229', 'sh.603016', 'sz.000001']
        logger.info(f'共 {len(codes)} 只股票，开始逐只爬取概念...')

        if self.driver is None:
            self.driver = self.get_login_driver()
        driver = self.driver

        result: dict[str, list[dict]] = {}
        failed: list[str] = []

        with logging_redirect_tqdm():
            for code in tqdm.tqdm(codes, desc='爬取概念', dynamic_ncols=True):
                concepts = self._scrape_one(driver, code)
                if concepts:
                    result[code] = concepts
                else:
                    failed.append(code)

        if failed:
            logger.warning(f'{len(failed)} 只股票获取失败: {failed[:20]}...')

        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f'覆盖保存完成: {OUTPUT_FILE}, 共 {len(result)} 只股票')
        return OUTPUT_FILE

    def close(self):
        if self.driver:
            self.driver.quit()
            self.driver = None


if __name__ == '__main__':
    sc = ScrapConception()
    try:
        sc.run()
    finally:
        sc.close()
