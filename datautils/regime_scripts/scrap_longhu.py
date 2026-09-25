import json
import os
import re
import traceback
from datetime import datetime

import requests
import tqdm
from bs4 import BeautifulSoup
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm.contrib.logging import logging_redirect_tqdm

from AshareData.paths import SCRAP_DATA_DIR
from AshareData.utils.exchanges_utils.a_open import get_target_trade_date
from AshareData.datautils.regime_scripts.scrap import Scrap
from AshareData.utils.log_util import get_logger, setup_logging
from AshareData.utils.tsh_utils.login_util import login_tsh, get_tsh_cookies_static

setup_logging()
logger = get_logger('龙虎榜爬取')


class ScrapLonghu(Scrap):
    def __init__(self):
        self.url = 'https://data.10jqka.com.cn/market/longhu/'
        self.login_url = 'https://t.10jqka.com.cn/circle/211687/'
        self.ajax_url = 'https://data.10jqka.com.cn/ifmarket/lhbggxq/report/{date}/'
        self.result_path = f'{SCRAP_DATA_DIR}/longhu'
        os.makedirs(self.result_path, exist_ok=True)

    def normalize_date(self, date):
        if isinstance(date, datetime):
            return date.strftime('%Y-%m-%d')
        date = str(date).strip()
        if re.fullmatch(r'\d{8}', date):
            return datetime.strptime(date, '%Y%m%d').strftime('%Y-%m-%d')
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', date):
            return date
        raise ValueError(f'Unsupported date format: {date}')

    def get_save_path(self, date):
        target_date = self.normalize_date(date)
        return f'{self.result_path}/{target_date}.json'

    def normalize_trade_date(self, date):
        return self.normalize_date(date)

    def get_saved_dates(self):
        saved_dates = []
        for filename in os.listdir(self.result_path):
            if not filename.endswith('.json'):
                continue
            date_str = filename[:-5]
            try:
                saved_dates.append(self.normalize_date(date_str))
            except ValueError:
                continue
        return sorted(set(saved_dates))

    def get_auto_scrap_dates(self, all_dates, end_date, earliest_date='2020-01-02'):
        earliest_trade_date = self.normalize_trade_date(earliest_date)
        end_trade_date = self.normalize_trade_date(end_date)
        saved_dates = set(self.get_saved_dates())

        pending_dates = []
        for trade_date in map(self.normalize_trade_date, all_dates):
            if trade_date < earliest_trade_date or trade_date > end_trade_date:
                continue
            if trade_date in saved_dates:
                continue
            pending_dates.append(trade_date)
        return pending_dates

    def build_session(self, date=None):
        target_date = self.normalize_date(date) if date else datetime.now().strftime('%Y-%m-%d')
        anonymous = self.create_session_from_cookies([])
        if self.is_session_valid(anonymous, target_date):
            logger.info('匿名会话可直接抓取龙虎榜')
            return anonymous
        cookie_list = get_tsh_cookies_static()
        if cookie_list:
            session = self.create_session_from_cookies(cookie_list)
            if self.is_session_valid(session, target_date):
                logger.info('使用本地 tsh cookies 直接抓取龙虎榜')
                return session
            logger.info('本地 tsh cookies 已失效，切换到 login_tsh 刷新 cookies')

        driver = self.get_driver(headless=True)
        try:
            driver.get(self.login_url)
            wait = WebDriverWait(driver, 15)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

            ret = login_tsh(driver)
            if not ret:
                raise RuntimeError('login_tsh returned False')

            driver.get(self.url)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
            wait.until(lambda d: '个股明细' in d.page_source)
            return self.create_session_from_cookies(driver.get_cookies())
        finally:
            driver.quit()

    def create_session_from_cookies(self, cookie_list):
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                          '(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36',
            'Referer': self.url,
            'X-Requested-With': 'XMLHttpRequest',
        })
        for cookie in cookie_list or []:
            session.cookies.set(
                cookie['name'],
                cookie['value'],
                domain=cookie.get('domain'),
                path=cookie.get('path', '/'),
            )
        return session

    def is_session_valid(self, session, date=None):
        target_date = self.normalize_date(date) if date else datetime.now().strftime('%Y-%m-%d')
        try:
            response = session.get(self.ajax_url.format(date=target_date), timeout=15)
            response.raise_for_status()
            return 'stockcont' in response.text and 'm-table' in response.text
        except Exception as exc:
            logger.info(f'静态 cookies 校验失败: {exc}')
            return False

    def fetch_html(self, session, date):
        target_date = self.normalize_date(date)
        response = session.get(self.ajax_url.format(date=target_date), timeout=30)
        response.raise_for_status()
        if 'stockcont' not in response.text or 'm-table' not in response.text:
            raise ValueError(f'Unexpected longhu response for {target_date}')
        return response.text

    def normalize_amount_text(self, text):
        return re.sub(r'\s+', '', text or '')

    def parse_amount_summary(self, detail_block):
        summary_node = detail_block.select_one('.cell-cont.cjmx > p')
        if summary_node is None:
            return {}
        detail_link = summary_node.select_one('a')
        summary_text = summary_node.get_text(' ', strip=True)
        if summary_text.startswith('详细'):
            summary_text = summary_text[2:].strip()
        match = re.search(
            r'成交额：\s*(?P<turnover>.+?)\s+合计买入：\s*(?P<total_buy>.+?)\s+'
            r'合计卖出：\s*(?P<total_sell>.+?)\s+净额：\s*(?P<net_amount>.+)$',
            summary_text,
        )
        result = {
            'detail_url': detail_link.get('href') if detail_link else '',
            'summary_text': re.sub(r'\s+', ' ', summary_text).strip(),
        }
        if match:
            result.update({
                'turnover_amount': self.normalize_amount_text(match.group('turnover')),
                'total_buy_amount': self.normalize_amount_text(match.group('total_buy')),
                'total_sell_amount': self.normalize_amount_text(match.group('total_sell')),
                'net_amount': self.normalize_amount_text(match.group('net_amount')),
            })
        return result

    def parse_department_row(self, tr):
        cells = tr.find_all('td', recursive=False)
        if len(cells) < 4:
            return None
        name_cell = cells[0]
        anchor = name_cell.select_one('a')
        tags = [label.get_text(strip=True) for label in name_cell.select('label') if label.get_text(strip=True)]
        name = ''
        if anchor is not None:
            name = anchor.get('title') or anchor.get_text(strip=True)
        if not name:
            name = name_cell.get_text(' ', strip=True)
        return {
            'name': name,
            'tags': tags,
            'url': anchor.get('href') if anchor else '',
            'buy_amount_wan': cells[1].get_text(strip=True),
            'sell_amount_wan': cells[2].get_text(strip=True),
            'net_amount_wan': cells[3].get_text(strip=True),
        }

    def parse_department_table(self, table):
        title_cell = table.select_one('thead th')
        rows = []
        for tr in table.select('tbody tr'):
            item = self.parse_department_row(tr)
            if item is not None:
                rows.append(item)
        return {
            'title': title_cell.get_text(strip=True) if title_cell else '',
            'rows': rows,
        }

    def parse_stock_detail(self, detail_block):
        rid = detail_block.get('rid', '')
        stock_code = detail_block.get('stockcode', '')
        title_node = detail_block.select_one('p')
        title_text = title_node.get_text(' ', strip=True) if title_node else ''
        title_match = re.match(r'(?P<name>.+?)\((?P<code>\d+)\)明细：(?P<reason>.+)', title_text)
        summary = self.parse_amount_summary(detail_block)
        tables = detail_block.select('table.m-table')
        buy_top5 = self.parse_department_table(tables[0]) if len(tables) > 0 else {'title': '', 'rows': []}
        sell_top5 = self.parse_department_table(tables[1]) if len(tables) > 1 else {'title': '', 'rows': []}
        result = {
            'rid': rid,
            'stock_code': stock_code,
            'stock_name': title_match.group('name') if title_match else '',
            'reason': title_match.group('reason') if title_match else title_text,
            'title': title_text,
            'buy_top5_departments': buy_top5,
            'sell_top5_departments': sell_top5,
        }
        result.update(summary)
        return result

    def parse_stock_row(self, tr, detail_map):
        cells = tr.find_all('td', recursive=False)
        if len(cells) < 7:
            return None
        stock_anchor = cells[2].select_one('a.stock')
        if stock_anchor is None:
            return None
        rid = stock_anchor.get('rid', '')
        code = cells[1].get_text(strip=True)
        if code.isdigit():
            code = code.zfill(6)
        return {
            'rid': rid,
            'tag': cells[0].get_text(strip=True),
            'code': code,
            'name': stock_anchor.get_text(strip=True),
            'detail_url': stock_anchor.get('href', ''),
            'close': cells[3].get_text(strip=True),
            'change_percent': cells[4].get_text(strip=True),
            'turnover': cells[5].get_text(strip=True),
            'net_buy': cells[6].get_text(strip=True),
            'detail': detail_map.get(rid, {}),
        }

    def parse_longhu_html(self, html, date):
        target_date = self.normalize_date(date)
        soup = BeautifulSoup(html, 'html.parser')
        detail_map = {}
        for detail_block in soup.select('.rightcol .stockcont[rid]'):
            detail = self.parse_stock_detail(detail_block)
            detail_map[detail['rid']] = detail

        stocks = []
        for tr in soup.select('.leftcol .m-table tbody tr'):
            item = self.parse_stock_row(tr, detail_map)
            if item is not None:
                stocks.append(item)

        return {
            'date': target_date,
            'stocks': stocks,
        }

    def get_content(self, date, session=None):
        target_date = self.normalize_date(date)
        active_session = session or self.build_session(target_date)
        html = self.fetch_html(active_session, target_date)
        data = self.parse_longhu_html(html, target_date)
        logger.info(f'龙虎榜抓取完成, date={target_date}, stock_count={len(data["stocks"])}')
        return data

    def save(self, dates):
        if isinstance(dates, (str, datetime)):
            date_list = [self.normalize_date(dates)]
            single_date = True
        else:
            date_list = [self.normalize_date(date) for date in dates]
            single_date = False

        session = self.build_session(date_list[0])
        results = {}
        for date in date_list:
            data = self.get_content(date, session=session)
            save_path = self.get_save_path(date)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f'龙虎榜数据已保存: {save_path}')
            results[date] = {
                'save_path': save_path,
                'data': data,
            }

        if single_date:
            return results[date_list[0]]
        return results

    def auto_save_missing(self, earliest_date='2020-01-02'):
        from AshareData.utils.exchanges_utils.a_open import get_target_trade_date, get_trade_date_list

        all_dates = get_trade_date_list()
        end_date = get_target_trade_date()
        pending_dates = self.get_auto_scrap_dates(
            all_dates=all_dates,
            end_date=end_date,
            earliest_date=earliest_date,
        )
        logger.info(f'待抓取龙虎榜日期: {pending_dates}')
        if not pending_dates:
            return {}

        session = self.build_session(pending_dates[0])
        results = {}
        failed = 0
        with tqdm.tqdm(total=len(pending_dates), desc='Scraping longhu') as pbar, logging_redirect_tqdm():
            for date in pending_dates:
                try:
                    data = self.get_content(date, session=session)
                except ValueError as exc:
                    # 页面异常（当日无数据或登录态失效）：跳过不落盘，下次运行会重试
                    failed += 1
                    logger.error(f'龙虎榜抓取异常跳过: date={date}, error={exc}, 连续失败={failed}')
                    if failed >= 3:
                        raise RuntimeError('连续3个交易日抓取异常，疑似登录态失效，终止本次补抓') from exc
                    pbar.update(1)
                    continue
                failed = 0
                save_path = self.get_save_path(date)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                logger.info(f'龙虎榜数据已保存: {save_path}')
                results[date] = {
                    'save_path': save_path,
                    'data': data,
                }
                pbar.update(1)
        return results


def is_latest():
    """本地数据是否已覆盖到最近交易日（get_target_trade_date）。无参。"""
    target = get_target_trade_date()
    if not target:
        return False
    return os.path.isfile(f'{SCRAP_DATA_DIR}/longhu/{target[:4]}-{target[4:6]}-{target[6:8]}.json')


if __name__ == '__main__':
    try:
        scraper = ScrapLonghu()
        result = scraper.auto_save_missing(earliest_date='2020-01-02')
    except Exception:
        logger.error(traceback.format_exc())
        raise