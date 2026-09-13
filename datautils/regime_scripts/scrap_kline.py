"""Scrape kline images (分时/日K/周K/月K) for stocks in a given Excel sheet
and insert them back into the sheet as embedded images.

Usage
-----
    scraper = ScrapKline(
        xlsx_path='...predict_regime.xlsx',
        sheet_name='focus_pool',
        save_dir='predict_v1',
        scrap_continue=True,
        date='20260521',
    )
    scraper.get_content_with_selenium()
"""
from __future__ import annotations

import base64
import io
import os
import time
import traceback
from collections import Counter
from datetime import datetime
from typing import List

import tqdm
from PIL import Image
from openpyxl import Workbook, load_workbook
from openpyxl.drawing.image import Image as xlsx_Image
from openpyxl.styles import Alignment, PatternFill
from openpyxl.utils import get_column_letter
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm.contrib.logging import logging_redirect_tqdm

from env_setting import ROOT, CHROME_DRIVER_PATH
from AshareData.datautils.regime_scripts.scrap import Scrap
from AshareData.utils.log_util import setup_logging, get_logger
from AshareData.utils.tsh_utils.login_util import login_wencai

setup_logging()
logger = get_logger('K线抓取')

# Ordered list of kline tab names as they appear on wencai
KLINE_TYPES = ['日K', '分时', '周K', '月K']


class ScrapKline(Scrap):
    def __init__(
        self,
        xlsx_path: str,
        sheet_name: str,
        save_dir: str,
        scrap_continue: bool = True,
        date: str = '',
    ):
        self.url = 'https://www.iwencai.com/unifiedwap/home/index'
        self.xlsx_path = xlsx_path
        self.sheet_name = sheet_name
        self.save_dir = save_dir
        self.scrap_continue = scrap_continue
        if not date:
            date = datetime.now().strftime('%Y%m%d')
        self.date = date
        self.kline_root = (
            f'{ROOT}/AshareData/dataset/scrap_data/kline/{save_dir}/{date}'
        )

    # ------------------------------------------------------------------
    # Driver / login
    # ------------------------------------------------------------------

    def get_login_driver(self):
        driver = self.get_driver()
        driver.get(self.url)
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body'))
        )
        ret = login_wencai(driver)
        if not ret:
            driver.quit()
            return None
        return driver

    # ------------------------------------------------------------------
    # Kline scraping (mirrors ScrapZTFX.get_kline)
    # ------------------------------------------------------------------

    def get_kline(self, driver, url: str, code: str, kline_path: str):
        """Navigate to *url*, iterate every kline tab and save a PNG per tab."""
        for attempt in range(3):
            try:
                driver.get(url)
                kline2 = WebDriverWait(driver, 10).until(
                    EC.any_of(
                        EC.presence_of_element_located(
                            (By.CLASS_NAME, 'jgy_kline2_page')
                        ),
                        EC.presence_of_element_located(
                            (By.CLASS_NAME, 'kline2')
                        ),
                    )
                )
                break
            except Exception:
                if attempt == 2:
                    logger.error(f'get_kline: failed to load page, code={code}')
                    return

        WebDriverWait(kline2, 10).until(
            EC.any_of(
                EC.presence_of_element_located((By.CLASS_NAME, 'jgy_kline_tab')),
                EC.presence_of_element_located((By.CLASS_NAME, 'kline2_tab')),
            )
        )
        k_tab_items = kline2.find_elements(
            By.CSS_SELECTOR,
            '.select_box, .kline2_tab_item, .kline2_tab_item.kline2_select_item',
        )
        for k_t in k_tab_items:
            driver.execute_script('arguments[0].click();', k_t)
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'kline_chart'))
            )
            time.sleep(1)
            # Capture and crop screenshot
            mw = 720
            raw = kline2.screenshot_as_png
            image = Image.open(io.BytesIO(raw))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            iw, ih = image.size
            if iw > mw:
                image = image.crop((0, 0, mw, ih))
            save_path = os.path.join(kline_path, f'{code}_{k_t.text}.png')
            image.save(save_path)
            logger.info(f'saved: {save_path}')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _short_code(code: str) -> str:
        """'sh.600000' -> '600000'; already short codes are returned as-is."""
        return code.split('.')[-1] if '.' in code else code

    def _klines_already_scraped(self, short_code: str) -> bool:
        """Return True if all 4 kline PNGs for *short_code* already exist."""
        if not os.path.isdir(self.kline_root):
            return False
        existing = set(os.listdir(self.kline_root))
        return all(
            any(f.startswith(f'{short_code}_{ktype}') for f in existing)
            for ktype in KLINE_TYPES
        )

    def _lookup_kline_files(self, short_code: str) -> List[str]:
        """Return file paths in KLINE_TYPES order; empty string if not found."""
        result = {k: '' for k in KLINE_TYPES}
        if os.path.isdir(self.kline_root):
            for fname in os.listdir(self.kline_root):
                for ktype in KLINE_TYPES:
                    if fname.startswith(f'{short_code}_{ktype}'):
                        result[ktype] = os.path.join(self.kline_root, fname)
        return [result[k] for k in KLINE_TYPES]

    def _read_codes_from_sheet(self) -> List[str]:
        """Read all stock codes from the 'code' column of the target sheet."""
        wb = load_workbook(self.xlsx_path, read_only=True, data_only=True)
        ws = wb[self.sheet_name]
        first_row = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
        headers = list(first_row)
        if 'code' not in headers:
            wb.close()
            raise ValueError(
                f"'code' column not found in sheet '{self.sheet_name}'. "
                f"Available headers: {headers}"
            )
        code_idx = headers.index('code')
        codes: List[str] = []
        for row in ws.iter_rows(min_row=2, values_only=True):
            val = row[code_idx]
            if val:
                codes.append(str(val).strip())
        wb.close()
        return codes

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def get_content_with_selenium(self):
        """Scrape klines for all stocks in the sheet, then insert images."""
        os.makedirs(self.kline_root, exist_ok=True)
        codes = self._read_codes_from_sheet()
        logger.info(
            f'Total stocks: {len(codes)}, sheet={self.sheet_name}, '
            f'kline_root={self.kline_root}'
        )

        driver = self.get_login_driver()
        if driver is None:
            logger.error('Login failed, aborting.')
            return

        try:
            with tqdm.tqdm(total=len(codes)) as pbar, logging_redirect_tqdm():
                for code in codes:
                    short_code = self._short_code(code)
                    if self._klines_already_scraped(short_code):
                        logger.info(f'skip (already scraped): code={code}')
                        pbar.update(1)
                        continue
                    try:
                        url = (
                            f'https://www.iwencai.com/unifiedwap/result'
                            f'?w={short_code}&querytype=stock'
                        )
                        logger.info(f'processing: code={code}')
                        self.get_kline(driver, url, short_code, self.kline_root)
                        logger.info(f'kline done: code={code}')
                    except Exception:
                        logger.error(
                            f'Error scraping kline code={code}:\n'
                            f'{traceback.format_exc()}'
                        )
                    finally:
                        pbar.update(1)
        finally:
            driver.quit()

        self.insert_klines()

    # ------------------------------------------------------------------
    # Image insertion
    # ------------------------------------------------------------------

    def insert_klines(self):
        """Build a standalone workbook for the target sheet and insert kline images."""
        source_wb = load_workbook(self.xlsx_path, read_only=True, data_only=False)
        source_ws = source_wb[self.sheet_name]

        first_row = list(
            next(source_ws.iter_rows(min_row=1, max_row=1, values_only=True))
        )
        if 'code' not in first_row:
            source_wb.close()
            raise ValueError(
                f"'code' column not found in sheet '{self.sheet_name}'."
            )
        code_idx = first_row.index('code')
        conception_idx = first_row.index('conception') if 'conception' in first_row else -1
        last_day_uplimit_idx = (
            first_row.index('last_day_uplimit')
            if 'last_day_uplimit' in first_row
            else -1
        )

        # Shared styles
        header_fill = PatternFill(
            start_color='FFEBCD', end_color='FFEBCD', fill_type='solid'
        )
        center_align = Alignment(
            horizontal='center', vertical='center', wrap_text=True
        )
        left_align = Alignment(
            horizontal='left', vertical='center', wrap_text=True
        )

        # Compute suggest from prob columns (original 0-based indices)
        def _fv(row_vals, *names):
            for n in names:
                if n in first_row:
                    v = row_vals[first_row.index(n)]
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        return 0.0
            return 0.0

        def _suggest(row_vals):
            return (
                _fv(row_vals, 'prob_uplimit')
                + _fv(row_vals, 'prob_big_up')
                - _fv(row_vals, 'prob_downlimit')
                - _fv(row_vals, 'prob_big_down')
                - _fv(row_vals, 'prob_notrade', 'prob_no_trade')
            )

        def _row_uplimit_flag(row_vals):
            if last_day_uplimit_idx < 0:
                return 0
            try:
                return int(float(row_vals[last_day_uplimit_idx] or 0))
            except (TypeError, ValueError):
                return 0

        def _row_conception(row_vals):
            if conception_idx < 0:
                return ''
            value = row_vals[conception_idx]
            return '' if value is None else str(value).strip()

        # Read all data rows into memory before modifying the sheet
        data_rows = [
            list(row)
            for row in source_ws.iter_rows(min_row=2, values_only=True)
            if any(v is not None for v in row)
        ]
        source_wb.close()

        if last_day_uplimit_idx >= 0:
            uplimit_rows = [row for row in data_rows if _row_uplimit_flag(row) == 1]
            non_uplimit_rows = [row for row in data_rows if _row_uplimit_flag(row) == 0]
            conception_counts = Counter(
                conception
                for conception in (_row_conception(row) for row in uplimit_rows)
                if conception
            )
            conception_suggest_sums: dict[str, float] = {}
            for row in uplimit_rows:
                conception = _row_conception(row)
                if not conception:
                    continue
                conception_suggest_sums[conception] = (
                    conception_suggest_sums.get(conception, 0.0) + _suggest(row)
                )
            conception_avg_suggests = {
                conception: conception_suggest_sums[conception] / count
                for conception, count in conception_counts.items()
                if count > 0
            }
            uplimit_rows.sort(
                key=lambda row: (
                    -conception_counts.get(_row_conception(row), 0),
                    -conception_avg_suggests.get(_row_conception(row), 0.0),
                    -_suggest(row),
                )
            )
            non_uplimit_rows.sort(key=lambda row: -_suggest(row))
            sheet_specs = [
                (f'涨停池({len(uplimit_rows)})', uplimit_rows),
                (f'未涨停池({len(non_uplimit_rows)})', non_uplimit_rows),
            ]
        else:
            data_rows.sort(key=lambda row: -_suggest(row))
            sheet_specs = [(self.sheet_name, data_rows)]

        # Insert 4 kline columns right after last_day_uplimit
        if last_day_uplimit_idx >= 0:
            insert_col = last_day_uplimit_idx + 2  # 1-based, after it
        else:
            insert_col = len(first_row) + 1

        kline_titles = ['日K', '分时', '周K', '月K']
        output_headers = list(first_row)
        for offset, title in enumerate(kline_titles):
            output_headers.insert(insert_col - 1 + offset, title)

        def _build_output_row(row_vals):
            output_row = []
            for orig_0idx, val in enumerate(row_vals):
                if orig_0idx == insert_col - 1:
                    output_row.extend([''] * len(kline_titles))
                if first_row[orig_0idx] == 'suggest':
                    val = round(_suggest(row_vals), 6)
                output_row.append(val)
            if insert_col - 1 >= len(row_vals):
                output_row.extend([''] * len(kline_titles))
            return output_row

        def _populate_sheet(ws, rows):
            ws.freeze_panes = 'A2'
            ws.append(output_headers)
            for col_index, cell in enumerate(ws[1], start=1):
                cell.alignment = center_align
                if insert_col <= col_index < insert_col + len(kline_titles):
                    cell.fill = header_fill
                    ws.column_dimensions[get_column_letter(col_index)].width = 62

            max_widths = [len(str(header or '')) for header in output_headers]
            inserted = 0
            for row_num, row_vals in enumerate(rows, start=2):
                output_row = _build_output_row(row_vals)
                for col_index, val in enumerate(output_row, start=1):
                    ws.cell(row=row_num, column=col_index, value=val).alignment = left_align
                    if insert_col <= col_index < insert_col + len(kline_titles):
                        continue
                    cell_text = '' if val is None else str(val)
                    if len(cell_text) > max_widths[col_index - 1]:
                        max_widths[col_index - 1] = len(cell_text)

                if not row_vals[code_idx]:
                    continue
                short_code = self._short_code(str(row_vals[code_idx]).strip())
                kline_files = self._lookup_kline_files(short_code)
                any_inserted = False
                for offset, kpath in enumerate(kline_files):
                    if not kpath or not os.path.isfile(kpath):
                        continue
                    kimg = xlsx_Image(kpath)
                    kimg.width = int(kimg.width * 0.65)
                    kimg.height = int(kimg.height * 0.65)
                    ws.add_image(kimg, f'{get_column_letter(insert_col + offset)}{row_num}')
                    any_inserted = True
                if any_inserted:
                    ws.row_dimensions[row_num].height = 270
                    inserted += 1

            for col_index, width in enumerate(max_widths, start=1):
                if insert_col <= col_index < insert_col + len(kline_titles):
                    continue
                ws.column_dimensions[get_column_letter(col_index)].width = min(
                    max(width + 2, 10),
                    40,
                )
            return inserted

        output_wb = Workbook()
        first_title, first_rows = sheet_specs[0]
        output_ws = output_wb.active
        output_ws.title = first_title
        inserted = _populate_sheet(output_ws, first_rows)
        for title, rows in sheet_specs[1:]:
            inserted += _populate_sheet(output_wb.create_sheet(title=title), rows)

        stem, ext = os.path.splitext(self.xlsx_path)
        output_path = f'{stem}_kline{ext}'
        output_wb.save(output_path)
        logger.info(
            f'Kline images inserted for {inserted} stocks. '
            f'Saved: {output_path}'
        )


if __name__ == '__main__':
    from env_setting import ROOT
    from AshareData.utils.exchanges_utils.a_open import get_trade_date_list, get_target_trade_date
    trade_list = get_trade_date_list()
    target_date = get_target_trade_date()
    next_date = trade_list[trade_list.index(target_date) + 1]
    scraper = ScrapKline(
        xlsx_path=(
            f'{ROOT}/PPOFramework/regime_models/models/model_result_online/{next_date}/{next_date}_predict_regime.xlsx'
        ),
        sheet_name='focus_pool',
        save_dir='predict_v1',
        scrap_continue=True,
        date=next_date,
    )
    scraper.get_content_with_selenium()