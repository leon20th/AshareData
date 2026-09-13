import os
from env_setting import ROOT
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import ElementClickInterceptedException
import time
import base64
import re
from datetime import datetime, timedelta
from AshareData.utils.log_util import setup_logging, get_logger
from AshareData.datautils.regime_scripts.scrap import Scrap
from AshareData.utils.tsh_utils.login_util import login_tsh
import json
import traceback
import tqdm

setup_logging()
logger = get_logger('韭研爬取')

class ScrapFupanImgs(Scrap):
    def __init__(self, time_limit=''):
        #self.url = "https://t.10jqka.com.cn/lgt/user_page/?userid=601319077#/"
        self.url = 'https://t.10jqka.com.cn/circle/211687/'
        self.time_range = []
        if not time_limit:
            # 今天0点
            self.time_begin = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            self.time_begin = datetime.strptime(time_limit, '%Y-%m-%d %H:%M')
        self.time_end = datetime.today().replace(hour=23, minute=59, second=59, microsecond=0)
        self.time_range = [i.strftime('%Y-%m-%d %H:%M') for i in (self.time_begin, self.time_end)]
        date = self.time_begin.strftime('%Y%m%d')
        self.path = f'{ROOT}/AshareData/dataset/scrap_data/zhangtingimgs'
        os.makedirs(self.path, exist_ok=True)
        self.save_path = f'{self.path}/{date}.json'
        self.save_dir = self.path

    def get_current_page(self, driver, timeout=10) -> int:
        wait = WebDriverWait(driver, timeout)
        cur = wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, 'li.page.active span.curNum'))
        )
        return int(cur.text.strip())

    def safe_click(self, driver, element):
        driver.execute_script(
            "arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
            element,
        )
        time.sleep(0.2)
        try:
            element.click()
        except ElementClickInterceptedException:
            driver.execute_script("arguments[0].click();", element)

    def goto_page(self, driver, n: int, timeout=10):
        wait = WebDriverWait(driver, timeout)

        # 1) 打开页码下拉（你的 ul_page_wrap 默认 display:none）
        page_show = wait.until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'li.page.active .pageShow'))
        )
        self.safe_click(driver, page_show)

        # 2) 点击目标页（用 data-page 精准定位）
        target_selector = f'li.page.active .ul_page .a_page[data-page="{n}"]'
        target = wait.until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, target_selector))
        )
        self.safe_click(driver, target)

        # 3) 等待当前页数字更新为 n
        wait.until(
            lambda d: d.find_element(By.CSS_SELECTOR, 'li.page.active span.curNum').text.strip() == str(n)
        )

    def get_content_with_selenium(self):
        """
        使用Selenium获取动态加载的内容
        """
        driver = self.get_driver()
        self.block_network_imgs(driver)
        scrap_article = []

        try:
            driver.get(self.url)

            # 等待页面加载
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # 登录
            ret = login_tsh(driver)
            logger.info(ret)

            # 等待页面加载
            wait = WebDriverWait(driver, 10)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))

            # 滚动页面以确保所有内容加载
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)


            # 遍历所有文章
            scrap_stop = False
            process_page = 1
            while True:
                wait = WebDriverWait(driver, 10)
                wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, 'postlist-ul')))
                post_list = driver.find_element(By.CLASS_NAME, 'postlist-ul')
                posts = post_list.find_elements(By.CSS_SELECTOR, "[class='db single-main']")
                for idx, post_ele in enumerate(posts):
                    if scrap_stop:
                        break
                    post_id = post_ele.get_attribute('data-pid')
                    post_title_ele = post_ele.find_element(By.CLASS_NAME, 'post-title')
                    driver.execute_script("arguments[0].click();", post_ele)
                    dialog_name = f'J_Dialog_{idx}'
                    element = driver.find_element(By.CSS_SELECTOR, "[class='tac detail-title']")
                    title = element.text
                    date_str = driver.find_element(By.CSS_SELECTOR, "[class='detail-date c999']").text
                    time_str = driver.find_element(By.CSS_SELECTOR, "[class='detail-time c999']").text
                    content = driver.find_element(By.CSS_SELECTOR, "[class='wdwrap post-text-main c444']").text
                    imgs = driver.find_elements(By.CSS_SELECTOR, "[class='wdwrap post-text-main c444'] img")
                    img_urls = [img.get_attribute('src') for img in imgs]
                    if (date_str + ' ' + time_str) < self.time_range[0]:
                        logger.info(f'scrap done, scrap article, {len(scrap_article)}')
                        scrap_stop = True
                        break
                    if self.time_range[0] <= (date_str + ' ' + time_str) < self.time_range[1]:
                        scrap_article.append(dict(
                            pid=post_id,
                            title=title,
                            date=date_str,
                            time=time_str,
                            content=content,
                            img_urls=img_urls
                        ))
                    dialog_close = driver.find_element(By.CSS_SELECTOR, f"[class='ui-dialog-close postdetail-dialog-close']")
                    driver.execute_script("arguments[0].click();", dialog_close)
                if scrap_stop:
                    break
                cur_page = self.get_current_page(driver)
                logger.info(f'已抓取当前页: {cur_page}, 准备进入下一页: {cur_page + 1}')
                if cur_page % 3 == 0:
                    # 重新打开新页面
                    self.reopen_current_page(driver)
                    self.block_network_imgs(driver)
                    logger.info(f'已重新打开页面，当前页: {cur_page}')
                self.goto_page(driver, cur_page + 1)
        except Exception as e:
            logger.error(f"Error during scraping fupan imgs: {traceback.format_exc()}")
        finally:
            driver.quit()
        return scrap_article

    def download_image(self, url):
        """
        下载图片并返回二进制数据
        """
        import requests
        try:
            response = requests.get(url)
            if response.status_code == 200:
                return response.content
            else:
                logger.info(f"Failed to download image from {url}, status code: {response.status_code}")
                return None
        except Exception as e:
            logger.info(f"Error downloading image from {url}: {e}")
            return None

    def save_article(self):
        scrap_article = self.get_content_with_selenium()
        date2articles = {}
        for article in scrap_article:
            date = article['date']
            date2articles.setdefault(date, []).append(article)
        
        for date, articles in tqdm.tqdm(date2articles.items(), desc='Saving articles'):
            date_save_dir = os.path.join(self.save_dir, date)
            os.makedirs(date_save_dir, exist_ok=True)
            save_path = os.path.join(date_save_dir, f'{date}.json')
            img_urls = []
            with open(save_path, 'w', encoding='utf-8') as f:
                for article in articles:
                    f.write(json.dumps(article, ensure_ascii=False) + '\n')
                    img_urls += article['img_urls']
            for idx, url in enumerate(img_urls):
                image_data = self.download_image(url)
                if image_data:
                    filename = f"{date_save_dir}/img-{idx}.png"
                    with open(filename, 'wb') as f:
                        f.write(image_data)
        return date2articles

if __name__ == "__main__":
    from AshareData.utils.exchanges_utils.a_open import is_a_share_open_today
    path = f'{ROOT}/AshareData/dataset/scrap_data/zhangtingimgs'
    max_date = max(os.listdir(path))
    max_date_next = (datetime.strptime(max_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
    time_limit = f'{max_date_next} 00:00'
    print(f"上次爬取日期: {max_date}, 本次爬取日期: {max_date_next}")
    # 使用示例
    scraper = ScrapFupanImgs(time_limit=time_limit)
    scraper.save_article()