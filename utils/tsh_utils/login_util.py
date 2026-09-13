import json
import os
import sys
import tqdm
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
import time

os.makedirs('business_tmp_files/tsh', exist_ok=True)

# 滑块验证码模型：AshareData/utils/slide_captcha_model 子模块
# （git@github.com:leon20th/slide_captcha_model.git，MiniYOLO + 线上权重）
_SLIDE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', 'slide_captcha_model'))
if os.path.dirname(_SLIDE_ROOT) not in sys.path:
    sys.path.append(os.path.dirname(_SLIDE_ROOT))   # 包式导入 slide_captcha_model.*（append，不抢同名模块）
_SLIDE_WEIGHTS = os.path.join(_SLIDE_ROOT, 'models', 'mini_yolo_online_best.pth')

# 破解验证码
def get_yolo_model():
    """加载滑块定位模型：直接用 slide_captcha_model 子模块（MiniYOLO + 线上权重）。"""
    import torch
    from slide_captcha_model.model import MiniYOLO

    ckpt = torch.load(_SLIDE_WEIGHTS, map_location='cpu', weights_only=False)
    model = MiniYOLO()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    return model

def predict_captcha(model, image_path):
    """预测滑块 x 位移：预处理与 slide_captcha_model/predict.py 一致
    （Resize(282,162)+ToTensor，x 归一×282）。"""
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    with torch.no_grad():
        image = Image.open(image_path).convert('RGB')
        input_tensor = transforms.Compose([
            transforms.Resize((282, 162)),
            transforms.ToTensor(),
        ])(image).unsqueeze(0)  # 添加批次维度
        outputs = model(input_tensor)
        pred_x = int(outputs[0, 0].item() * 282)
        return pred_x


def get_tsh_cookies_static():
    try:
        with open("business_tmp_files/tsh/tsh_cookies.json", "r") as f:
            cookie = json.load(f)
            return cookie
    except:
        return None


def login_tsh(driver):
    '''
    <div data-statid="sns_fxts_index.agree" class="btn riskhint-D-sure btn-h36 tk-riskhint-D-sure bluebg">我已阅读并同意<span class="timeBox" style="display: none;">（<span class="timeCount">0</span>S）</span></div>
    '''
    try:
        with open("business_tmp_files/tsh/tsh_cookies.json", "r") as f:
            cookie = json.load(f)
        driver.delete_all_cookies()
        for cookie_dict in cookie:
            driver.add_cookie(cookie_dict)
        driver.refresh()
        print('使用cookie登录中')
    except Exception as e:
        print('尝试使用cookie失败: ', e)

    try:
        risk_btn = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, '.btn.riskhint-D-sure.btn-h36.tk-riskhint-D-sure.bluebg'))
        )
        # 等待可点击
        WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, '.btn.riskhint-D-sure.btn-h36.tk-riskhint-D-sure.bluebg'))
        )
        risk_btn.click()
        print('点击风险提示同意按钮')
    except Exception as e:
        print(f'风险提示按钮未找到，可能已同意，继续登录')

    try:
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'J_THS_LoginBox'))
        )
    except Exception as e:
        print(f'登录框未找到，可能已登录，返回')
        return True
    need_login = driver.find_elements(By.ID, 'J_THS_LoginBox')
    if not need_login:
        return True

    try:
        WebDriverWait(driver, 10).until(
            EC.frame_to_be_available_and_switch_to_it((By.CSS_SELECTOR, '#J_THS_LoginBox iframe'))
        )

        model = get_yolo_model()

        username_input = driver.find_element(By.ID, 'uname')
        password_input = driver.find_element(By.ID, 'passwd')

        username_input.clear()
        password_input.clear()
        username_input.send_keys("mx_lxy72vqqf")
        password_input.send_keys("liang26535724")

        login_button = driver.find_element(By.CSS_SELECTOR, ".n_f.pointer.tc.submit_btn.enable_submit_btn")
        login_button.click()

        captcha_element = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'slicaptcha'))
        )

        cpt_err = None
        for i in range(10):
            try:
                # 刷新
                slicaptcha_icon = driver.find_element(By.ID, 'slicaptcha-icon')
                slicaptcha_icon.click()

                try:
                    captcha_warn_element = WebDriverWait(driver, 1).until(
                        EC.visibility_of_element_located((By.ID, 'slicaptcha-warn-btn'))
                    )
                    warn_btn = driver.find_element(By.ID, 'slicaptcha-warn-btn')
                    warn_btn.click()
                except:
                    pass

                captcha_img = WebDriverWait(driver, 15).until(
                    EC.visibility_of_element_located((By.ID, 'slicaptcha-img'))
                )
                captcha_img.screenshot(f'business_tmp_files/tsh/captcha_image.png')

                try:
                    pred_x = predict_captcha(model, 'business_tmp_files/tsh/captcha_image.png')
                    slider = driver.find_element(By.ID, 'slider')
                    from selenium.webdriver import ActionChains
                    action = ActionChains(driver)
                    action.click_and_hold(slider).move_by_offset(pred_x, 0).release().perform()
                except:
                    pass

                try:
                    time.sleep(2)
                    captcha_element.is_enabled()
                except:
                    # 保存cookie
                    cookies = driver.get_cookies()
                    with open("business_tmp_files/tsh/tsh_cookies.json", "w") as f:
                        json.dump(cookies, f)
                    return True
            except Exception as cpt_err:
                print(f"尝试第{i+1}次破解失败: {cpt_err}")
                continue
    except Exception as e:
        return f"登录失败: {e}"
    return False


def login_wencai(driver):
    try:
        with open("business_tmp_files/tsh/wencai_cookies.json", "r") as f:
            cookie = json.load(f)
        driver.delete_all_cookies()
        for cookie_dict in cookie:
            driver.add_cookie(cookie_dict)
        driver.refresh()
        print('使用cookie登录中')
    except Exception as e:
        print('尝试使用cookie失败: ', e)

    try:
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, '.login_btn.nav_word'))
        )
    except:
        return True
    need_login = driver.find_elements(By.CSS_SELECTOR, '.login_btn.nav_word')
    if not need_login:
        return True

    need_login[0].click()

    try:
        WebDriverWait(driver, 10).until(
            EC.frame_to_be_available_and_switch_to_it((By.ID, 'login_iframe'))
        )

        model = get_yolo_model()

        driver.find_element(By.ID, 'to_account_login').click()

        username_input = driver.find_element(By.ID, 'uname')
        password_input = driver.find_element(By.ID, 'passwd')

        username_input.clear()
        password_input.clear()
        username_input.send_keys("mx_lxy72vqqf")
        password_input.send_keys("liang26535724")

        login_button = driver.find_element(By.CSS_SELECTOR, ".n_f.pointer.tc.submit_btn.enable_submit_btn")
        login_button.click()

        captcha_element = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'slicaptcha'))
        )

        cpt_err = None
        for i in range(10):
            try:
                # 刷新
                slicaptcha_icon = driver.find_element(By.ID, 'slicaptcha-icon')
                slicaptcha_icon.click()

                try:
                    captcha_warn_element = WebDriverWait(driver, 1).until(
                        EC.visibility_of_element_located((By.ID, 'slicaptcha-warn-btn'))
                    )
                    warn_btn = driver.find_element(By.ID, 'slicaptcha-warn-btn')
                    warn_btn.click()
                except:
                    pass

                captcha_img = WebDriverWait(driver, 15).until(
                    EC.visibility_of_element_located((By.ID, 'slicaptcha-img'))
                )
                captcha_img.screenshot(f'business_tmp_files/tsh/captcha_image.png')

                try:
                    pred_x = predict_captcha(model, 'business_tmp_files/tsh/captcha_image.png')
                    slider = driver.find_element(By.ID, 'slider')
                    from selenium.webdriver import ActionChains
                    action = ActionChains(driver)
                    action.click_and_hold(slider).move_by_offset(pred_x, 0).release().perform()
                except:
                    pass

                try:
                    time.sleep(2)
                    captcha_element.is_enabled()
                except:
                    # 保存cookie
                    cookies = driver.get_cookies()
                    with open("business_tmp_files/tsh/wencai_cookies.json", "w") as f:
                        json.dump(cookies, f)
                    return True
            except Exception as cpt_err:
                print(f"尝试第{i+1}次破解失败: {cpt_err}")
                continue
    except Exception as e:
        return f"登录失败: {e}"
    return False


def manual_login_wencai():
    import time
    from env_setting import CHROME_DRIVER_PATH
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.service import Service

    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36')

    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-popup-blocking")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_argument("--disable-extensions")
    service = Service(executable_path=CHROME_DRIVER_PATH)
    driver = webdriver.Chrome(options=chrome_options, service=service)
    url = 'https://www.iwencai.com/unifiedwap/home/index'
    driver.get(url)
    print("请手动登录完成后关闭浏览器")
    for i in tqdm.tqdm(range(60), desc="等待登录完成"):
        time.sleep(1)
    # 保存cookie
    cookies = driver.get_cookies()
    with open("business_tmp_files/tsh/wencai_cookies.json", "w") as f:
        json.dump(cookies, f)
    print("登录信息已保存，下次将自动登录")
    driver.quit()
    

if __name__ == "__main__":
    manual_login_wencai()