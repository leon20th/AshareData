import json
import os
import sys
import tqdm
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

from AshareData.paths import CHROME_DRIVER_PATH, TMP_DIR

os.makedirs(TMP_DIR, exist_ok=True)

# 滑块验证码模型：tsh_utils/slide_captcha_model 子模块（git@github.com:leon20th/slide_captcha_model.git，MiniYOLO + 线上权重）
_SLIDE_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 'slide_captcha_model'))
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
        with open(f"{TMP_DIR}/tsh_cookies.json", "r") as f:
            cookie = json.load(f)
            return cookie
    except:
        return None


def login_tsh(driver):
    '''
    <div data-statid="sns_fxts_index.agree" class="btn riskhint-D-sure btn-h36 tk-riskhint-D-sure bluebg">我已阅读并同意<span class="timeBox" style="display: none;">（<span class="timeCount">0</span>S）</span></div>
    '''
    try:
        with open(f"{TMP_DIR}/tsh_cookies.json", "r") as f:
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
    except Exception:
        print('风险提示按钮未找到，可能已同意，继续登录')

    try:
        WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'J_THS_LoginBox'))
        )
    except Exception:
        print('登录框未找到，可能已登录，返回')
        return True
    need_login = driver.find_elements(By.ID, 'J_THS_LoginBox')
    if not need_login:
        return True

    try:
        WebDriverWait(driver, 10).until(
            EC.frame_to_be_available_and_switch_to_it((By.CSS_SELECTOR, '#J_THS_LoginBox iframe'))
        )

        model = get_yolo_model()

        with open(f"{TMP_DIR}/tsh_account.json", "r", encoding="utf-8") as f:
            account = json.load(f)

        username_input = driver.find_element(By.ID, 'uname')
        password_input = driver.find_element(By.ID, 'passwd')

        username_input.clear()
        password_input.clear()
        username_input.send_keys(account["username"])
        password_input.send_keys(account["password"])

        login_button = driver.find_element(By.CSS_SELECTOR, ".n_f.pointer.tc.submit_btn.enable_submit_btn")
        login_button.click()

        captcha_element = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'slicaptcha'))
        )

        for i in range(10):
            try:
                # 刷新
                slicaptcha_icon = driver.find_element(By.ID, 'slicaptcha-icon')
                slicaptcha_icon.click()

                try:
                    WebDriverWait(driver, 1).until(
                        EC.visibility_of_element_located((By.ID, 'slicaptcha-warn-btn'))
                    )
                    warn_btn = driver.find_element(By.ID, 'slicaptcha-warn-btn')
                    warn_btn.click()
                except:
                    pass

                captcha_img = WebDriverWait(driver, 15).until(
                    EC.visibility_of_element_located((By.ID, 'slicaptcha-img'))
                )
                captcha_img.screenshot(f'{TMP_DIR}/captcha_image.png')

                try:
                    pred_x = predict_captcha(model, f'{TMP_DIR}/captcha_image.png')
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
                    with open(f"{TMP_DIR}/tsh_cookies.json", "w") as f:
                        json.dump(cookies, f)
                    return True
            except Exception as cpt_err:
                print(f"尝试第{i+1}次破解失败: {cpt_err}")
                continue
    except Exception as e:
        return f"登录失败: {e}"
    return False


_NETLOG_HOOK_JS = """
window.__netlog = window.__netlog || [];
if (!window.__hooked) {
  window.__hooked = true;
  const oo = XMLHttpRequest.prototype.open, os = XMLHttpRequest.prototype.send;
  XMLHttpRequest.prototype.open = function(m, u) { this.__m = m; this.__u = String(u); return oo.apply(this, arguments); };
  XMLHttpRequest.prototype.send = function(b) {
    try { window.__netlog.push({dir: 'REQ', m: this.__m, u: this.__u, body: String(b || '').slice(0, 400)}); } catch (e) {}
    const self = this;
    this.addEventListener('load', function() {
      try { window.__netlog.push({dir: 'RESP', u: self.__u, status: self.status, resp: String(self.responseText || '').slice(0, 400)}); } catch (e) {}
    });
    return os.apply(this, arguments);
  };
}
"""


def _wencai_logged_in(driver, captcha_element, base_cookie_names):
    """登录成功判定：验证码元素摘除，或出现新的 cookie（登录凭证直证）。"""
    try:
        captcha_element.is_enabled()
    except Exception:
        return True
    try:
        names = {c['name'] for c in driver.get_cookies()}
    except Exception:
        return False
    return bool(names - base_cookie_names)


def _wencai_login_error(driver):
    """从网络钩子里读取最近一次登录响应的错误信息。"""
    try:
        events = driver.execute_script(
            "return (window.__netlog || []).filter(e => e.u && e.u.indexOf('dologin') >= 0 && e.dir === 'RESP').slice(-2)")
    except Exception:
        return None
    for ev in events or []:
        resp = ev.get('resp') or ''
        try:
            data = json.loads(resp)
            return f"{data.get('errorcode')}: {data.get('errormsg')}"
        except Exception:
            if resp:
                return resp[:120]
    return None


def login_wencai(driver):
    try:
        with open(f"{TMP_DIR}/wencai_cookies.json", "r") as f:
            cookie = json.load(f)
        driver.delete_all_cookies()
        loaded = 0
        for cookie_dict in cookie:
            try:
                driver.add_cookie(cookie_dict)
                loaded += 1
            except Exception:
                pass
        driver.refresh()
        print(f'使用cookie登录中（{loaded}/{len(cookie)} 条）')
    except Exception as e:
        print('尝试使用cookie失败: ', e)

    # 新版登录入口为左下角 span.login，兼容旧版 .login_btn.nav_word
    try:
        WebDriverWait(driver, 8).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'span.login, .login_btn.nav_word'))
        )
    except Exception:
        print('未出现登录入口（已登录或页面结构变化），跳过登录')
        return True
    need_login = (driver.find_elements(By.CSS_SELECTOR, 'span.login')
                  or driver.find_elements(By.CSS_SELECTOR, '.login_btn.nav_word'))
    try:
        need_login[0].click()
    except Exception:
        driver.execute_script('arguments[0].click();', need_login[0])

    try:
        WebDriverWait(driver, 8).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, '#login_iframe'))
        )
    except Exception:
        # 子元素点击未触发弹窗时，再试父级容器
        parents = driver.find_elements(By.CSS_SELECTOR, '.user-info')
        if parents:
            driver.execute_script('arguments[0].click();', parents[0])

    try:
        WebDriverWait(driver, 10).until(
            EC.frame_to_be_available_and_switch_to_it((By.ID, 'login_iframe'))
        )

        # 注入 XHR 钩子：记录登录请求/响应，便于输出被拒原因
        try:
            driver.execute_script(_NETLOG_HOOK_JS)
        except Exception:
            pass

        model = get_yolo_model()

        account_tab = driver.find_elements(By.ID, 'to_account_login')
        if account_tab:
            try:
                account_tab[0].click()
            except Exception:
                driver.execute_script('arguments[0].click();', account_tab[0])

        with open(f"{TMP_DIR}/tsh_account.json", "r", encoding="utf-8") as f:
            account = json.load(f)

        username_input = driver.find_element(By.ID, 'uname')
        password_input = driver.find_element(By.ID, 'passwd')

        username_input.clear()
        password_input.clear()
        username_input.send_keys(account["username"])
        password_input.send_keys(account["password"])
        time.sleep(1)

        buttons = driver.find_elements(By.CSS_SELECTOR, '.enable_submit_btn') or \
            [b for b in driver.find_elements(By.CSS_SELECTOR, '.submit_btn') if b.is_displayed()]
        try:
            buttons[0].click()
        except Exception:
            driver.execute_script('arguments[0].click();', buttons[0])

        captcha_element = WebDriverWait(driver, 10).until(
            EC.visibility_of_element_located((By.ID, 'slicaptcha'))
        )
        base_cookie_names = {c['name'] for c in driver.get_cookies()}

        for i in range(10):
            # 验证码不可见时（上轮滑过但登录被拒），重新点提交触发新验证码
            captcha_els = driver.find_elements(By.ID, 'slicaptcha')
            if not captcha_els or not captcha_els[0].is_displayed():
                buttons = [b for b in driver.find_elements(By.CSS_SELECTOR, '.submit_btn') if b.is_displayed()]
                if not buttons:
                    print('登录按钮不可见，停止重试，以匿名会话继续抓取')
                    break
                try:
                    buttons[0].click()
                except Exception:
                    driver.execute_script('arguments[0].click();', buttons[0])
                try:
                    WebDriverWait(driver, 6).until(EC.visibility_of_element_located((By.ID, 'slicaptcha')))
                except Exception:
                    print('验证码未出现，停止重试，以匿名会话继续抓取')
                    break
            try:
                # 刷新
                slicaptcha_icon = driver.find_element(By.ID, 'slicaptcha-icon')
                try:
                    slicaptcha_icon.click()
                except Exception:
                    driver.execute_script('arguments[0].click();', slicaptcha_icon)

                try:
                    WebDriverWait(driver, 1).until(
                        EC.visibility_of_element_located((By.ID, 'slicaptcha-warn-btn'))
                    )
                    warn_btn = driver.find_element(By.ID, 'slicaptcha-warn-btn')
                    try:
                        warn_btn.click()
                    except Exception:
                        driver.execute_script('arguments[0].click();', warn_btn)
                except Exception:
                    pass

                captcha_img = WebDriverWait(driver, 15).until(
                    EC.visibility_of_element_located((By.ID, 'slicaptcha-img'))
                )
                captcha_img.screenshot(f'{TMP_DIR}/captcha_image.png')

                try:
                    pred_x = predict_captcha(model, f'{TMP_DIR}/captcha_image.png')
                    slider = driver.find_element(By.ID, 'slider')
                    from selenium.webdriver import ActionChains
                    action = ActionChains(driver)
                    action.click_and_hold(slider).move_by_offset(pred_x, 0).release().perform()
                except Exception:
                    pass
            except Exception as cpt_err:
                print(f"尝试第{i+1}次破解失败: {cpt_err}")
                continue

            # 判定本轮结果（最多等 10 秒）：登录成功 / 验证码重排 / 登录被拒
            outcome = None
            for _ in range(20):
                time.sleep(0.5)
                if _wencai_logged_in(driver, captcha_element, base_cookie_names):
                    outcome = 'ok'
                    break
                captcha_els = driver.find_elements(By.ID, 'slicaptcha')
                if captcha_els and captcha_els[0].is_displayed():
                    outcome = 'captcha_fail'
                    break
            if outcome == 'ok':
                # 登录成功后立即保存 cookie
                try:
                    driver.switch_to.default_content()
                except Exception:
                    pass
                cookies = driver.get_cookies()
                with open(f"{TMP_DIR}/wencai_cookies.json", "w") as f:
                    json.dump(cookies, f)
                print(f'登录成功，cookies 已保存（{len(cookies)} 条）')
                return True
            if outcome == 'captcha_fail':
                print(f'第{i+1}次滑块验证未通过，重试')
                continue
            # 组件已关闭但未产生登录凭证 → 登录请求被拒（如账号密码错误），停止重试防锁号
            err = _wencai_login_error(driver)
            print(f'登录被服务器拒绝（{err or "未知原因"}），停止重试，以匿名会话继续抓取')
            try:
                driver.switch_to.default_content()
            except Exception:
                pass
            return True
    except Exception as e:
        print(f'登录流程失败: {e}，以匿名会话继续抓取')
        try:
            driver.switch_to.default_content()
        except Exception:
            pass
        return True
    print('滑块验证多次未通过，以匿名会话继续抓取')
    try:
        driver.switch_to.default_content()
    except Exception:
        pass
    return True


def manual_login_wencai():
    import time
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
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
    with open(f"{TMP_DIR}/wencai_cookies.json", "w") as f:
        json.dump(cookies, f)
    print("登录信息已保存，下次将自动登录")
    driver.quit()
    

if __name__ == "__main__":
    manual_login_wencai()