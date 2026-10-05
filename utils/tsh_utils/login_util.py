import json
import os
import re
import sys
import tqdm
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
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


def _wencai_logged_in(driver, captcha_element=None, base_cookie_names=None):
    """登录成功判定：主文档里“登录”入口消失（页面改为展示用户信息）即视为已登录。

    比 cookie 差异可靠得多——提交登录时站点会下发若干与登录无关的 cookie，仅凭“有无新
    cookie”会把未登录误判为成功（实测会）。判定时临时切到主文档，随后切回登录 iframe（若仍在）。
    """
    try:
        driver.switch_to.default_content()
    except Exception:
        return False
    try:
        entries = (driver.find_elements(By.CSS_SELECTOR, 'span.login')
                   or driver.find_elements(By.CSS_SELECTOR, '.login_btn.nav_word'))
        gone = not any(e.is_displayed() for e in entries)
    except Exception:
        gone = False
    try:
        driver.switch_to.frame(driver.find_element(By.ID, 'login_iframe'))
    except Exception:
        pass
    return gone


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


def _account_format_rejected(uname):
    """站点新版登录 JS 不允许「手机号/邮箱 + 密码」登录；命中即清空用户名并中止提交（不发任何请求）。

    正则与提示语来自 upass 登录页 phone_format_reg.js
    （account_format_error_msg='暂不支持手机号/邮箱+密码登录'）及 main.js 提交处理器中的同名判断。
    命中时提交会在前端被中止 → 既没有 dologin 请求、也不会出现滑块验证码（本函数用于提前定位，
    避免白等 10s 并给出可操作提示）。返回提示语，未命中返回 None。
    """
    u = (uname or '').strip()
    phone = re.match(r'^1[3-9]\d{9}$', u)
    mail = re.match(r'^(\w+)([\-+.][\w]+)*@(\w[\-\w]*\.){1,5}([A-Za-z]){2,6}$', u)
    if phone or mail:
        kind = '手机号' if phone else '邮箱'
        return (f'登录用户名是{kind}格式 —— 问财已不支持「手机号/邮箱 + 密码」网页登录：'
                f'提交会被前端清空并中止，因此既不发登录请求、也不会出现滑块验证码。'
                f'请在 AshareData/.cache/tsh_account.json 里改用该账号的“用户名”（非手机号/邮箱），'
                f'或改用手动登录刷新 cookie（python AshareData/utils/tsh_utils/login_util.py）。')
    return None


def _ask_cli(prompt):
    """命令行读取用户输入；非交互终端（如无人值守的 update_all）直接返回 None，避免卡死。"""
    if not (sys.stdin and sys.stdin.isatty()):
        print('（当前不是交互终端，无法输入短信验证码）')
        return None
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print('（已取消输入）')
        return None


# 风险处置页（errorcode=-10510）短信验证元素，来自 upass 登录页 verification_v2.min.js
_VERIF_SMS_INPUT = '.verificationDOM_smsCodeInput'
_VERIF_SMS_BTN = '.verificationDOM_smsCodeBtn'
_VERIF_SMS_SUBMIT = '.verificationDOM_submit'
_VERIF_SMS_ERROR = '.verificationDOM_smsCodeError'


def _sms_verification_present(driver):
    try:
        return any(e.is_displayed() for e in driver.find_elements(By.CSS_SELECTOR, _VERIF_SMS_INPUT))
    except Exception:
        return False


def _slider_visible(driver):
    try:
        els = driver.find_elements(By.ID, 'slicaptcha')
        return bool(els) and els[0].is_displayed()
    except Exception:
        return False


def _wait_stage(driver, captcha_element, base_cookie_names, timeout=10):
    """轮询判定当前处于哪个验证阶段，返回 'sms' / 'slider' / 'ok' / None。

    短信验证（安全验证）优先于滑块：风险处置页会盖住滑块，且滑块元素失效时
    _wencai_logged_in 可能误判，必须先识别短信页。多阶段（滑块→短信→再滑块）由调用方循环处理。
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _sms_verification_present(driver):
            return 'sms'
        if _slider_visible(driver):
            return 'slider'
        if _wencai_logged_in(driver, captcha_element, base_cookie_names):
            return 'ok'
        time.sleep(0.5)
    return None


def _handle_sms_verification(driver, attempts=3):
    """站点要求短信二次验证：点“获取验证码”→ 命令行输入手机收到的验证码 → 提交。

    触发条件：密码登录返回 errorcode=-10510（风险处置页），页面插入
    `.verificationDOM`（verification_v2.min.js），含 获取验证码 / 验证码输入(6位) / 提交。
    验证码无法绕过，必须由本人从手机读取后在此输入。校验失败会重新提示输入（最多 attempts 次）。

    返回 True 表示已提交（结果由外层用 cookie 判定）；False 表示未完成，外层应转匿名会话
    （注意：此时**不要**再回到滑块重试，否则会重复触发风险页、页面会不断叠加验证面板）。
    """
    try:
        WebDriverWait(driver, 6).until(
            EC.visibility_of_element_located((By.CSS_SELECTOR, _VERIF_SMS_INPUT)))
    except Exception:
        return False
    # 尚未发送则先点“获取验证码”（倒计时中按钮显示 “60 s”，此时不重复点）
    try:
        btn = driver.find_element(By.CSS_SELECTOR, _VERIF_SMS_BTN)
        if btn.is_displayed() and '获取' in (btn.text or ''):
            try:
                btn.click()
            except Exception:
                driver.execute_script('arguments[0].click();', btn)
            print('已请求发送短信验证码，请查看短信')
    except Exception:
        pass

    for k in range(max(1, attempts)):
        code = _ask_cli('请输入收到的短信验证码（6 位数字，直接回车跳过并转匿名）：'
                        + (f'（第 {k + 1}/{attempts} 次）' if k else ''))
        if not code:
            print('未输入短信验证码 —— 跳过短信验证，以匿名会话继续抓取')
            return False
        try:
            box = driver.find_element(By.CSS_SELECTOR, _VERIF_SMS_INPUT)
            box.clear()
            box.send_keys(code)
            time.sleep(0.5)
            sub = driver.find_element(By.CSS_SELECTOR, _VERIF_SMS_SUBMIT)
            try:
                sub.click()
            except Exception:
                driver.execute_script('arguments[0].click();', sub)
            print('已提交短信验证码，等待校验…')
        except Exception as e:
            print(f'短信验证码提交失败: {type(e).__name__}: {e}')
            return False
        # 等校验结果：验证页消失即视为通过；出现错误提示则重新输入
        for _ in range(10):
            time.sleep(0.5)
            if not _sms_verification_present(driver):
                return True
            errs = [e.text.strip() for e in driver.find_elements(By.CSS_SELECTOR, _VERIF_SMS_ERROR)
                    if e.is_displayed() and (e.text or '').strip()]
            if errs:
                print(f'短信验证码校验未通过: {errs[0]}')
                break
        else:
            return True
    return True


def _save_wencai_cookies(driver):
    try:
        driver.switch_to.default_content()
    except Exception:
        pass
    cookies = driver.get_cookies()
    with open(f"{TMP_DIR}/wencai_cookies.json", "w") as f:
        json.dump(cookies, f)
    print(f'登录成功，cookies 已保存（{len(cookies)} 条）')
    return cookies


def _report_captcha_missing(driver):
    """提交登录后 10s 内未出现滑块验证码：落地现场（截图 + 元素 + 服务器结论）供排查。

    这条分支原先直接抛 TimeoutException 被外层吞掉，只打出一行带 chromedriver 原生栈的
    "登录流程失败: Message: ..."，看不出到底哪一步不对。常见成因：站点改版把该账号换成
    其它验证方式（图形码/手机）、登录被直接拒绝（风控/锁号）、或用户名是手机号/邮箱格式
    （前端直接中止提交，见 _account_format_rejected）—— 因此这里显式报告后仍以匿名会话
    继续，保持原有降级行为不变。
    """
    err = _wencai_login_error(driver)
    probes = {}
    for sel in ('#slicaptcha', '.sliderContainer', '#captcha > *', 'img.img-code',
                '#account_captcha', '.msg_box', '[class*=error]'):
        try:
            els = driver.find_elements(By.CSS_SELECTOR, sel)
            probes[sel] = f'{len(els)}/{sum(1 for e in els if e.is_displayed())}'
        except Exception:
            probes[sel] = 'ERR'
    shot = f'{TMP_DIR}/login_captcha_missing.png'
    try:
        driver.save_screenshot(shot)
    except Exception:
        shot = '（截图失败）'
    print(f'提交登录后 10s 内未出现滑块验证码 #slicaptcha —— 服务器结论：{err or "无"}；'
          f'元素(found/visible) {probes}；现场截图 {shot}；转匿名会话继续抓取')


def _normalize_cookies(data):
    """把各种手动导出的 cookie 形式归一成 driver.add_cookie 可用的字典列表。

    支持：① cookie 列表 JSON（EditThisCookie / devtools 导出）；
    ② {名: 值} 对象；③ 原始请求头字符串 "k=v; k2=v2"。无法识别的条目直接跳过。
    """
    if isinstance(data, str):
        out = []
        for part in data.replace('\n', ';').split(';'):
            if '=' in part:
                k, v = part.split('=', 1)
                k = k.strip()
                if k:
                    out.append({'name': k, 'value': v.strip()})
        return out
    if isinstance(data, dict):
        if data and all(isinstance(v, dict) for v in data.values()):
            data = list(data.values())                       # {"c1": {...}, "c2": {...}}
        else:
            return [{'name': str(k), 'value': str(v)} for k, v in data.items()]
    if not isinstance(data, list):
        return []
    out = []
    same_site = {'lax': 'Lax', 'strict': 'Strict', 'none': 'None',
                 'no_restriction': 'None', 'unspecified': None}
    for c in data:
        if not isinstance(c, dict) or not c.get('name'):
            continue
        d = {'name': c['name'], 'value': str(c.get('value', ''))}
        for k in ('domain', 'path'):
            if c.get(k):
                d[k] = c[k]
        if c.get('secure') is not None:
            d['secure'] = bool(c['secure'])
        if c.get('httpOnly') is not None:
            d['httpOnly'] = bool(c['httpOnly'])
        exp = c.get('expiry', c.get('expirationDate'))
        try:
            if exp:
                d['expiry'] = int(float(exp))
        except (TypeError, ValueError):
            pass
        ss = c.get('sameSite')
        if ss:
            v = same_site.get(str(ss).lower(), ss)
            if v:
                d['sameSite'] = v
        out.append(d)
    return out


def _load_wencai_cookies():
    """读取 wencai_cookies.json（兼容手动导出格式），返回可 add_cookie 的字典列表。"""
    with open(f"{TMP_DIR}/wencai_cookies.json", "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        return _normalize_cookies(json.loads(raw))
    except json.JSONDecodeError:
        return _normalize_cookies(raw)


_MANUAL_COOKIE_HINT = (
    '  登录未成功。可改用“浏览器手动登录 + cookies 落盘”两条路：\n'
    '    方式一（推荐，脚本代劳）：python AshareData/utils/tsh_utils/login_util.py --manual\n'
    '    方式二（手动放置）：浏览器登录 https://www.iwencai.com 后导出 cookies，\n'
    '      保存到 {tmp}/wencai_cookies.json（支持 EditThisCookie 导出的 JSON 列表、\n'
    '      {{"名":"值"}} 对象、或 "k=v; k2=v2" 字符串）。之后各抓取任务会优先读该文件直接复用。'
)


def _manual_cookie_hint():
    print(_MANUAL_COOKIE_HINT.format(tmp=TMP_DIR))


def login_wencai(driver):
    try:
        cookie = _load_wencai_cookies()
        driver.delete_all_cookies()
        loaded = 0
        for c in cookie:
            try:
                driver.add_cookie(c)
                loaded += 1
            except Exception:
                try:                                          # 兜底：仅 name/value（不带 domain/path）
                    driver.add_cookie({'name': c['name'], 'value': c['value']})
                    loaded += 1
                except Exception:
                    pass
        driver.refresh()
        print(f'使用cookie登录中（{loaded}/{len(cookie)} 条）')
    except FileNotFoundError:
        print(f'未找到 cookie 文件（首次登录正常）：{TMP_DIR}/wencai_cookies.json')
    except Exception as e:
        print(f'读取 cookie 文件失败（{type(e).__name__}: {e}），将走账号密码登录')

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

        with open(f"{TMP_DIR}/tsh_account.json", "r", encoding="utf-8") as f:
            account = json.load(f)

        fmt_msg = _account_format_rejected(account.get("username"))
        if fmt_msg:
            print(fmt_msg)
            return True

        # 确保“密码登录”面板已就绪：默认可能停在短信登录面板，或上一步 tab 点击未生效 →
        # 此时 #uname 虽在 DOM 中却不可交互（clear() 会抛 element not interactable）。重试一次切换。
        try:
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, 'uname')))
        except Exception:
            try:
                tab = driver.find_element(By.ID, 'to_account_login')
                try:
                    tab.click()
                except Exception:
                    driver.execute_script('arguments[0].click();', tab)
            except Exception:
                pass
            WebDriverWait(driver, 5).until(EC.element_to_be_clickable((By.ID, 'uname')))

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

        base_cookie_names = {c['name'] for c in driver.get_cookies()}
        _els = driver.find_elements(By.ID, 'slicaptcha')
        captcha_element = _els[0] if _els else None
        img_fail = 0
        ever_verified = False
        rejected = False
        stages = 8          # 最多处理的验证阶段数（滑块/短信各算一阶段，防死循环）

        while stages > 0:
            stages -= 1
            stage = _wait_stage(driver, captcha_element, base_cookie_names, timeout=10)
            if stage == 'ok':
                _save_wencai_cookies(driver)
                return True
            if stage == 'sms':
                # 安全验证（短信二次验证）：点“获取验证码”→ 命令行输入 → 提交；完后回到循环重新判定
                ever_verified = True
                if not _handle_sms_verification(driver):
                    return True
                for _ in range(20):
                    time.sleep(0.5)
                    if _wencai_logged_in(driver, captcha_element, base_cookie_names):
                        _save_wencai_cookies(driver)
                        return True
                    if not _sms_verification_present(driver):
                        break
                continue
            if stage == 'slider':
                ever_verified = True
                print(f'处理滑块验证码（剩余阶段预算 {stages}）…', flush=True)
                try:
                    slicaptcha_icon = driver.find_element(By.ID, 'slicaptcha-icon')
                    try:
                        slicaptcha_icon.click()
                    except Exception:
                        driver.execute_script('arguments[0].click();', slicaptcha_icon)
                    try:
                        WebDriverWait(driver, 1).until(
                            EC.visibility_of_element_located((By.ID, 'slicaptcha-warn-btn')))
                        warn_btn = driver.find_element(By.ID, 'slicaptcha-warn-btn')
                        try:
                            warn_btn.click()
                        except Exception:
                            driver.execute_script('arguments[0].click();', warn_btn)
                    except Exception:
                        pass
                    captcha_img = WebDriverWait(driver, 15).until(
                        EC.visibility_of_element_located((By.ID, 'slicaptcha-img')))
                    # 等图片真正加载完（naturalWidth>0）：取图失败时截图是空白，预测必错且白耗重试
                    try:
                        WebDriverWait(driver, 8).until(
                            lambda d: (d.execute_script('return arguments[0].naturalWidth', captcha_img) or 0) > 0)
                    except Exception:
                        img_fail += 1
                        print(f'滑块验证码图片未加载出来（captcha.10jqka.com.cn 取图失败，连续 {img_fail} 次）', flush=True)
                        if img_fail >= 3:
                            print('滑块图片连续多次取不到，停止重试 —— 请检查网络/稍后重跑，或改用 --manual 手动登录', flush=True)
                            break
                        continue
                    img_fail = 0
                    captcha_img.screenshot(f'{TMP_DIR}/captcha_image.png')
                    pred_x = predict_captcha(model, f'{TMP_DIR}/captcha_image.png')
                    from selenium.webdriver import ActionChains
                    drag_err = None
                    for _ in range(2):                      # 拖动偶发 element not interactable，重试一次
                        try:
                            slider = driver.find_element(By.ID, 'slider')
                            ActionChains(driver).click_and_hold(slider).move_by_offset(pred_x, 0).release().perform()
                            drag_err = None
                            break
                        except Exception as e:
                            drag_err = e
                            time.sleep(0.5)
                    if drag_err:
                        raise drag_err
                except Exception as cpt_err:
                    print(f'滑块处理失败: {cpt_err}', flush=True)
                time.sleep(2)
                _err = _wencai_login_error(driver)
                # -11400/-10507 都是“需要滑块验证”，属可重试；其它错误码视为最终拒绝，立即停止防锁号
                if _err and not (_err.startswith('-11400') or _err.startswith('-10507')):
                    print(f'登录被服务器拒绝（{_err}），停止重试，以匿名会话继续抓取')
                    rejected = True
                    break
                continue
            # 未检测到任何验证元素：优先判“被服务器拒绝”（防锁号），否则点一次提交再触发
            err = _wencai_login_error(driver)
            if err:
                print(f'登录被服务器拒绝（{err}），停止重试，以匿名会话继续抓取')
                rejected = True
                break
            buttons = [b for b in driver.find_elements(By.CSS_SELECTOR, '.submit_btn') if b.is_displayed()]
            if not buttons:
                print('未出现验证码且登录按钮不可见，停止重试，以匿名会话继续抓取')
                break
            try:
                buttons[0].click()
            except Exception:
                driver.execute_script('arguments[0].click();', buttons[0])

        if not ever_verified:
            _report_captcha_missing(driver)
        elif not rejected and not _sms_verification_present(driver) and not _slider_visible(driver):
            # 验证阶段都通过、当前无待处理验证且未被服务器拒绝 → 视为已登录，落 cookies
            _save_wencai_cookies(driver)
            try:
                driver.switch_to.default_content()
            except Exception:
                pass
            return True
    except Exception as e:
        print(f'登录流程失败: {e}，以匿名会话继续抓取')
        _manual_cookie_hint()
        try:
            driver.switch_to.default_content()
        except Exception:
            pass
        return True
    if not rejected:
        print('验证未完成（未在阶段预算内登录成功），以匿名会话继续抓取')
    _manual_cookie_hint()
    try:
        driver.switch_to.default_content()
    except Exception:
        pass
    return True


def login_wencai_interactive(headless=False):
    """交互式登录问财（命令行输入短信验证码），成功后写 cookies。

    流程：自动填账号密码（TMP_DIR/tsh_account.json）→ 自动过滑块验证码 →
    若站点要求短信二次验证（风险处置页），则提示“获取验证码”，由本人从手机读取后
    在命令行输入即可（验证码无法绕过）。成功后 cookies 写入 TMP_DIR/wencai_cookies.json，
    之后各抓取任务直接复用、无需再登录。

    用法:
      python AshareData/utils/tsh_utils/login_util.py            # 有头浏览器
      python AshareData/utils/tsh_utils/login_util.py --headless
    """
    from AshareData.datautils.regime_scripts.scrap import Scrap
    cookie_f = f'{TMP_DIR}/wencai_cookies.json'
    before = os.path.getmtime(cookie_f) if os.path.exists(cookie_f) else None
    driver = Scrap().get_driver(headless=headless)
    try:
        driver.get('https://www.iwencai.com/unifiedwap/home/index')
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, 'body')))
        login_wencai(driver)
    finally:
        try:
            driver.quit()
        except Exception:
            pass
    ok = os.path.exists(cookie_f) and os.path.getmtime(cookie_f) != before
    if ok:
        print('登录成功：cookies 已更新 → ' + cookie_f)
    else:
        print('登录未完成：cookies 未更新（将在抓取时以匿名会话继续）')
        _manual_cookie_hint()
    return ok


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
    print('请在浏览器里完成登录（账号密码 → 滑块 → 手机短信验证码）；')
    print('检测到登录成功后会自动保存 cookies（最多等待 5 分钟，可随时关闭浏览器提前结束）')
    saved = False
    for _ in tqdm.tqdm(range(300), desc='等待登录完成'):
        time.sleep(1)
        try:
            entry = (driver.find_elements(By.CSS_SELECTOR, 'span.login')
                     or driver.find_elements(By.CSS_SELECTOR, '.login_btn.nav_word'))
            if not any(e.is_displayed() for e in entry):     # 登录入口消失 → 已登录
                saved = True
                break
        except Exception:
            saved = True                                     # 浏览器已关闭
            break
    cookies = driver.get_cookies()
    with open(f"{TMP_DIR}/wencai_cookies.json", "w") as f:
        json.dump(cookies, f)
    print('登录信息已保存' if saved else '等待超时，仍保存了当前 cookies（可能未登录成功）')
    driver.quit()
    

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description='问财登录（账号密码 + 滑块；如需短信验证则在命令行输入验证码）')
    ap.add_argument('--headless', action='store_true', help='无头模式（默认有头，便于观察）')
    ap.add_argument('--manual', action='store_true', help='改为纯手动登录（打开浏览器等待人工完成并保存 cookies）')
    a = ap.parse_args()
    if a.manual:
        manual_login_wencai()
    else:
        sys.exit(0 if login_wencai_interactive(headless=a.headless) else 1)