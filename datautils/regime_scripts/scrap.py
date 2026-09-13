from env_setting import CHROME_DRIVER_PATH
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service

class Scrap:
    def __init__(self, date=None):
        self.url = ""

    def get_driver(self, headless=True):
        # 设置Chrome选项
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
        
        # 增强的headless配置
        if headless:
            chrome_options.add_argument('--headless=new')  # 使用新版headless模式
            chrome_options.add_argument('--window-size=1920,1080')  # 设置窗口大小
            chrome_options.add_argument('--disable-gpu')  # 禁用GPU加速
            chrome_options.add_argument('--disable-software-rasterizer')  # 禁用软件光栅化
            chrome_options.add_argument('--no-first-run')  # 跳过首次运行
            chrome_options.add_argument('--no-default-browser-check')  # 跳过默认浏览器检查
            chrome_options.add_argument('--disable-component-extensions-with-background-pages')  # 禁用带背景页的组件扩展
            chrome_options.add_argument('--disable-default-apps')  # 禁用默认应用
            chrome_options.add_argument('--disable-sync')  # 禁用同步
            chrome_options.add_argument('--disable-translate')  # 禁用翻译
            chrome_options.add_argument('--metrics-recording-only')  # 仅记录指标
            chrome_options.add_argument('--safebrowsing-disable-auto-update')  # 禁用安全浏览自动更新
            chrome_options.add_argument('--disable-background-networking')  # 禁用后台网络
            chrome_options.add_argument('--disable-client-side-phishing-detection')  # 禁用客户端钓鱼检测
            chrome_options.add_argument('--disable-hang-monitor')  # 禁用挂起监视器
            chrome_options.add_argument('--disable-prompt-on-repost')  # 禁用重新发布提示
            chrome_options.add_argument('--disable-domain-reliability')  # 禁用域可靠性
            chrome_options.add_argument('--disable-features=AudioServiceOutOfProcess')  # 禁用音频服务进程外功能
            chrome_options.add_argument('--disable-features=IsolateOrigins,site-per-process')  # 禁用站点隔离
            chrome_options.add_argument('--disable-ipc-flooding-protection')  # 禁用IPC洪水保护
            chrome_options.add_argument('--disable-renderer-backgrounding')  # 禁用渲染器后台处理
            chrome_options.add_argument('--disable-backgrounding-occluded-windows')  # 禁用被遮挡窗口的后台处理
            chrome_options.add_argument('--disable-field-trial-config')  # 禁用字段试验配置
            chrome_options.add_argument('--disable-back-forward-cache')  # 禁用前进/后退缓存

        service = Service(executable_path=CHROME_DRIVER_PATH)
        driver = webdriver.Chrome(options=chrome_options, service=service)
        
        # 反检测：移除webdriver属性
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                })
            """
        })
        
        return driver

