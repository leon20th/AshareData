"""AshareData 路径寻址：以本包在系统中的实际位置为基准，不依赖上层工程配置。

- 数据目录全部由 ASHARE_ROOT 派生（dataset/...）；
- 交易元数据（股票代码表/交易日历）默认取同级工程的 utils/exchanges_meta；
- chromedriver 按 环境变量 → 常见路径 → PATH 依次探测。
"""
import os
import shutil

# ---- 本包位置（自动寻址起点）----
ASHARE_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- 数据目录 ----
DATASET_DIR = os.path.join(ASHARE_ROOT, 'dataset')
KLINE_DATA_DIR = os.path.join(DATASET_DIR, 'kline_data')
DAILY_KLINE_DIR = os.path.join(KLINE_DATA_DIR, 'daily_kline')
M15_KLINE_DIR = os.path.join(KLINE_DATA_DIR, 'm15_kline')
INFO_DIR = os.path.join(KLINE_DATA_DIR, 'info')
SCRAP_DATA_DIR = os.path.join(DATASET_DIR, 'scrap_data')
BUILT_DATA_DIR = os.path.join(DATASET_DIR, 'built_data')
BASE_FEATURE_DIR = os.path.join(BUILT_DATA_DIR, 'base_feature')

# ---- 外部文件 ----
META_DIR = os.path.join(os.path.dirname(ASHARE_ROOT), 'utils', 'exchanges_meta')   # 股票代码表/交易日历
TMP_DIR = os.path.join(os.path.dirname(ASHARE_ROOT), 'business_tmp_files', 'tsh')  # 爬虫运行期临时文件

# ---- chromedriver ----
CHROME_DRIVER_PATH = (
    os.environ.get('CHROME_DRIVER_PATH')
    or next((p for p in (
        os.path.expanduser('~/deps/chromedriver-linux64/chromedriver'),
        os.path.expanduser('~/Documents/workplace/deps/chromedriver-mac-arm64/chromedriver'),
        '/usr/bin/chromedriver',
        '/usr/local/bin/chromedriver',
    ) if os.path.exists(p)), None)
    or shutil.which('chromedriver')
)
