"""AshareData 路径寻址：以本包在系统中的实际位置为基准，不依赖上层工程配置。

- 数据目录全部由 ASHARE_ROOT 派生（dataset/...）；
- 交易元数据（股票代码表/交易日历）随仓库携带在 utils/exchanges_meta；
- chromedriver 按 环境变量 → setup.sh 默认安装路径（随系统/架构） → PATH 依次探测。
"""
import os
import platform
import shutil
import sys

# ---- 本包位置（自动寻址起点）----
ASHARE_ROOT = os.path.dirname(os.path.abspath(__file__))

# ---- 数据目录 ----
DATASET_DIR = os.path.join(ASHARE_ROOT, 'dataset')
KLINE_DATA_DIR = os.path.join(DATASET_DIR, 'kline_data')
DAILY_KLINE_DIR = os.path.join(KLINE_DATA_DIR, 'daily_kline')
DAILY_KLINE_V2_DIR = os.path.join(KLINE_DATA_DIR, 'daily_kline_v2')   # 扶摇 v2: raw 主表+事件因子+读时复权
M15_KLINE_DIR = os.path.join(KLINE_DATA_DIR, 'm15_kline')
INFO_DIR = os.path.join(KLINE_DATA_DIR, 'info')
SCRAP_DATA_DIR = os.path.join(DATASET_DIR, 'scrap_data')
BUILT_DATA_DIR = os.path.join(DATASET_DIR, 'built_data')
BASE_FEATURE_DIR = os.path.join(BUILT_DATA_DIR, 'base_feature')

# ---- 外部文件 ----
META_DIR = os.path.join(ASHARE_ROOT, 'utils', 'exchanges_meta')   # 股票代码表/交易日历
TMP_DIR = os.path.join(ASHARE_ROOT, '.cache')   # 运行期缓存（账号/cookie/验证码截图，不进 git）

# ---- 业务默认值 ----
_BEGIN = '20200101'   # 历史数据默认起始日（无已有数据时的兜底）

# ---- chromedriver（默认位置与 setup.sh 的安装路径一致：~/.asharedata_deps；环境变量可覆盖） ----
if sys.platform == 'darwin':
    _CHROME_TAG = 'mac-arm64' if platform.machine() == 'arm64' else 'mac-x64'
else:
    _CHROME_TAG = 'linux64'

CHROME_DRIVER_PATH = (
    os.environ.get('CHROME_DRIVER_PATH')
    or next((p for p in (
        os.path.expanduser(f'~/.asharedata_deps/chromedriver-{_CHROME_TAG}/chromedriver'),
        '/usr/bin/chromedriver',
        '/usr/local/bin/chromedriver',
    ) if os.path.exists(p)), None)
    or shutil.which('chromedriver')
)
