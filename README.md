# AshareData

A 股本地数据层：行情数据更新、爬取数据（涨停/跌停/龙虎榜）、特征构建与训练/推理用 DataLoader。

## 目录结构

```
AshareData/
├── dataloader.py                  # 训练/推理数据出口（parquet → numpy 样本）
├── paths.py                       # 路径寻址（以本包位置为基准自动定位）
├── datautils/
│   ├── dataloaders/feature_build/ # 特征构建（base_feature / 涨跌停 / 市场统计 / 龙虎榜 / 横截面排名）
│   ├── kline_scripts/             # 行情更新（日线 / 15分钟线、停牌检测）
│   └── regime_scripts/            # 爬虫（涨停板 / 跌停板 / 龙虎榜）
└── utils/
    ├── exchanges_utils/           # 交易日历、股票代码 ↔ 名称检索
    ├── exchanges_meta/            # 股票代码表、交易日历（随仓库携带）
    ├── kline_data_utils/          # baostock 行情查询
    ├── paddle_ocr_utils/          # 表格图片 OCR
    ├── tsh_utils/                 # 同花顺/问财登录（含滑块验证码）
    └── slide_captcha_model/       # 子模块：滑块定位模型
```

## 数据目录（`dataset/`，不入库）

| 目录 | 内容 |
| --- | --- |
| `kline_data/daily_kline/` | 日线 CSV（如 `sh.600000.csv`） |
| `kline_data/m15_kline/` | 15 分钟线 CSV |
| `kline_data/info/` | 交易信息（如 `notrade_yet.csv` 停牌记录） |
| `scrap_data/zhangtingban/`、`scrap_data/dietingban/` | 涨停板 / 跌停板抓取产物（xlsx） |
| `scrap_data/longhu/` | 龙虎榜抓取产物（json） |
| `built_data/base_feature/` | 特征产物 `<code>.parquet` + `meta.json` |
| `built_data/*.parquet` | 涨跌停 / 市场统计 / 龙虎榜 / 横截面排名特征 |

## 快速使用

```bash
# 更新行情（baostock）
python AshareData/datautils/kline_scripts/update_kline.py

# 增量重建特征（base_feature 及派生 parquet）
python AshareData/datautils/dataloaders/feature_build/build_all_feature.py

# 更新交易日历（写入 utils/exchanges_meta/trade_date_list.json）
python -c "from AshareData.utils.exchanges_utils import a_open; a_open.update_a_open()"
```

```python
from AshareData.paths import BASE_FEATURE_DIR
from AshareData.dataloader import KlineDataConfig, build_train_and_val_dataloaders
```

## 依赖与外部文件

- Python 3.10+；主要三方库：`pandas`、`pyarrow`、`polars`、`numpy`、`tqdm`、`selenium`、`baostock`、`openpyxl`、`akshare`（更新日历）、`pypinyin`（可选）、`torch`/`torchvision`（滑块模型与 OCR）。
- **chromedriver**：爬虫用；按 环境变量 `CHROME_DRIVER_PATH` → 常见安装路径 → `PATH` 依次探测。
- **子模块**：首次克隆后执行 `git submodule update --init`。

## 说明

- 所有路径经 `AshareData.paths` 自动寻址，不依赖外部环境变量或工程配置。
- 交易元数据（股票代码表 / 交易日历）随仓库携带于 `utils/exchanges_meta/`。
- `dataset/` 与运行期临时文件不入库，由使用方自行准备 / 生成。
