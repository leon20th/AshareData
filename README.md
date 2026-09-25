# AshareData

A 股本地数据层：行情数据更新、爬取数据（涨停/跌停/龙虎榜）、特征构建与训练/推理用 DataLoader。

## 目录结构

```
AshareData/
├── dataloader.py                  # 训练/推理数据出口（parquet → numpy 样本）
├── paths.py                       # 路径寻址（以本包位置为基准自动定位）
├── datautils/
│   ├── dataloaders/feature_build/ # 特征构建（base_feature / 涨跌停 / 市场统计 / 龙虎榜 / 横截面排名 / 股性 traits / 开盘啦事件结构）
│   ├── kline_scripts/             # 行情 v2 构建（quick_kline）/ 15分钟线 / 停牌名单 / update_all 统一入口
│   └── regime_scripts/            # 爬虫（涨停板 / 跌停板 / 龙虎榜 / 开盘啦题材榜）
└── utils/
    ├── exchanges_utils/           # 交易日历、股票代码 ↔ 名称检索
    ├── exchanges_meta/            # 股票代码表、交易日历（随仓库携带）
    ├── kline_data_utils/          # baostock 行情查询
    └── tsh_utils/                 # 同花顺/问财登录
        └── slide_captcha_model/   # 子模块：滑块定位模型
```

## 数据目录（`dataset/`，不入库）

| 目录 | 内容 |
| --- | --- |
| `kline_data/daily_kline_v2/` | 日线 v2 CSV（未复权原值 + 事件因子，**读时复权**；旧库 `daily_kline/` 已于 2026-09-26 撤除） |
| `kline_data/m15_kline/` | 15 分钟线 CSV |
| `kline_data/info/` | 交易信息（如 `notrade_yet.csv` 停牌记录） |
| `scrap_data/zhangtingban/`、`scrap_data/dietingban/` | 涨停板 / 跌停板抓取产物（xlsx） |
| `scrap_data/longhu/` | 龙虎榜抓取产物（json） |
| `scrap_data/kaipanla/` | 开盘啦题材榜抓取产物（json） |
| `built_data/base_feature/` | 特征产物 `<code>.parquet` + `meta.json` |
| `built_data/*.parquet` | 涨跌停 / 市场统计 / 龙虎榜 / 横截面排名 / 股性 traits（t_*）/ 事件结构（ev_*、kp_*）特征 |

## 快速使用

```bash
# 全量更新（v2 日线 / m15 / 停牌名单 / 新闻 / 榜单 / 特征，含重试与告警）
python AshareData/datautils/update_all.py

# 抓取开盘啦题材榜（增量补齐）
python AshareData/datautils/regime_scripts/scrap_kaipanla.py --update

# 增量重建特征（base_feature 及派生 parquet，含股性 traits 与开盘啦结构特征）
python AshareData/datautils/dataloaders/feature_build/build_all_feature.py

# 更新交易日历（写入 utils/exchanges_meta/trade_date_list.json）
python -c "from AshareData.utils.exchanges_utils import a_open; a_open.update_a_open()"
```

```python
from AshareData.paths import BASE_FEATURE_DIR
from AshareData.dataloader import KlineDataConfig, build_train_and_val_dataloaders
```

## 依赖与外部文件

- Python 3.10+；依赖清单见 `requirements.txt`（`setup.sh` 自动安装）：`pandas`、`pyarrow`、`polars`、`numpy`、`tqdm`、`selenium`、`baostock`、`openpyxl`、`pypinyin`（可选）、`torch`/`torchvision`（滑块定位模型）。
- **chromedriver**：爬虫用；`setup.sh` 安装到 `~/.asharedata_deps/`，按 环境变量 `CHROME_DRIVER_PATH` → 该目录 → 系统路径 → `PATH` 依次探测。
- **子模块**：首次克隆后执行 `git submodule update --init`。

## 说明

- 所有路径经 `AshareData.paths` 自动寻址，不依赖外部环境变量或工程配置。
- 交易元数据（股票代码表 / 交易日历）随仓库携带于 `utils/exchanges_meta/`。
- `dataset/` 与运行期临时文件不入库，由使用方自行准备 / 生成。
