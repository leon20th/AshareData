# AshareData

A 股本地数据层：行情数据更新、爬取数据（涨停/跌停/龙虎榜）、特征构建与训练/推理用 DataLoader。

## 目录结构

```
AshareData/
├── dataloader.py                  # 训练/推理数据出口（parquet → numpy 样本）
├── paths.py                       # 路径寻址（以本包位置为基准自动定位）
├── datautils/
│   ├── dataloaders/feature_build/ # 特征构建（base_feature / 涨跌停 / 市场统计 / 龙虎榜 / 横截面排名 / 股性 traits / 开盘啦事件结构 / 新闻快讯计数）
│   ├── kline_scripts/             # 行情构建（v2=quick_kline / tushare 版=tushare_kline）/ 15分钟线 / 停牌名单 / update_all 统一入口
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
| `kline_data/daily_kline_v2/` | 日线 v2 CSV（未复权原值 + 事件因子，**读时复权**；2026-09-28 起消费链已切 ts 版，本目录留档） |
| `kline_data/daily_kline_ts/` | **当前日线主源**：tushare 版 CSV（未复权原值 + 复权因子 aux，**读时复权**；token 见 `.keys/.tushare_token`） |
| `kline_data/m15_kline/` | 15 分钟线 CSV（老库；2026-09-28 起消费链已切 m15_kline_ts） |
| `kline_data/m15_kline_ts/` | **当前 m15 主源**：tushare 版（不复权原值；历史由旧库换算重建 + 去重修复；增量补齐按缺口日期分流——近 ~62 交易日缺口走新浪，更早缺口走 baostock 小段；源缺日期记入 `_aux/m15_unavailable.json` 后各轮自动跳过；stk_mins 独立权限未开通） |
| `kline_data/info/` | 交易信息（如 `notrade_yet.csv` 停牌记录） |
| `scrap_data/zhangtingban/`、`scrap_data/dietingban/` | 涨停板 / 跌停板抓取产物（xlsx） |
| `scrap_data/longhu/` | 龙虎榜抓取产物（json） |
| `scrap_data/kaipanla/` | 开盘啦题材榜抓取产物（json） |
| `built_data/base_feature/` | 特征产物 `<code>.parquet` + `meta.json` |
| `built_data/*.parquet` | 涨跌停 / 市场统计 / 龙虎榜 / 横截面排名 / 股性 traits（t_*）/ 事件结构（ev_*、kp_*）/ 新闻快讯计数（nw_*）特征 |

## 快速使用

```bash
# 全量更新（v2 日线 / m15 / 停牌名单 / 新闻 / 榜单 / 特征，含重试与告警）
python AshareData/datautils/update_all.py

# tushare 版日K（增量；--probe 连通/权限自检、--latest 库状态问答）
python AshareData/datautils/kline_scripts/tushare_kline.py

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

- Python 3.9+（依赖只设下限，`pip` 自动选择与当前解释器兼容的版本，不要求固定 Python 版本；仅 `pandas` 带 `<3` 上限，以保证各机器写出的数据 dtype 一致）；依赖清单见 `requirements.txt`（`setup.sh` 自动安装）：`pandas`、`pyarrow`、`polars`、`numpy`、`tqdm`、`selenium`、`baostock`、`openpyxl`、`pypinyin`（可选）、`tushare`、`torch`/`torchvision`（滑块定位模型）、`huggingface-hub`（外部模型下载）。
- **chromedriver**：爬虫用；`setup.sh` 安装到 `~/.asharedata_deps/`，按 环境变量 `CHROME_DRIVER_PATH` → 该目录 → 系统路径 → `PATH` 依次探测。
- **外部模型**：`Fin-Retriever-base`（中文金融检索 BERT，sentence-transformers 格式，768 维，用于涨停原因 / 概念文本嵌入）由 `setup.sh` 自动下载到 `models/Fin-Retriever-base`（不入库）；运行期经 `AshareData.paths.FIN_RETRIEVER_DIR` 寻址。
- **子模块**：首次克隆后执行 `git submodule update --init`。

## 说明

- 所有路径经 `AshareData.paths` 自动寻址，不依赖外部环境变量或工程配置。
- 交易元数据（股票代码表 / 交易日历）随仓库携带于 `utils/exchanges_meta/`。
- `dataset/` 与运行期临时文件不入库，由使用方自行准备 / 生成。
- **问财登录**：账号密码存 `AshareData/.cache/tsh_account.json`（`setup.sh` 写入，不进 git）。
  若站点要求**短信二次验证**（风险处置页），登录会在命令行提示输入手机收到的验证码（无法绕过）；
  成功后 cookies 写入 `.cache/wencai_cookies.json`，之后各抓取任务直接复用。
  手动刷新 cookies：`python AshareData/utils/tsh_utils/login_util.py`（默认有头；`--headless` 无头；`--manual` 纯手动）。
  注：非交互终端（无人值守）不会等待输入，会直接以匿名会话继续。
  滑块验证码图取不到时（`captcha.10jqka.com.cn`）会自动重试，连续失败会明确提示并建议改用 `--manual`；
  `--manual` 会等你手动完成（含短信验证）并自动保存 cookies（最多 5 分钟）。
  登录按“验证阶段”循环处理：滑块与短信“安全验证”可**任意顺序/重复出现**（如 滑块→短信→再滑块），
  逐阶段处理直到主页面“登录”入口消失即判成功并写 cookies；被服务器拒绝（如密码错误）会立即停止防锁号。
  **登录失败时会提示改用浏览器手动登录**：可跑 `--manual`（脚本代劳），或自行在浏览器登录
  https://www.iwencai.com 后导出 cookies 覆盖 `.cache/wencai_cookies.json`——该文件兼容
  EditThisCookie 导出的 JSON 列表、`{"名":"值"}` 对象、或 `"k=v; k2=v2"` 字符串三种格式。
