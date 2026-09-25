#!/usr/bin/env bash
# AshareData 环境安装：
#   1) 同花顺账号密码 + 扶摇 API Key → 写入本地缓存（已配置则跳过；可用环境变量非交互传入）
#   2) 安装 Python 依赖（requirements.txt）
#   3) 安装 Chrome + chromedriver（macOS / Linux 自动分支）
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
TMP_DIR="$ROOT/.cache"
ACCOUNT_FILE="$TMP_DIR/tsh_account.json"
FUYAO_KEY_FILE="$ROOT/.keys/.fuyao_api.json"
DEPS_DIR="$HOME/.asharedata_deps"
OS="$(uname -s)"

stty sane 2>/dev/null || true   # 修复上次中断（Ctrl-C 等）可能残留的终端状态（如无回显）

# ---------------- 1. 本地凭据（同花顺账号 / 扶摇 API Key） ----------------
# 「是否已配置」用 python3 解析 JSON 判定；解释器缺失时命令失败 → 视为未配置，回退交互输入
# ---- 同花顺账号 ----
TSH_CACHED=""
if [ -n "${TSH_USER:-}" ] && [ -n "${TSH_PASS:-}" ]; then
    echo "==> 使用环境变量 TSH_USER/TSH_PASS 写入账号"
elif python3 -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); sys.exit(0 if d.get("username") and d.get("password") else 1)' "${ACCOUNT_FILE}" 2>/dev/null; then
    echo "==> 已配置同花顺账号，跳过输入（如需更换：删除 ${ACCOUNT_FILE} 后重跑）"
    TSH_CACHED=1
fi

if [ -z "${TSH_CACHED}" ]; then
    if [ -z "${TSH_USER:-}" ] || [ -z "${TSH_PASS:-}" ]; then
        echo "==> 配置同花顺/问财账号，用于抓取涨停板和龙虎榜等（仅写入本地缓存 ${ACCOUNT_FILE}）"
        read -r -p "账号: " TSH_USER || { echo "读取输入失败：请在交互式终端中运行" >&2; exit 1; }
        read -r -p "密码: " TSH_PASS || { echo "读取密码失败" >&2; exit 1; }
    fi
    mkdir -p "${TMP_DIR}"
    printf '{"username": "%s", "password": "%s"}\n' "${TSH_USER}" "${TSH_PASS}" > "${ACCOUNT_FILE}"
    chmod 600 "${ACCOUNT_FILE}"
    echo "已写入 ${ACCOUNT_FILE}"
fi

# ---- 扶摇（fuyao）API Key ----
# 读取端见 datautils/kline_scripts/quick_kline.py 的 _sess()：取第一个长度 >10 的字符串值，
# 故字段名不敏感（此处用 api_key），但长度必须 >10 才会被识别。
FUYAO_KEY="${FUYAO_API_KEY:-}"
FUYAO_CACHED=""
if [ -n "${FUYAO_KEY}" ]; then
    echo "==> 使用环境变量 FUYAO_API_KEY 写入扶摇 API Key"
elif python3 -c 'import json,sys; d=json.load(open(sys.argv[1],encoding="utf-8")); sys.exit(0 if any(isinstance(v,str) and len(v)>10 for v in d.values()) else 1)' "${FUYAO_KEY_FILE}" 2>/dev/null; then
    echo "==> 已配置扶摇 API Key，跳过输入（如需更换：删除 ${FUYAO_KEY_FILE} 后重跑）"
    FUYAO_CACHED=1
else
    echo "==> 配置扶摇数据接口 API Key（fuyao.aicubes.cn，用于日K v2 行情）"
    read -r -p "API Key（直接回车跳过）: " FUYAO_KEY || { echo "读取输入失败：请在交互式终端中运行" >&2; exit 1; }
fi

if [ -z "${FUYAO_CACHED}" ]; then
    if [ -z "${FUYAO_KEY}" ]; then
        echo "注意：未配置扶摇 API Key，日K v2 行情相关脚本将不可用（可稍后重跑本脚本）" >&2
    elif [ "${#FUYAO_KEY}" -le 10 ]; then
        echo "注意：API Key 长度需大于 10（读取端会忽略更短的值），本次未写入" >&2
    else
        mkdir -p "${ROOT}/.keys"
        printf '{"api_key": "%s"}\n' "${FUYAO_KEY}" > "${FUYAO_KEY_FILE}"
        chmod 600 "${FUYAO_KEY_FILE}"
        echo "已写入 ${FUYAO_KEY_FILE}"
    fi
fi

# ---------------- 2. Python 依赖 ----------------
PYTHON="${PYTHON:-$(command -v python || command -v python3 || echo python3)}"
echo "==> Python 依赖（${PYTHON}）"
if "$PYTHON" -c "import pandas, numpy, polars, pyarrow, openpyxl, requests, tqdm, bs4, lxml, selenium, baostock, pypinyin, torch, torchvision, PIL" 2>/dev/null; then
    echo "依赖已满足，跳过安装"
elif ! "$PYTHON" -m pip install -r "$ROOT/requirements.txt"; then
    echo "pip 安装失败。若报 externally-managed-environment（系统 Python 限制），请指定虚拟环境的解释器重跑：" >&2
    echo "  PYTHON=/path/to/venv/bin/python ./setup.sh" >&2
    exit 1
fi

# ---------------- 3. Chrome + chromedriver ----------------
# chromedriver 从 Chrome for Testing 下载；版本优先对齐已安装 Chrome，无对应包时回退最新稳定版
install_chromedriver() {
    local deps_dir="$1" platform="$2" chrome_ver="$3"
    local zip="$deps_dir/chromedriver-$platform.zip"
    local cft="https://storage.googleapis.com/chrome-for-testing-public"
    local bin="$deps_dir/chromedriver-$platform/chromedriver"
    # 已装且与本机 Chrome 大版本一致 → 跳过下载；否则每次重跑都要联网（离线会失败）且白下 9MB
    if [ -x "$bin" ] && "$bin" --version 2>/dev/null | grep -q "ChromeDriver ${chrome_ver%%.*}."; then
        echo "chromedriver 已就绪（大版本 ${chrome_ver%%.*} 匹配），跳过下载"
        return 0
    fi
    mkdir -p "$deps_dir"
    if ! curl -fL --max-time 300 -o "$zip" "$cft/$chrome_ver/$platform/chromedriver-$platform.zip"; then
        chrome_ver="$(curl -fsSL --max-time 60 \
            https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions.json \
            | python3 -c 'import sys, json; print(json.load(sys.stdin)["channels"]["Stable"]["version"])')"
        curl -fL --max-time 300 -o "$zip" "$cft/$chrome_ver/$platform/chromedriver-$platform.zip"
        echo "注意：回退到 Chrome for Testing ${chrome_ver}（大版本可能与本机 Chrome 不同）" >&2
    fi
    rm -rf "$deps_dir/chromedriver-$platform"
    unzip -q "$zip" -d "$deps_dir"
    rm -f "$zip"
    chmod +x "$deps_dir/chromedriver-$platform/chromedriver"
    if [ "$OS" = "Darwin" ]; then
        xattr -d com.apple.quarantine "$deps_dir/chromedriver-$platform/chromedriver" 2>/dev/null || true
    fi
    echo "chromedriver 已安装: $deps_dir/chromedriver-$platform/chromedriver"
}

if [ "$OS" = "Darwin" ]; then
    # ================= macOS =================
    echo "==> macOS：安装 Google Chrome"
    if [ ! -d "/Applications/Google Chrome.app" ]; then
        command -v brew >/dev/null || { echo "请先安装 Homebrew: https://brew.sh" >&2; exit 1; }
        brew install --cask google-chrome
    else
        echo "Chrome 已安装，跳过"
    fi
    CHROME_VER="$("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --version | awk '{print $NF}')"
    case "$(uname -m)" in
        arm64) PLATFORM="mac-arm64" ;;
        *)     PLATFORM="mac-x64" ;;
    esac
    install_chromedriver "$DEPS_DIR" "$PLATFORM" "$CHROME_VER"
else
    # ================= Linux（Debian/Ubuntu，含 WSL） =================
    echo "==> Linux：安装 Google Chrome"
    if ! command -v google-chrome >/dev/null; then
        ARCH="$(dpkg --print-architecture)"
        DEB="/tmp/google-chrome-stable_current_${ARCH}.deb"
        curl -fL --max-time 600 -o "$DEB" "https://dl.google.com/linux/direct/google-chrome-stable_current_${ARCH}.deb"
        sudo apt-get install -y "$DEB"
    else
        echo "Chrome 已安装，跳过"
    fi
    command -v unzip >/dev/null || sudo apt-get install -y unzip
    CHROME_VER="$(google-chrome --version | awk '{print $NF}')"
    install_chromedriver "$DEPS_DIR" "linux64" "$CHROME_VER"
fi

echo
echo "完成。"
