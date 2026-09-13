#!/usr/bin/env bash
# AshareData 环境安装：
#   1) 交互输入同花顺账号密码 → 写入本地缓存
#   2) 安装 Python 依赖（requirements.txt）
#   3) 安装 Chrome + chromedriver（macOS / Linux 自动分支）
set -euo pipefail

cd "$(dirname "$0")"
ROOT="$(pwd)"
TMP_DIR="$ROOT/.cache"
ACCOUNT_FILE="$TMP_DIR/tsh_account.json"
DEPS_DIR="$HOME/.asharedata_deps"
OS="$(uname -s)"

# ---------------- 1. 账号密码 → 本地缓存 ----------------
echo "==> 配置同花顺/问财账号，用于抓取涨停板和龙虎榜等（仅写入本地缓存 ${ACCOUNT_FILE}）"
read -r -p "账号: " TSH_USER
read -r -s -p "密码: " TSH_PASS; echo
read -r -s -p "确认密码: " TSH_PASS2; echo
[ "$TSH_PASS" = "$TSH_PASS2" ] || { echo "两次密码不一致" >&2; exit 1; }
mkdir -p "$TMP_DIR"
printf '{"username": "%s", "password": "%s"}\n' "$TSH_USER" "$TSH_PASS" > "$ACCOUNT_FILE"
chmod 600 "$ACCOUNT_FILE"
echo "已写入 $ACCOUNT_FILE"

# ---------------- 2. Python 依赖 ----------------
PYTHON="${PYTHON:-$(command -v python || command -v python3 || echo python3)}"
echo "==> 安装 Python 依赖（$PYTHON）"
"$PYTHON" -m pip install -r "$ROOT/requirements.txt"
echo "Python 依赖已安装"

# ---------------- 3. Chrome + chromedriver ----------------
# chromedriver 从 Chrome for Testing 下载；版本优先对齐已安装 Chrome，无对应包时回退最新稳定版
install_chromedriver() {
    local deps_dir="$1" platform="$2" chrome_ver="$3"
    local zip="$deps_dir/chromedriver-$platform.zip"
    local cft="https://storage.googleapis.com/chrome-for-testing-public"
    mkdir -p "$deps_dir"
    if ! curl -fL --max-time 300 -o "$zip" "$cft/$chrome_ver/$platform/chromedriver-$platform.zip"; then
        chrome_ver="$(curl -fsSL --max-time 60 \
            https://googlechromelabs.github.io/chrome-for-testing/last-known-good-versions.json \
            | python3 -c 'import sys, json; print(json.load(sys.stdin)["channels"]["Stable"]["version"])')"
        curl -fL --max-time 300 -o "$zip" "$cft/$chrome_ver/$platform/chromedriver-$platform.zip"
        echo "注意：回退到 Chrome for Testing $chrome_ver（大版本可能与本机 Chrome 不同）" >&2
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
