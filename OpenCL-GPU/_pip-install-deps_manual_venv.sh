#!/bin/bash

# _pip-install-deps... — prepare Python environment for OpenCL GPU vanity-key generator
#
# Required packages (with minimum versions):
#   pyopencl     >= 2026.1
#   cryptography >= 3.1
#
# Steps:
#   0. Setup proxy (HTTP_PROXY/HTTPS_PROXY) if internet needed
#   1. Check if packages installed with required versions → exit OK
#   2. If python >= 3.12 → pip install --user (bootstrap pip via get-pip if missing)
#   3. Get Python 3.12+ binary → create venv → pip install packages
#   4. If nothing works → print manual fallback

set -e

echo "=== Required packages ==="
echo "  pyopencl     >= 2026.1"
echo "  cryptography >= 3.1"
echo ""

# Default proxy (set to empty to disable by default)
PROXY=""

# ============================================================
# Step 0: Proxy setup (if needed for downloads)
# ============================================================

# This script checks the following env vars for proxy config:
#   HTTP_PROXY, http_proxy, HTTPS_PROXY, https_proxy
# (checked in that order; first non-empty wins)
_PROXY="${HTTP_PROXY:-${http_proxy:-${HTTPS_PROXY:-${https_proxy:-}}}}"
if [ -n "$_PROXY" ]; then
    echo "Using proxy from environment: $_PROXY"
    export HTTP_PROXY="$_PROXY"
    export HTTPS_PROXY="$_PROXY"
    export http_proxy="$_PROXY"
    export https_proxy="$_PROXY"
    export CURL_PROXY="$_PROXY"
elif [ -n "$PROXY" ]; then
    echo "Using proxy from script default: $PROXY"
    export HTTP_PROXY="$PROXY"
    export HTTPS_PROXY="$PROXY"
    export http_proxy="$PROXY"
    export https_proxy="$PROXY"
    export CURL_PROXY="$PROXY"
fi

echo ""
# Step 1: Check if all packages already installed
# ============================================================

# Find any python3 on the system
if ! command -v python3 >/dev/null 2>&1; then
    echo "No python3 found."
    _PYTHON3=""
else
    _PYTHON3="$(command -v python3)"
    # Skip conda/miniforge environments — pyopencl hangs in conda/venv
    if echo "$_PYTHON3" | grep -qi "conda\|miniforge\|anaconda"; then
        echo "Skipping conda/miniforge python3 at: $_PYTHON3"
        _PYTHON3=""
    else
        echo "Found python3: $_PYTHON3 ($($_PYTHON3 --version 2>&1))"
    fi
fi

# Check if all 3 packages import successfully with required versions
if [ -n "$_PYTHON3" ]; then
    echo "Checking packages: pyopencl, cryptography..."
    _VER_TMP="$(mktemp)"
    "$_PYTHON3" -c "
import sys
import pyopencl
import cryptography

pyocl_ver = getattr(pyopencl, '__version__', '0')
crypt_v = cryptography.__version__

ok = True

# Check pyopencl >= 2026.1
if pyocl_ver:
    v = [int(x) for x in pyocl_ver.split('.') if x.isdigit()]
    if v < [2026, 1]:
        ok = False

# Check cryptography >= 3.1
cv = [int(x) for x in crypt_v.split('.')[:2]]
if cv < [3, 1]:
    ok = False

# Report versions
print(f'pyopencl: {pyocl_ver}  cryptography: {crypt_v}')
if ok:
    print('ALL_OK')
else:
    print('NEEDS_INSTALL')
" > "$_VER_TMP" 2>&1 || true

    if grep -q "ALL_OK" "$_VER_TMP"; then
        echo "All dependencies already satisfied (versions OK):"
        echo "  $(grep -v ALL_OK "$_VER_TMP")"
        echo "Environment OK — nothing to do."
        rm -f "$_VER_TMP"
        exit 0
    fi
    echo "Required packages (pyopencl >= 2026.1, cryptography >= 3.1) not found."
    rm -f "$_VER_TMP"
fi

# --- Must not run as root ---
if [ "$EUID" = "0" ]; then
    echo "ERROR: must NOT run as root."
    exit 1
fi

# ============================================================
# Step 2: pip install --user (if python3 >= 3.12)
# ============================================================

if [ -n "$_PYTHON3" ]; then
    _PYPY="$(echo "$($_PYTHON3 --version 2>&1)" | grep -oP '(?<=Python )\d+\.\d+')"
    _PY_MAJOR="$(echo "$_PYPY" | cut -d. -f1)"
    _PY_MINOR="$(echo "$_PYPY" | cut -d. -f2)"
    if [ "$_PY_MAJOR" = "3" ] && [ "$_PY_MINOR" -ge 12 ]; then
        echo ""
        echo "--- Step 2: pip install --user ---"

        # Bootstrap pip if missing
        if ! "$_PYTHON3" -m pip --version >/dev/null 2>&1; then
            echo "pip not found — downloading get-pip.py..."
            if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
                echo "ERROR: need curl or wget"; exit 1
            fi
            mkdir -p "$HOME/.local/bin"
            if command -v wget >/dev/null 2>&1; then
                if [ -n "$_PROXY" ]; then
                    wget --proxy=on "https://bootstrap.pypa.io/get-pip.py" -O "$HOME/.local/get-pip.py"
                else
                    wget "https://bootstrap.pypa.io/get-pip.py" -O "$HOME/.local/get-pip.py"
                fi
            else
                if [ -n "$_PROXY" ]; then
                    curl --proxy "$_PROXY" -L "https://bootstrap.pypa.io/get-pip.py" -o "$HOME/.local/get-pip.py"
                else
                    curl -L "https://bootstrap.pypa.io/get-pip.py" -o "$HOME/.local/get-pip.py"
                fi
            fi
            if [ ! -s "$HOME/.local/get-pip.py" ]; then
                echo "ERROR: failed to download get-pip.py"; exit 1
            fi
            "$_PYTHON3" "$HOME/.local/get-pip.py" --user 2>&1 | tail -3
            rm -f "$HOME/.local/get-pip.py"
        fi

        if "$_PYTHON3" -m pip --version >/dev/null 2>&1; then
            echo "Installing packages via pip --user..."
            _PIP_OPTS="--user --quiet"
            if [ -n "$_PROXY" ]; then
                _PIP_OPTS="$_PIP_OPTS --proxy $_PROXY"
            fi
            if "$_PYTHON3" -m pip install $_PIP_OPTS \
                "pyopencl>=2026.1" "cryptography>=3.1" 2>&1; then
                echo "pip --user succeeded:"
                "$_PYTHON3" -c "
import pyopencl, cryptography
print('  pyopencl:    ' + getattr(pyopencl, '__version__', '?'))
print('  cryptography: ' + cryptography.__version__)
"
                exit 0
            fi
            echo "pip --user failed — will try venv."
        fi
    fi
fi

# ============================================================
# Step 3: Create venv + pip install
# ============================================================
echo ""
echo "--- Step 3: venv + pip install ---"

# Determine which python to use for venv
# If system python >= 3.12 → use it directly
# If system python < 3.12 → download Python 3.12 binary
_VENV_PYTHON=""
if [ -n "$_PYTHON3" ]; then
    _PYPY="$(echo "$($_PYTHON3 --version 2>&1)" | grep -oP '(?<=Python )\d+\.\d+')"
    _PY_MAJOR="$(echo "$_PYPY" | cut -d. -f1)"
    _PY_MINOR="$(echo "$_PYPY" | cut -d. -f2)"
    if [ "$_PY_MAJOR" = "3" ] && [ "$_PY_MINOR" -ge 12 ]; then
        _VENV_PYTHON="$_PYTHON3"
        echo "Using system python3 $_PYTHON3 ($($_PYTHON3 --version 2>&1)) for venv (>= 3.12)."
    fi
fi

# System python < 3.12 — need to download
if [ -z "$_VENV_PYTHON" ]; then
    _PY312="$HOME/.local/python312/bin/python3"

    if [ -x "$_PY312" ] && "$_PY312" --version 2>&1 | grep -qP 'Python 3\.1[2-9]'; then
        _VENV_PYTHON="$_PY312"
        echo "Using already-downloaded Python 3.12: $_PY312 ($($_PY312 --version 2>&1))"
    else
        echo "System python < 3.12 — downloading Python 3.12 binary..."

        if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
            echo "ERROR: curl or wget required for downloading Python 3.12"; exit 1
        fi

        _ARCH="$(uname -m)"
        echo "Downloading Python 3.12 standalone binary for $_ARCH..."

        # Get latest release tag
        _LATEST_TAG="$(curl --proxy "$_PROXY" -L 'https://api.github.com/repos/astral-sh/python-build-standalone/releases?per_page=5' | grep '"tag_name":' | head -1 | sed -E 's/.*"([^"]+)".*/\1/')"
        if [ -z "$_LATEST_TAG" ]; then
            echo "ERROR: could not get latest release tag"; exit 1
        fi
        echo "  Latest release: $_LATEST_TAG"

        # Find Python 3.12 install_only asset
        _URL=""
        case "$_ARCH" in
            x86_64)
                _URL="$(curl --proxy "$_PROXY" -sL "https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/$_LATEST_TAG" \
                    | grep '"browser_download_url"' \
                    | grep 'install_only' \
                    | grep '3\.12\.' \
                    | grep 'x86_64-unknown-linux-gnu' \
                    | grep -v 'musl' \
                    | grep -v 'stripped' \
                    | grep -v 'v2\|v3\|v4' \
                    | head -1 \
                    | sed 's/.*"browser_download_url": "//' \
                    | sed 's/"$//')"
                ;;
            aarch64)
                _URL="$(curl --proxy "$_PROXY" -sL "https://api.github.com/repos/astral-sh/python-build-standalone/releases/tags/$_LATEST_TAG" \
                    | grep '"browser_download_url"' \
                    | grep 'install_only' \
                    | grep '3\.12\.' \
                    | grep 'aarch64' \
                    | grep 'linux-gnu' \
                    | grep -v 'musl' \
                    | grep -v 'stripped' \
                    | head -1 \
                    | sed 's/.*"browser_download_url": "//' \
                    | sed 's/"$//')"
                ;;
            *)
                echo "ERROR: Unsupported architecture: $_ARCH"; exit 1
                ;;
        esac

        if [ -z "$_URL" ]; then
            echo "ERROR: could not find a suitable Python 3.12 binary for $_ARCH in release $_LATEST_TAG"
            exit 1
        fi
        echo "  URL: $_URL"

        _TARBALL="/tmp/python312-install_only.tar.gz"
        if command -v wget >/dev/null 2>&1; then
            echo "Downloading Python 3.12 with wget (resumable)..."
            if [ -n "$_PROXY" ]; then
                wget --proxy=on -c "$_URL" -O "$_TARBALL"
            else
                wget -c "$_URL" -O "$_TARBALL"
            fi
        else
            if [ -n "$_PROXY" ]; then
                curl --proxy "$_PROXY" -L "$_URL" -o "$_TARBALL"
            else
                curl -L "$_URL" -o "$_TARBALL"
            fi
        fi

        if [ ! -s "$_TARBALL" ]; then
            echo "ERROR: failed to download Python 3.12 binary"; rm -f "$_TARBALL"; exit 1
        fi

        echo "Extracting to $HOME/.local/python312..."
        mkdir -p "$HOME/.local/python312"
        tar xzf "$_TARBALL" -C "$HOME/.local/python312" --strip-components=1
        rm -f "$_TARBALL"
        _VENV_PYTHON="$_PY312"
        echo "Python 3.12: $_VENV_PYTHON ($($_VENV_PYTHON --version 2>&1))"
    fi
fi

# Now create venv
if [ -n "$_VENV_PYTHON" ]; then
    echo ""
    _VENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/venv"
    echo "Creating venv at: $_VENV_DIR ..."
    if [ -d "$_VENV_DIR" ]; then
        echo "WARNING: old venv exists at: $_VENV_DIR"
        read -r -p "Remove it? [y/N] " _REMOVE_VENV
        if echo "$_REMOVE_VENV" | grep -qi "^y"; then
            echo "Removing $_VENV_DIR ..."
            rm -rf "$_VENV_DIR"
        else
            echo "Skipping — user declined."
        fi
    fi
    if "$_VENV_PYTHON" -m venv "$_VENV_DIR" 2>&1; then
        echo "venv created — upgrading pip and installing packages..."
        "$_VENV_DIR/bin/pip" install --upgrade pip setuptools wheel 2>&1 | tail -3 || true
        if "$_VENV_DIR/bin/pip" install "pyopencl>=2026.1" "cryptography>=3.1" 2>&1; then
            echo ""
            echo "=== Verification ==="
            "$_VENV_DIR/bin/python3" -c "
import pyopencl, cryptography
v1 = getattr(pyopencl, '__version__', '?')
v2 = cryptography.__version__
print('  pyopencl:    ' + v1)
print('  cryptography: ' + v2)
"
            echo ""
            echo "To use: source $_VENV_DIR/bin/activate && python3 <script>.py"
            exit 0
        fi
    else
        echo "venv creation failed."
    fi
fi

# ============================================================
# Step 4: Failure — manual fallback
# ============================================================
echo ""
echo "=========================================="
echo "  ALL AUTOMATED STEPS FAILED"
echo "=========================================="
echo ""
echo "Manual:"
echo "  sudo apt install python3.12 python3.12-venv python3.12-dev libopencl-dev"
echo "  python3.12 -m venv ./venv && ./venv/bin/pip install pyopencl cryptography"
echo ""
exit 1
