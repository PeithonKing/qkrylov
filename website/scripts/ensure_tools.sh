#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOLS_DIR="${SCRIPT_DIR}/.tools"
TOOLS_BIN="${TOOLS_DIR}/bin"
mkdir -p "${TOOLS_BIN}"
export PATH="${TOOLS_BIN}:${PATH}"

# 1. Doxygen 1.18.0 Bootstrap
if ! command -v doxygen &>/dev/null || ! doxygen --version 2>/dev/null | grep -q "1.18"; then
    if [ ! -x "${TOOLS_BIN}/doxygen" ]; then
        echo "[ensure_tools] Bootstrapping Doxygen 1.18.0 (x86_64 Linux)..."
        DOXYGEN_TAR="${TOOLS_DIR}/doxygen-1.18.0.linux.bin.tar.gz"
        curl -fsSL "https://github.com/doxygen/doxygen/releases/download/Release_1_18_0/doxygen-1.18.0.linux.bin.tar.gz" -o "${DOXYGEN_TAR}"
        tar -xzf "${DOXYGEN_TAR}" -C "${TOOLS_DIR}"
        cp "${TOOLS_DIR}/doxygen-1.18.0/bin/doxygen" "${TOOLS_BIN}/doxygen"
        chmod +x "${TOOLS_BIN}/doxygen"
        rm -rf "${DOXYGEN_TAR}" "${TOOLS_DIR}/doxygen-1.18.0"
    fi
fi

# 2. Doxybook2 1.5.0 Bootstrap
if ! command -v doxybook2 &>/dev/null || ! doxybook2 --version 2>/dev/null | grep -q "1.5"; then
    if [ ! -x "${TOOLS_BIN}/doxybook2" ]; then
        echo "[ensure_tools] Bootstrapping Doxybook2 1.5.0 (x86_64 Linux)..."
        DOXYBOOK_ZIP="${TOOLS_DIR}/doxybook2-linux-amd64-v1.5.0.zip"
        curl -fsSL "https://github.com/matusnovak/doxybook2/releases/download/v1.5.0/doxybook2-linux-amd64-v1.5.0.zip" -o "${DOXYBOOK_ZIP}"
        unzip -q -o "${DOXYBOOK_ZIP}" bin/doxybook2 -d "${TOOLS_DIR}"
        chmod +x "${TOOLS_BIN}/doxybook2"
        rm -f "${DOXYBOOK_ZIP}"
    fi
fi

echo "[ensure_tools] Tools verified:"
echo "  Doxygen: $(doxygen --version)"
echo "  Doxybook2: $(doxybook2 --version)"
