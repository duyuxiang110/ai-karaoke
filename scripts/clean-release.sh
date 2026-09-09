#!/usr/bin/env bash
# 清理 release/ 目录，只保留 .dmg
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/release"

# 删除所有非 .dmg 文件（blockmap、yml、DS_Store 等）
find . -maxdepth 1 -type f ! -name "*.dmg" -delete

# 删除解包目录（mac-arm64/）
find . -maxdepth 1 -type d ! -name "." -exec rm -rf {} + 2>/dev/null || true

echo "release/ 已清理 — 只保留 .dmg"
ls -lh *.dmg 2>/dev/null || echo "（未找到 .dmg 文件）"
