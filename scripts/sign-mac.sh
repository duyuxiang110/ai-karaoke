#!/usr/bin/env bash
# 给 electron-builder 的 macOS 产物补 adhoc 签名（带资源封条）
#
# 背景: 没有 Developer ID 时 electron-builder 会「skipped macOS application code signing」，
# 产物只带 Electron 二进制的链接器签名、没有资源封条（Sealed Resources=none）。
# 这种包一旦带上 quarantine 标记（微信/浏览器下载），Gatekeeper 直接报
# 「已损坏，无法打开」且不给「仍要打开」；补上封条后才会变成可绕过的
# 「无法验证开发者」。
#
# 用法: 在 electron-builder 之后执行，然后用
#       npx electron-builder --prepackaged release/mac-arm64 重打 dmg
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="$ROOT/release/mac-arm64/AI Karaoke.app"

if [ ! -d "$APP" ]; then
  echo "找不到 $APP，请先执行 npm run electron:build" >&2
  exit 1
fi

# 由内而外: 先签嵌套的 framework / helper app，再签外层 bundle
find "$APP/Contents/Frameworks" -maxdepth 1 \
  \( -name "*.framework" -o -name "*.app" -o -name "*.dylib" \) -print0 |
  while IFS= read -r -d '' component; do
    codesign --force --sign - "$component"
  done

codesign --force --sign - "$APP"
codesign --verify --deep --strict "$APP"
echo "signed OK: $APP"
