#!/usr/bin/env bash
# 构建可分发的内嵌 Python 环境（macOS arm64）
#
# 用法:  bash scripts/build-python-runtime.sh
# 产物:  packaging/python-runtime/   独立 cpython 运行时（python-build-standalone）
#        packaging/python-site/      pip install --target 装出的可迁移 site-packages
#        packaging/models/           内置 demucs 权重（TORCH_HOME 布局）
#
# 说明:
#   - 依赖版本见 python/requirements-prod.txt，与开发 venv 精确对齐
#   - pyworld 无 macOS arm64 wheel，会在本机从 sdist 编译，需要 Xcode Command Line Tools
#   - 冒烟测试故意设置无效代理，任何联网下载都会立刻失败，以此证明权重确实来自内置缓存
#   - 脚本可重复执行：每次先清空 packaging/python-site
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACK="$ROOT/packaging"
RUNTIME="$PACK/python-runtime"
SITE="$PACK/python-site"
MODELS="$PACK/models"

PY_TAG="20260901"
PY_VER="3.11.16"
PY_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PY_TAG}/cpython-${PY_VER}+${PY_TAG}-aarch64-apple-darwin-install_only.tar.gz"

# 开发机上 demucs 首次运行时 torch.hub 缓存下来的权重；可用 DEMUCS_CACHE_FILE 覆盖
WEIGHTS_SRC="${DEMUCS_CACHE_FILE:-$HOME/.cache/torch/hub/checkpoints/955717e8-8726e21a.th}"

echo "== [1/5] 独立 Python 运行时 =="
mkdir -p "$RUNTIME"
if [ -x "$RUNTIME/python/bin/python3" ]; then
  echo "   已存在，跳过下载: $RUNTIME/python/bin/python3"
else
  echo "   下载 $PY_URL"
  # --http1.1: GitHub release 下载在 HTTP2 下偶发 framing 层错误
  curl -fL --retry 5 --retry-all-errors --retry-delay 3 --http1.1 \
    -o "$PACK/python-runtime.tar.gz" "$PY_URL"
  tar -xzf "$PACK/python-runtime.tar.gz" -C "$RUNTIME"
  rm -f "$PACK/python-runtime.tar.gz"
fi
"$RUNTIME/python/bin/python3" --version

echo "== [2/5] 安装依赖到 $SITE =="
rm -rf "$SITE"
mkdir -p "$SITE"
"$RUNTIME/python/bin/python3" -m pip install \
  --no-cache-dir --disable-pip-version-check \
  --target "$SITE" \
  -r "$ROOT/python/requirements-prod.txt"

echo "== [3/5] 内置 demucs 权重 =="
if [ ! -f "$WEIGHTS_SRC" ]; then
  echo "   找不到权重文件: $WEIGHTS_SRC" >&2
  echo "   先在开发环境跑一次分离让它下载，或用 DEMUCS_CACHE_FILE 指定路径" >&2
  exit 1
fi
mkdir -p "$MODELS/hub/checkpoints"
cp -f "$WEIGHTS_SRC" "$MODELS/hub/checkpoints/"

echo "== [4/5] 离线冒烟测试（无效代理，联网即失败）=="
HTTPS_PROXY="http://127.0.0.1:1" HTTP_PROXY="http://127.0.0.1:1" \
PYTHONPATH="$SITE" TORCH_HOME="$MODELS" PYTHONNOUSERSITE=1 \
"$RUNTIME/python/bin/python3" - <<'PY'
import fastapi, uvicorn, pydantic, websockets, requests
import numpy, scipy, librosa, soundfile, numba
import torch, torchaudio, pyworld
from demucs.pretrained import get_model
model = get_model('htdemucs')
print('   imports OK; htdemucs loaded offline, params =',
      sum(p.numel() for p in model.parameters()))
PY

echo "== [5/5] 体积 =="
du -sh "$RUNTIME" "$SITE" "$MODELS"
echo "完成。"
