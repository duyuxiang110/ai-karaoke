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
#   - 脚本可重复执行：site-packages 已存在则跳过安装；用 FORCE_REBUILD=1 强制重建
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

echo "== [1/6] 独立 Python 运行时 =="
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

echo "== [2/6] 安装依赖到 $SITE =="
if [ -n "${FORCE_REBUILD:-}" ]; then
  rm -rf "$SITE"
fi
if [ -d "$SITE" ] && [ -n "$(ls -A "$SITE" 2>/dev/null)" ]; then
  echo "   已存在，跳过安装: $SITE"
else
  rm -rf "$SITE"
  mkdir -p "$SITE"
  "$RUNTIME/python/bin/python3" -m pip install \
    --no-cache-dir --disable-pip-version-check \
    --target "$SITE" \
    -r "$ROOT/python/requirements-prod.txt"
fi

echo "== [3/6] 精简 site-packages =="
# 删除测试目录和字节码缓存（不删 testing/ — torch.testing 是运行时模块）
find "$SITE" -type d \( -name "tests" -o -name "test" -o -name "__pycache__" \) \
  -exec rm -rf {} + 2>/dev/null || true
# 删除运行时不会用到的传递依赖：
# - sympy: torch 在函数体内惰性 import，推理时不触发
# - networkx: torch 不在模块级 import
# - sklearn: librosa 仅在 decompose/segment 模块 import，这两个模块走 lazy_loader 不会加载
rm -rf "$SITE/sympy" "$SITE"/sympy-*.dist-info 2>/dev/null || true
rm -rf "$SITE/networkx" "$SITE"/networkx-*.dist-info 2>/dev/null || true
rm -rf "$SITE/sklearn" "$SITE"/scikit_learn-*.dist-info 2>/dev/null || true
echo "   精简完成"

echo "== [4/6] 内置 demucs 权重 =="
if [ ! -f "$WEIGHTS_SRC" ]; then
  echo "   找不到权重文件: $WEIGHTS_SRC" >&2
  echo "   先在开发环境跑一次分离让它下载，或用 DEMUCS_CACHE_FILE 指定路径" >&2
  exit 1
fi
mkdir -p "$MODELS/hub/checkpoints"
cp -f "$WEIGHTS_SRC" "$MODELS/hub/checkpoints/"

echo "== [5/6] 离线冒烟测试（无效代理，联网即失败）=="
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

echo "== [6/6] 体积 =="
du -sh "$RUNTIME" "$SITE" "$MODELS"
echo "完成。"
