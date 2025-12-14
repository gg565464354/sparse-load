#!/usr/bin/env bash
set -euo pipefail
set -x

# 1) 安装构建依赖
/root/miniconda3/envs/arkvale/bin/pip install -U pip setuptools wheel ninja cmake packaging

# 2) 指定 CUDA 与算力（4090 为 8.9）
export CUDA_HOME=/usr/local/cuda
export CUDACXX="$CUDA_HOME/bin/nvcc"
export TORCH_CUDA_ARCH_LIST="8.9"     # 如需兼容更多机型，可设 "8.0;8.6;8.9"
export CMAKE_ARGS="-DCMAKE_CUDA_ARCHITECTURES=89"
export MAX_JOBS=$(nproc)

# 2.1) 诊断信息与前置检查
if [ ! -x "$CUDACXX" ]; then
  echo "[FATAL] nvcc not found at $CUDACXX"
  echo "请先安装系统 CUDA Toolkit，或改用 Conda 安装 nvcc："
  echo "  /root/miniconda3/envs/arkvale/bin/conda install -y -c nvidia cuda-nvcc"
  exit 1
fi
"$CUDACXX" --version || true
/root/miniconda3/envs/arkvale/bin/python - <<'PY'
import torch
print("[INFO] torch.version.cuda =", torch.version.cuda)
PY

# 3) 切换到本地 ArkVale 源码目录并重装
cd /root/sparse-load/ArkVale/source
/root/miniconda3/envs/arkvale/bin/python -m pip uninstall -y arkvale || true
/root/miniconda3/envs/arkvale/bin/python -m pip install -v .