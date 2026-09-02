#!/usr/bin/env bash
# Build & install the fused Mamba-2 kernels (causal-conv1d + mamba-ssm) into the
# EXISTING project .venv.
#
# This venv currently has: torch 2.13.0+cu130 (CUDA 13) on CPython 3.12.
# There is NO prebuilt causal-conv1d / mamba-ssm wheel for torch 2.13, so pip
# must COMPILE them from source against the installed torch. A source build
# needs:
#   * nvcc whose MAJOR version matches torch's CUDA (13)  -> module cuda/13.2.1
#   * a real gcc/g++ (NOT the Intel icpc default)         -> module gcc/13.2.0
#   * ninja + a recent setuptools/wheel/packaging
#   * --no-build-isolation, so the compile uses THIS torch (otherwise pip would
#     fetch a fresh, possibly different torch into an isolated build env and the
#     resulting .so would be ABI-incompatible).
#
# RUN THIS ON A GPU (compute) NODE — login nodes have no nvcc and are too loaded
# to compile CUDA kernels:
#
#   tmux new -s build
#   salloc --account=PCS0289 --nodes=1 --gpus-per-node=1 --cpus-per-task=16 --mem=64G --time=2:00:00
#   cd ~/mambaformer && bash scripts/transfer/install_mamba_ssm.sh
#
# Override any of these via env, e.g. CUDA_MODULE=cuda/13.2.1 MAX_JOBS=8 bash ...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

CUDA_MODULE="${CUDA_MODULE:-cuda/13.2.1}"   # MAJOR must match torch's cu130 (=13)
GCC_MODULE="${GCC_MODULE:-gcc/13.2.0}"      # real gcc; CUDA 13 supports gcc 13/14
# A100 = sm_80. Building a single arch is much faster and uses less RAM than the
# default "all arches". Add ';9.0' for H100, ';8.6' for A10/RTX30, etc.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
export MAX_JOBS="${MAX_JOBS:-4}"            # caps parallel nvcc jobs (RAM guard)

echo ">>> loading toolchain modules ($CUDA_MODULE, $GCC_MODULE) ..."
module load "$CUDA_MODULE"
module load "$GCC_MODULE"
export CC=gcc CXX=g++

if ! command -v nvcc >/dev/null 2>&1; then
  echo "ERROR: nvcc not found. Are you on a GPU node with '$CUDA_MODULE' loaded?" >&2
  echo "       Login nodes have no nvcc; run this inside an salloc GPU session." >&2
  exit 1
fi
# Derive CUDA_HOME from nvcc so cpp_extension finds headers/libs reliably.
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"

echo "nvcc     : $(nvcc --version | tail -1)"
echo "gcc      : $(gcc --version | head -1)"
echo "CUDA_HOME: $CUDA_HOME"
echo "arch list: $TORCH_CUDA_ARCH_LIST   MAX_JOBS=$MAX_JOBS"

VENV="$ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
[ -x "$PY" ] || { echo "ERROR: $PY not found — wrong project root?" >&2; exit 1; }

"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

echo ">>> ensuring build deps are present (needed for --no-build-isolation) ..."
"$PIP" install --upgrade pip setuptools wheel ninja packaging

# Order matters: mamba-ssm links against causal-conv1d, so build it first.
echo ">>> building causal-conv1d (this compiles CUDA kernels; be patient) ..."
"$PIP" install -v --no-build-isolation causal-conv1d

echo ">>> building mamba-ssm ..."
"$PIP" install -v --no-build-isolation mamba-ssm

echo "==================== verify ===================="
"$PY" - <<'PY'
import torch
import causal_conv1d, mamba_ssm
print("causal_conv1d :", getattr(causal_conv1d, "__version__", "?"))
print("mamba_ssm     :", getattr(mamba_ssm, "__version__", "?"))
from graphmambaformer.accel import mamba_ssm_available
ok = mamba_ssm_available()
print("mamba_ssm_available:", ok)
# best-effort functional check on the GPU (import success is the real bar)
if ok and torch.cuda.is_available():
    try:
        from mamba_ssm import Mamba2
        m = Mamba2(d_model=256).cuda()
        x = torch.randn(1, 64, 256, device="cuda")
        y = m(x)
        print("Mamba2 forward OK:", tuple(y.shape))
    except Exception as e:  # noqa: BLE001
        print("note: import OK but functional check skipped:", e)
PY
echo "================================================"
echo "If 'mamba_ssm_available: True', restart training — the fused scan is now"
echo "used automatically. You can also relax/drop --max-read-len (the pure-Python"
echo "scan was what forced the 1024 cap on long ONT reads)."
