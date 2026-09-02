#!/usr/bin/env bash
# One-time environment setup for mambaformer on OSC Ascend (A100-80GB / a0022).
#
# Run this on the GPU node (or a login node with a CUDA toolkit module) AFTER
# the rsync from push_to_osc.sh has landed the project in ~/mambaformer.
#
#   cd ~/mambaformer
#   ./scripts/transfer/osc_setup.sh
#
# It builds a fresh .venv (never copy the source box's .venv), installs a CUDA
# torch wheel, the base deps, and — the important part on A100 — the fused
# Mamba-2 kernels (mamba-ssm + causal-conv1d) that replace the pure-Python scan
# which OOM'd on long reads. Ends by verifying the fused kernel is picked up.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# --- OSC modules ----------------------------------------------------------- #
# The torch CUDA wheel MUST match the nvcc toolkit that compiles mamba-ssm.
# Ascend exposes nvcc 12.4 (cuda/12.4.1), so we install a cu124 torch wheel.
# (The earlier failure was torch cu130 vs nvcc 12.4.) mamba-ssm also needs a
# real gcc/g++ — the default env here uses Intel icpc, which its build rejects.
CUDA_WHEEL="${CUDA_WHEEL:-cu124}"          # match the loaded cuda module's major.minor
CUDA_MODULE="${CUDA_MODULE:-cuda/12.4.1}"  # `module spider cuda` to confirm the name
GCC_MODULE="${GCC_MODULE:-gcc}"            # `module spider gcc` (some sites: gnu/…)
module load "$CUDA_MODULE" 2>/dev/null || echo "note: could not 'module load $CUDA_MODULE' — ensure nvcc is on PATH"
module load "$GCC_MODULE"  2>/dev/null || echo "note: could not 'module load $GCC_MODULE' — ensure g++ is on PATH"
# Force gcc/g++ so the mamba-ssm/causal-conv1d nvcc host compiler isn't icpc.
export CC="${CC:-gcc}" CXX="${CXX:-g++}"
command -v nvcc >/dev/null 2>&1 && echo "nvcc: $(nvcc --version | tail -1)" || echo "WARNING: nvcc not found — mamba-ssm build will fail"
echo "CC=$CC ($( $CC --version 2>/dev/null | head -1 ))  CXX=$CXX"

# --- Python venv ----------------------------------------------------------- #
PYTHON="${PYTHON:-python3}"
if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools ninja

# --- CUDA torch (+vision/audio) FIRST, pinned to the toolkit's CUDA -------- #
# Install the whole torch trio from the CUDA index so `requirements.txt`
# (torchvision/torchaudio, unpinned) can't later pull a mismatched cu130 torch
# from PyPI. pip's default "only-if-needed" then leaves these in place.
echo "installing torch/vision/audio ($CUDA_WHEEL) ..."
pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/${CUDA_WHEEL}"

# --- base deps (must NOT upgrade torch) ------------------------------------ #
pip install -r requirements.txt
echo "torch after base deps: $(python -c 'import torch;print(torch.__version__)')"

# --- fused Mamba-2 kernels (the A100 unlock) ------------------------------- #
# Built from source against the cu124 torch (now matching nvcc 12.4). Pinned to
# 2.2.x, which publishes prebuilt cu12/torch2.x/cp312 wheels (fewer heavy deps
# than 2.3.x, which only ships cu13 wheels). MAX_JOBS caps compile RAM.
export MAX_JOBS="${MAX_JOBS:-4}"
echo "installing causal-conv1d + mamba-ssm (MAX_JOBS=$MAX_JOBS, CC=$CC) ..."
pip install --no-build-isolation "causal-conv1d==1.4.0" || \
  echo "WARNING: causal-conv1d build failed (check nvcc/torch CUDA + gcc)"
pip install --no-build-isolation "mamba-ssm==2.2.2" || \
  echo "WARNING: mamba-ssm build failed — training still runs via the (capped) Python scan"

# triton usually ships with the CUDA torch wheel; ensure it's present.
pip install "triton>=2.3.0" || true

# --- verify ---------------------------------------------------------------- #
echo "=================== environment check ==================="
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
try:
    from graphmambaformer.accel import mamba_ssm_available
    print("mamba_ssm_available:", mamba_ssm_available())
except Exception as e:
    print("mamba_ssm_available check errored:", e)
PY
echo "========================================================="
echo "If mamba_ssm_available is True you can train FULL-LENGTH long reads."
echo "If False, keep --max-read-len 1024 (see train_osc.sbatch)."
