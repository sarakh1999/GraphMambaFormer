# OSC Ascend GPU environment for GraphMambaFormer.
#
# The home filesystem (/users/PCS0289/...) has a 500 GB quota that fills up fast.
# CuPy/Triton/torch JIT-compile CUDA kernels and cache the compiled binaries; if
# the cache lands on a full home dir the compile fails with
# "OSError: [Errno 122] Disk quota exceeded" and the whole cuda_rawkernel tier
# dies. This points every compile cache + TMPDIR at scratch (separate, ~1 PB).
#
# Usage (on a GPU node such as a0008):
#     cd /users/PCS0289/sarakhosravi/mambaformer
#     source osc_gpu_env.sh
#     source .venv/bin/activate

export SCRATCH_CACHE="/fs/scratch/PCS0289/${USER}"

# --- compile / JIT caches -------------------------------------------------- #
export CUPY_CACHE_DIR="${SCRATCH_CACHE}/.cupy_cache"          # CuPy NVRTC cubins
export TRITON_CACHE_DIR="${SCRATCH_CACHE}/.triton_cache"      # Triton kernels
export TORCHINDUCTOR_CACHE_DIR="${SCRATCH_CACHE}/.inductor_cache"  # torch.compile
export CUDA_CACHE_PATH="${SCRATCH_CACHE}/.nv_computecache"    # NVIDIA JIT cache

# --- general caches + temp (keep off the home quota) ----------------------- #
export XDG_CACHE_HOME="${SCRATCH_CACHE}/.cache"
export HF_HOME="${SCRATCH_CACHE}/huggingface"                 # HuggingFace models
export TMPDIR="${SCRATCH_CACHE}/tmp"                          # a0008 has TmpDisk=0

mkdir -p "${CUPY_CACHE_DIR}" "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" \
         "${CUDA_CACHE_PATH}" "${XDG_CACHE_HOME}" "${HF_HOME}" "${TMPDIR}"

# So `python scripts/check_gpu.py` and friends find the package from the repo root.
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd):${PYTHONPATH}"

# --- WFA-GPU (optional GPU gap-affine alignment + on-device CIGAR) ---------- #
# Built by scripts/build_wfa_gpu.sh into build/wfa_gpu/libgmf_wfa_gpu.so. When
# that shim is present this enables the GMF_WFA_GPU=1 extension path; otherwise
# the pipeline silently keeps using the CPU/CuPy WFA path. libwfagpu.so links
# libcudart.so.12, so a CUDA 12.x toolkit module must be loadable at runtime.
GMF_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-.}")" && pwd)"
GMF_WFA_GPU_DEFAULT_LIB="${GMF_REPO_ROOT}/build/wfa_gpu/libgmf_wfa_gpu.so"
if [ -f "${GMF_WFA_GPU_DEFAULT_LIB}" ]; then
    # Pull in a CUDA toolkit for libcudart.so.12 if one is not already active.
    if ! command -v nvcc >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
        module load "${GMF_CUDA_MODULE:-cuda/12.4.1}" >/dev/null 2>&1 || true
    fi
    export GMF_WFA_GPU=1
    export GMF_WFA_GPU_LIB="${GMF_WFA_GPU_DEFAULT_LIB}"
    export LD_LIBRARY_PATH="${GMF_REPO_ROOT}/third_party/WFA-GPU/build:${GMF_REPO_ROOT}/third_party/WFA-GPU/external/WFA2-lib/lib:${LD_LIBRARY_PATH}"
    if command -v nvcc >/dev/null 2>&1; then
        _GMF_CUDA_LIB="$(cd "$(dirname "$(command -v nvcc)")/../lib64" 2>/dev/null && pwd || true)"
        [ -n "${_GMF_CUDA_LIB}" ] && export LD_LIBRARY_PATH="${_GMF_CUDA_LIB}:${LD_LIBRARY_PATH}"
    fi
    echo "WFA-GPU enabled: ${GMF_WFA_GPU_LIB}"
else
    echo "WFA-GPU not built (run scripts/build_wfa_gpu.sh to enable GMF_WFA_GPU); using CPU/CuPy WFA"
fi

echo "OSC GPU env ready: caches -> ${SCRATCH_CACHE}"
