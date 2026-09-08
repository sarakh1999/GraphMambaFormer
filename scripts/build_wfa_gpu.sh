#!/usr/bin/env bash
#
# Build WFA-GPU + the graphmambaformer C-ABI shim into libgmf_wfa_gpu.so.
#
# WFA-GPU (https://github.com/quim0/WFA-GPU, MIT) is the maintained,
# CUDA-12-capable replacement for the archived GenomeWorks cudaaligner: batched
# gap-affine pairwise alignment with the CIGAR computed on the GPU. This script
# clones and compiles it, then compiles csrc/wfa_gpu_shim.c against it.
#
# Run on a GPU node (e.g. a0015) with the CUDA toolkit loaded:
#
#     cd ~/mambaformer
#     source osc_gpu_env.sh
#     bash scripts/build_wfa_gpu.sh
#
# On success it prints the environment variables to export (append them to
# osc_gpu_env.sh) so the Python binding can find the library. Until this runs,
# graphmambaformer.accel.wfa_gpu_ops.available() is False and the pipeline uses
# the existing CPU/CuPy WFA path unchanged.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENDOR="${GMF_WFA_GPU_SRC:-$REPO_ROOT/third_party/WFA-GPU}"
SHIM_SRC="$REPO_ROOT/csrc/wfa_gpu_shim.c"
OUT_DIR="${GMF_WFA_GPU_BUILD:-$REPO_ROOT/build/wfa_gpu}"
WFA_GPU_REF="${GMF_WFA_GPU_COMMIT:-main}"

# Ensure a CUDA toolkit (nvcc) is on PATH. On module-based clusters (OSC Ascend)
# osc_gpu_env.sh only sets cache dirs — the toolkit itself comes from an Lmod
# module — so try to load one automatically, then guide the user if that fails.
if ! command -v nvcc >/dev/null 2>&1; then
    if command -v module >/dev/null 2>&1; then
        echo ">> nvcc not found; trying to load a CUDA module (${GMF_CUDA_MODULE:-cuda})"
        module load "${GMF_CUDA_MODULE:-cuda}" >/dev/null 2>&1 || true
    fi
fi
if ! command -v nvcc >/dev/null 2>&1; then
    echo "ERROR: nvcc (CUDA toolkit) not found."
    if command -v module >/dev/null 2>&1; then
        echo "  Load one first, e.g.:"
        echo "      module avail cuda            # list available versions"
        echo "      module load cuda/12.4.1      # pick one; a 12.x toolkit is recommended for WFA-GPU"
        echo "  or point this script at a specific module:"
        echo "      GMF_CUDA_MODULE=cuda/12.4.1 bash scripts/build_wfa_gpu.sh"
    else
        echo "  Install the CUDA toolkit or add nvcc to PATH, then re-run."
    fi
    exit 1
fi
echo ">> build_wfa_gpu.sh: PIC-rebuild + whole-archive shim link (v2)"
echo ">> Using nvcc: $(command -v nvcc) [$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | head -1)]"
command -v git  >/dev/null 2>&1 || { echo "ERROR: git not found."; exit 1; }
command -v gcc  >/dev/null 2>&1 || { echo "ERROR: gcc not found."; exit 1; }
[ -f "$SHIM_SRC" ] || { echo "ERROR: shim source missing: $SHIM_SRC"; exit 1; }

mkdir -p "$(dirname "$VENDOR")" "$OUT_DIR"

if [ ! -d "$VENDOR/.git" ]; then
    echo ">> Cloning WFA-GPU into $VENDOR"
    git clone https://github.com/quim0/WFA-GPU.git "$VENDOR"
fi
(
    cd "$VENDOR"
    git fetch --all --tags --quiet || true
    git checkout "$WFA_GPU_REF" --quiet || true
    git submodule update --init --recursive
)

WFAGPU_LIBDIR="$VENDOR/build"
WFA2_LIBDIR="$VENDOR/external/WFA2-lib/lib"

if [ -f "$WFAGPU_LIBDIR/libwfagpu.so" ] && [ "${GMF_WFA_GPU_REBUILD:-0}" != "1" ]; then
    echo ">> Found existing $WFAGPU_LIBDIR/libwfagpu.so; skipping WFA-GPU CUDA rebuild"
    echo "   (set GMF_WFA_GPU_REBUILD=1 to force a full rebuild)"
else
    echo ">> Building WFA-GPU (compiles WFA2-lib + libwfagpu.so; this can take a few minutes)"
    ( cd "$VENDOR" && ./build.sh )
fi
[ -f "$WFAGPU_LIBDIR/libwfagpu.so" ] || { echo "ERROR: libwfagpu.so not found in $WFAGPU_LIBDIR"; exit 1; }

# WFA-GPU builds WFA2-lib as a -fPIE static archive (libwfa.a), which cannot be
# linked into a shared object ("relocation R_X86_64_PC32 against `stderr' ...
# recompile with -fPIC"). Rebuild the C libraries as position-independent and
# archive libwfa.a ourselves. We deliberately do NOT use the Makefile's
# `all`/`lib_wfa` target: it also archives the (C++-only, here unused) libwfacpp.a
# from build/cpp/*.o and errors out when that directory is empty. Building the C
# subdirs directly also drops any C++-compiler dependency. libwfagpu.so is
# unaffected: it leaves the WFA2 symbols undefined and resolves them from our .so
# at load time.
echo ">> Rebuilding WFA2-lib as PIC (WFA-GPU builds it -fPIE, unusable in a .so)"
WFA2_DIR="$VENDOR/external/WFA2-lib"
make -C "$WFA2_DIR" clean >/dev/null 2>&1 || true
mkdir -p "$WFA2_DIR/build" "$WFA2_DIR/lib"
make -C "$WFA2_DIR" alignment system utils wavefront \
    CC=gcc CC_FLAGS="-Wall -g -fPIC -O3 -march=native"
rm -f "$WFA2_LIBDIR/libwfa.a"
ar -rsc "$WFA2_LIBDIR/libwfa.a" "$WFA2_DIR"/build/*.o
[ -f "$WFA2_LIBDIR/libwfa.a" ] || { echo "ERROR: PIC libwfa.a not produced in $WFA2_LIBDIR"; exit 1; }

# Locate the CUDA runtime dir (next to nvcc) so libwfagpu.so's libcudart is found
# at load time via a transitive DT_RPATH (see --disable-new-dtags below).
CUDA_LIBDIR="$(cd "$(dirname "$(command -v nvcc)")/../lib64" 2>/dev/null && pwd || true)"

echo ">> Compiling the graphmambaformer shim -> $OUT_DIR/libgmf_wfa_gpu.so"
# -lwfagpu (shared) provides the wfagpu_* API. The WFA2 archive is embedded with
# --whole-archive: ld defaults to --allow-shlib-undefined when making a .so, so
# without it libwfagpu.so's WFA2 references would be left unresolved at dlopen.
# Old-style (transitive) RPATHs let the whole chain — including libcudart — load
# without extra LD_LIBRARY_PATH entries.
gcc -O3 -fPIC -shared "$SHIM_SRC" -o "$OUT_DIR/libgmf_wfa_gpu.so" \
    -I "$VENDOR/lib" -I "$VENDOR" \
    -L "$WFAGPU_LIBDIR" \
    -Wl,--disable-new-dtags \
    -Wl,-rpath,"$WFAGPU_LIBDIR" -Wl,-rpath,"$WFA2_LIBDIR" \
    ${CUDA_LIBDIR:+-Wl,-rpath,"$CUDA_LIBDIR"} \
    -lwfagpu \
    -Wl,--whole-archive "$WFA2_LIBDIR/libwfa.a" -Wl,--no-whole-archive \
    -lm -lrt -lpthread -fopenmp
[ -f "$OUT_DIR/libgmf_wfa_gpu.so" ] || { echo "ERROR: shim link failed."; exit 1; }

echo ""
echo "=========================================================================="
echo "WFA-GPU shim built: $OUT_DIR/libgmf_wfa_gpu.so"
echo "Add these to your environment (e.g. append to osc_gpu_env.sh):"
echo ""
echo "  export GMF_WFA_GPU_LIB=\"$OUT_DIR/libgmf_wfa_gpu.so\""
echo "  export LD_LIBRARY_PATH=\"$WFAGPU_LIBDIR:$WFA2_LIBDIR${CUDA_LIBDIR:+:$CUDA_LIBDIR}:\$LD_LIBRARY_PATH\""
echo ""
echo "The CUDA toolkit module used to build must also be loadable at runtime"
echo "(libwfagpu.so needs libcudart.so.12); keep 'module load ${GMF_CUDA_MODULE:-cuda/12.4.1}' in your job."
echo ""
echo "Then enable the GPU WFA path with:  export GMF_WFA_GPU=1"
echo "Verify with:  PYTHONPATH=. python scripts/verify_wfa_gpu.py"
echo "=========================================================================="
