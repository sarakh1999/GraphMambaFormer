#!/usr/bin/env bash
# One-shot HG005 chr21 pangenome-windows run on the LOCAL node (e.g. a0024).
# No Slurm / no 'largemem' partition needed: it does the CPU cache build and the
# multi-GPU DDP training sequentially in one process, so there is no NCCL barrier
# timeout (the cache is warm before torchrun starts) and no multi-line paste to
# get mangled. The GPU count is auto-detected (override with NGPU=<n>); it runs
# on 1, 2, or however many GPUs the node exposes.
#
# Run it detached so it survives an SSH drop:
#   cd ~/mambaformer
#   tmux new -s hg005 'bash scripts/transfer/run_hg005_local.sh 2>&1 | tee logs/hg005_local.log'
# Reattach with:  tmux attach -t hg005     (detach: Ctrl-b then d)
#
# Re-running is safe: an existing cache is reused, and an existing checkpoint is
# resumed (set FRESH=1 to ignore an existing checkpoint and start over).
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root (mambaformer)
# shellcheck disable=SC1091
source .venv/bin/activate
module load cuda/13.2.1 gcc/13.2.0 2>/dev/null || true

# The slice/fetch scripts are no longer present in the repo, but the manifest
# they produced is. Train directly on the existing plain manifest (all three
# modalities, 1434 windows, pointing at the shared HG002 graph + HG005 BAMs).
MANIFEST=data/hprc/manifests/chr21_HG005_pangenome_windows.bal.json
# Model width. d_model=256 -> ~15.1M params (the full model); the CLI default of
# 64 (~3M) is only meant for quick CPU smoke tests. Override with DMODEL=<n>.
DMODEL="${DMODEL:-256}"
# Batch size. Must match the batch size the dataset cache was built for, since
# the cache key depends on it. bs=4 keeps the historical cache/checkpoint dirs
# for back-compat; other sizes get their own isolated dirs. Override BATCH_SIZE=<n>.
BATCH_SIZE="${BATCH_SIZE:-4}"
if [ "$BATCH_SIZE" = "4" ]; then
  # Prebuilt 26G bs=4 cache lives here; keep this pointed at it so Step 2 reuses it.
  CACHE_DIR="${CACHE_DIR:-/fs/ess/PCS0289/mambaformer_cache_cpu}"
  OUT="${OUT:-/fs/ess/PCS0289/mambaformer_runs/chr21_hg005_ddp_d${DMODEL}}"
else
  CACHE_DIR="${CACHE_DIR:-/fs/ess/PCS0289/mambaformer_cache_cpu_bs${BATCH_SIZE}}"
  OUT="${OUT:-/fs/ess/PCS0289/mambaformer_runs/chr21_hg005_ddp_d${DMODEL}_bs${BATCH_SIZE}}"
fi
# GPU count: honour an explicit NGPU, else honour CUDA_VISIBLE_DEVICES, else ask
# the driver. Works for 1, 2, or however many GPUs the node actually exposes.
detect_ngpu() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    local ids="${CUDA_VISIBLE_DEVICES//,/ }"
    # shellcheck disable=SC2086
    set -- $ids
    echo "$#"
    return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n
    n="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c .)"
    [ "${n:-0}" -gt 0 ] && { echo "$n"; return; }
  fi
  echo 1
}
NGPU="${NGPU:-$(detect_ngpu)}"
mkdir -p "$CACHE_DIR" "$OUT" logs

echo "=== node: $(hostname) | $(date) ==="
nvidia-smi || true

# --- Step 1: sanity-check the manifest ----------------------------------------
if [ ! -f "$MANIFEST" ]; then
  echo "FATAL: manifest not found: $MANIFEST" >&2
  exit 1
fi
echo "=== [1/3] using manifest: $MANIFEST ==="

# --- Step 2: build the dataset cache on CPU, ONCE, to completion --------------
# Reuse the read-independent reference-index sub-cache so per-window graph/FM
# indexes are loaded, not rebuilt.
cp -rn data/dataset_cache/ref_index "$CACHE_DIR"/ 2>/dev/null || true

if ls "$CACHE_DIR"/*HG005*.pt >/dev/null 2>&1; then
  echo "=== [2/3] dataset cache already present -> skipping build ==="
  ls -lh "$CACHE_DIR"/*HG005*.pt
else
  echo "=== [2/3] building dataset cache (CPU, ~1-1.5h, do not interrupt) ==="
  PYTHONPATH=. python scripts/train.py \
    --data real \
    --manifest "$MANIFEST" \
    --dataset-cache-dir "$CACHE_DIR" \
    --batch-size 4 --max-read-len 65536 \
    --device cpu --epochs 0 \
    --out /fs/ess/PCS0289/mambaformer_runs/cache_build_hg005
  echo "=== cache build done ==="
  ls -lh "$CACHE_DIR"/*HG005*.pt
fi

# --- Step 4: DDP training (loads warm cache -> no barrier timeout) ------------
# Runs on $NGPU GPUs (auto-detected above). NGPU=1 is a plain single-GPU run;
# NGPU>=2 is DDP with one process per GPU.
RESUME_ARGS=()
if [ "${FRESH:-0}" != "1" ] && [ -f "$OUT/last.pt" ]; then
  echo "=== [3/3] resuming from $OUT/last.pt on ${NGPU} GPU(s) (d_model=${DMODEL}) ==="
  RESUME_ARGS=(--resume auto)
else
  echo "=== [3/3] starting fresh ${NGPU}-GPU DDP training (d_model=${DMODEL}) ==="
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node="$NGPU" scripts/train.py \
  --data real \
  --manifest "$MANIFEST" \
  --dataset-cache-dir "$CACHE_DIR" \
  --d-model "$DMODEL" \
  --batch-size "$BATCH_SIZE" --grad-accum 8 --amp-dtype bf16 \
  --max-read-len 65536 \
  --epochs 12 --require-gpu \
  --workers 8 --prefetch 4 \
  --validate-every-steps 500 --plot-every-steps 250 \
  --intra-val-max-batches 12 --save-every-steps 20 \
  "${RESUME_ARGS[@]}" \
  --out "$OUT"

echo "=== DONE -> $OUT ==="
echo "checkpoints: $OUT  (best -> checkpoint.pt, last -> last.pt)"
