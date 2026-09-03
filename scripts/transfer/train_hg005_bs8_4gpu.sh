#!/usr/bin/env bash
# HG005 chr21 pangenome-windows — 4x A100-80GB, batch-size 8, DDP, FIXED stack.
#
# Run it ON THE GPU NODE (e.g. a0008), in your own shell so it can write to
# /fs/ess and see the GPUs:
#
#   cd ~/mambaformer
#   tmux new -s hg005bs8 'bash scripts/transfer/train_hg005_bs8_4gpu.sh 2>&1 | tee logs/hg005_bs8_4gpu.log'
#   # reattach: tmux attach -t hg005bs8   (detach: Ctrl-b then d)
#
# What it does:
#   1. Ensures the batch-size-8 dataset cache exists. The earlier CPU build read
#      all the BAMs (the ~4.5h phase) but was killed during torch.save, so only a
#      0-byte .tmp survived. Rather than re-read the BAMs, this RE-CHUNKS the
#      complete batch-size-4 cache (26 GB, identical reads + reference indexes)
#      into a batch-size-8 cache in a couple of minutes. No BAM re-reading.
#      (Regroup loads ~26 GB and writes ~26 GB, so it needs ~60 GB RAM — trivial
#       on an A100 node, and it runs ONCE in a single process BEFORE torchrun so
#       DDP ranks never race to rebuild the cache and time out at the NCCL
#       barrier.)
#   2. Launches 4-GPU single-node DDP (one process per GPU, gradients all-reduced)
#      at batch-size 8 per GPU, bf16, full d_model=256, with host workers tuned so
#      the CPU-side seeding/chaining keeps all 4 GPUs fed (high utilisation).
#
# Chain-ranking loss fix (was ~0): this launches scripts/train_fixed.py, which
# activates the FIXED stack with zero edits to the original modules:
#   * FixedTargetBuilder — PINS >=1 hard-negative decoy chain per read, so every
#     read has >=2 candidates and the listwise cross-entropy is no longer the
#     degenerate single-candidate case (softmax of one element == 1.0, grad == 0).
#     Also switches the position head to a learnable local (in-window) target.
#   * FixedGraphMambaLoss — router load-balancing (stops the 100%-"fast" collapse)
#     + eager Kendall log-variances (so the learnable loss weights actually train).
# Supervision is rebuilt live each step from the cached (reads, indexes), so these
# fixes apply on the EXISTING cache — no cache rebuild needed for them.
# Watch history/plot 06 "chain_candidates_per_read" (>1) and the "chain" loss
# term (now non-zero) to confirm the fix is live.
#
# Prefer plain train.py (chain fix only, via the decoy_chains=1 default; no router/
# Kendall/position changes)?  Set  ENTRY=scripts/train.py.
set -euo pipefail

cd "$(dirname "$0")/../.."            # repo root (mambaformer)
# shellcheck disable=SC1091
source .venv/bin/activate
module load cuda/13.2.1 gcc/13.2.0 2>/dev/null || true

ENTRY="${ENTRY:-scripts/train_fixed.py}"
MANIFEST=data/hprc/manifests/chr21_HG005_pangenome_windows.bal.json
DMODEL="${DMODEL:-256}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

# batch-size-8 cache dir + the exact filename train.py computes for
# (manifest + batch-size 8 + max-read-len 65536): the digest is taken from the
# .tmp file the original bs8 build left behind, so train.py will find and reuse
# this file instead of rebuilding.
CACHE_DIR="${CACHE_DIR:-/fs/ess/PCS0289/mambaformer_cache_cpu_bs8}"
BS8_CACHE="$CACHE_DIR/chr21_HG005_pangenome_windows.bal.7249afbb7ce2ba62.pt"
BS4_CACHE="${BS4_CACHE:-/fs/ess/PCS0289/mambaformer_cache_cpu/chr21_HG005_pangenome_windows.bal.1d1443d1290bcb29.pt}"

OUT="${OUT:-/fs/ess/PCS0289/mambaformer_runs/chr21_hg005_ddp_d${DMODEL}_bs${BATCH_SIZE}_fixed}"

# GPU count: honour NGPU, else CUDA_VISIBLE_DEVICES, else the driver.
detect_ngpu() {
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    local ids="${CUDA_VISIBLE_DEVICES//,/ }"; set -- $ids; echo "$#"; return
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local n; n="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -c .)"
    [ "${n:-0}" -gt 0 ] && { echo "$n"; return; }
  fi
  echo 1
}
NGPU="${NGPU:-$(detect_ngpu)}"

# Host worker threads that feed the GPUs (seeding/chaining/supervision are the
# CPU-bound stages that otherwise starve the GPUs). Split the node's cores across
# the DDP ranks so the pools don't oversubscribe: cores/NGPU, capped so a single
# rank can't hog the box.
CORES="$(nproc)"
WORKERS="${WORKERS:-$(( CORES / NGPU > 14 ? 14 : CORES / NGPU ))}"
[ "$WORKERS" -lt 2 ] && WORKERS=2
PREFETCH="${PREFETCH:-6}"

mkdir -p "$OUT" logs "$CACHE_DIR"

echo "=== node: $(hostname) | $(date) | cores=$CORES | GPUs=$NGPU ==="
nvidia-smi || true
echo "=== entry: $ENTRY | manifest: $MANIFEST | d_model=$DMODEL | batch-size=$BATCH_SIZE | grad-accum=$GRAD_ACCUM ==="
echo "=== cache: $BS8_CACHE ==="
echo "=== out:   $OUT | workers/rank=$WORKERS prefetch=$PREFETCH ==="

if [ ! -f "$MANIFEST" ]; then
  echo "FATAL: manifest not found: $MANIFEST" >&2; exit 1
fi

# Make sure the read-independent reference-index sub-cache is present so no window
# graph/FM-index is rebuilt (it is shared with the bs4 cache).
cp -rn data/dataset_cache/ref_index "$CACHE_DIR"/ 2>/dev/null || true

# --- Ensure the batch-size-8 cache exists (regroup from bs4, fast) ------------
if [ -f "$BS8_CACHE" ]; then
  echo "=== [1/2] bs8 dataset cache present -> reuse ==="
  ls -lh "$BS8_CACHE"
elif [ -f "$BS4_CACHE" ]; then
  echo "=== [1/2] bs8 cache missing -> re-chunking bs4 cache (no BAM re-read) ==="
  PYTHONPATH=. python scripts/transfer/regroup_dataset_cache.py \
    --src "$BS4_CACHE" \
    --dst "$BS8_CACHE" \
    --batch-size "$BATCH_SIZE"
  ls -lh "$BS8_CACHE"
else
  echo "FATAL: neither the bs8 cache nor the bs4 source cache exists." >&2
  echo "       bs8: $BS8_CACHE" >&2
  echo "       bs4: $BS4_CACHE" >&2
  exit 1
fi

# --- Resume if a checkpoint exists (safe re-run) -----------------------------
# NOTE: the FIXED criterion adds trainable Kendall log-variance params, so its
# checkpoints are NOT interchangeable with a plain-train.py run's last.pt. This
# only resumes checkpoints written by this same (fixed) launcher.
RESUME_ARGS=()
if [ "${FRESH:-0}" != "1" ] && [ -f "$OUT/last.pt" ]; then
  echo "=== [2/2] resuming from $OUT/last.pt on ${NGPU} GPU(s) ==="
  RESUME_ARGS=(--resume auto)
else
  echo "=== [2/2] starting fresh ${NGPU}-GPU DDP training (bs=${BATCH_SIZE}, d_model=${DMODEL}) ==="
fi

# A100 = Ampere: bf16 (no GradScaler). NO --compile (crashes mamba-ssm Triton).
# expandable_segments cuts allocator fragmentation (turns a borderline step into
# a hard OOM otherwise).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Keep BLAS from spawning its own thread army inside each DDP rank on top of the
# host worker pool (that oversubscription is a common GPU-starving stall).
export OMP_NUM_THREADS="$WORKERS"
export MKL_NUM_THREADS="$WORKERS"

torchrun --standalone --nproc_per_node="$NGPU" "$ENTRY" \
  --data real \
  --manifest "$MANIFEST" \
  --dataset-cache-dir "$CACHE_DIR" \
  --d-model "$DMODEL" \
  --batch-size "$BATCH_SIZE" --grad-accum "$GRAD_ACCUM" --amp-dtype bf16 \
  --max-read-len 65536 \
  --epochs 12 --require-gpu \
  --workers "$WORKERS" --prefetch "$PREFETCH" \
  --validate-every-steps 500 --plot-every-steps 250 \
  --intra-val-max-batches 12 --save-every-steps 1000 \
  "${RESUME_ARGS[@]}" \
  --out "$OUT"

echo "=== DONE -> $OUT ==="
echo "checkpoints: $OUT  (best -> checkpoint.pt, last -> last.pt)"
echo "plots:       $OUT/plots/   (06_label_balance.png shows chain_candidates_per_read)"
