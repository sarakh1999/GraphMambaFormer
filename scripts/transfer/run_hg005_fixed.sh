#!/usr/bin/env bash
# HG005 chr21 run with the FIXED loss stack (scripts/train_fixed.py).
#
# Identical to run_hg005_local.sh except:
#   * it launches scripts/train_fixed.py, which activates ALL fixes:
#       - router load-balance + eager Kendall weights (graph_mamba_loss_fixed.py)
#       - pinned chain decoys + local position target (targets_fixed.py)
#     Targets are rebuilt every step from the cached (reads, reference), so these
#     apply on the EXISTING cache with no rebuild needed.
#   * it writes to a *_fixed OUT dir so it never collides with the currently
#     running (unfixed) job's checkpoints.
#
# Because the fixed criterion adds trainable log-variance parameters, its
# checkpoints are NOT compatible with the old run's last.pt; this launcher
# therefore starts the fixed run fresh (set RESUME=1 only to resume a *_fixed
# checkpoint made by this same script).
#
# Run detached so it survives an SSH drop:
#   cd ~/mambaformer
#   tmux new -s hg005fix 'bash scripts/transfer/run_hg005_fixed.sh 2>&1 | tee logs/hg005_fixed.log'
set -euo pipefail

cd "$(dirname "$0")/../.."          # repo root (mambaformer)
# shellcheck disable=SC1091
source .venv/bin/activate
module load cuda/13.2.1 gcc/13.2.0 2>/dev/null || true

MANIFEST=data/hprc/manifests/chr21_HG005_pangenome_windows.bal.json
DMODEL="${DMODEL:-256}"
BATCH_SIZE="${BATCH_SIZE:-4}"
if [ "$BATCH_SIZE" = "4" ]; then
  CACHE_DIR="${CACHE_DIR:-/fs/ess/PCS0289/mambaformer_cache_cpu}"
  OUT="${OUT:-/fs/ess/PCS0289/mambaformer_runs/chr21_hg005_ddp_d${DMODEL}_fixed}"
else
  CACHE_DIR="${CACHE_DIR:-/fs/ess/PCS0289/mambaformer_cache_cpu_bs${BATCH_SIZE}}"
  OUT="${OUT:-/fs/ess/PCS0289/mambaformer_runs/chr21_hg005_ddp_d${DMODEL}_bs${BATCH_SIZE}_fixed}"
fi

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

echo "=== node: $(hostname) | $(date) | FIXED stack ==="
nvidia-smi || true

if [ ! -f "$MANIFEST" ]; then
  echo "FATAL: manifest not found: $MANIFEST" >&2
  exit 1
fi
echo "=== [1/2] using manifest: $MANIFEST ==="

# Reuse the warm CPU cache (data is unchanged by the fixes).
cp -rn data/dataset_cache/ref_index "$CACHE_DIR"/ 2>/dev/null || true
if ls "$CACHE_DIR"/*HG005*.pt >/dev/null 2>&1; then
  echo "=== dataset cache present -> reusing ==="
  ls -lh "$CACHE_DIR"/*HG005*.pt
else
  echo "FATAL: no dataset cache in $CACHE_DIR; build it first (run_hg005_local.sh)" >&2
  exit 1
fi

RESUME_ARGS=()
if [ "${RESUME:-0}" = "1" ] && [ -f "$OUT/last.pt" ]; then
  echo "=== [2/2] resuming FIXED run from $OUT/last.pt on ${NGPU} GPU(s) ==="
  RESUME_ARGS=(--resume auto)
else
  echo "=== [2/2] starting FRESH ${NGPU}-GPU FIXED run (d_model=${DMODEL}) ==="
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node="$NGPU" scripts/train_fixed.py \
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
