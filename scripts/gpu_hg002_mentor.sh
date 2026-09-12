#!/usr/bin/env bash
# =============================================================================
# MENTOR-SPEC universal train + eval for HG002 chr21.
#
# Matches the mentor's data recipe (vs the count-balanced universal script):
#   * Illumina short reads  -> ~12x coverage  (subsampled: 48000 reads, uniform)
#   * ONT + PacBio HiFi     -> ALL long reads, at (near) full length
#
# Differences from scripts/gpu_hg002_universal.sh:
#   - manifest caps Illumina to 48000 reads (~12x) and leaves long reads uncapped
#   - NO --balance-modalities (that equalizes COUNTS, which throws away most
#     Illumina AND does not "use all long reads"); instead keep the natural
#     composition and use --modality-loss-weight so long reads still matter
#   - --max-read-len 24576 so full-length HiFi and most ONT are actually fed to
#     the model (4096 truncated them to the first ~4kb)
#
# RUNTIME NOTE: 48k Illumina reads dominate by count, so an epoch is large.
#   With small batches (long reads are memory-heavy) expect ~1h+/epoch on one
#   A100. Early-stopping on macro_locus_accuracy (patience) usually cuts it short.
#   Lower EPOCHS or BATCH if needed; raise BATCH if no OOM to go faster.
#
# HOW TO RUN (on the GPU node you hold; GPU 2 by default):
#   cd /users/PCS0289/sarakhosravi/mambaformer
#   CUDA_VISIBLE_DEVICES=2 setsid bash scripts/gpu_hg002_mentor.sh \
#       > .tmp/mentor_run.log 2>&1 < /dev/null &
#   tail -f .tmp/mentor_run.log
# =============================================================================

#SBATCH --job-name=gmf_hg002_mentor
#SBATCH --account=PCS0289
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

D_MODEL="${D_MODEL:-256}"
EPOCHS="${EPOCHS:-10}"
BATCH="${BATCH:-4}"                 # if OOM at 24k length -> set BATCH=2; if room -> 8
WORKERS="${WORKERS:-4}"
MAX_READ_LEN="${MAX_READ_LEN:-24576}"  # covers HiFi fully + most ONT
# The epoch-boundary validation runs the FULL end-to-end aligner over every val
# batch; uncapped that is 1-2h/epoch. The manifest lists the rare long-read
# modalities FIRST (val is not shuffled), so this prefix cap still covers all 3
# modalities -> macro_locus_accuracy stays valid. 0 = full pass.
EPOCH_VAL_MAX_BATCHES="${EPOCH_VAL_MAX_BATCHES:-400}"

MANIFEST="${MANIFEST:-scripts/manifests/hg002_chr21_mentor.json}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_TRAIN="${OUT_TRAIN:-data/training_runs/hg002_chr21_mentor_${STAMP}}"
OUT_EVAL="${OUT_EVAL:-data/eval_runs/hg002_mentor_${STAMP}}"
CKPT="$OUT_TRAIN/checkpoint.pt"

echo "############################################################"
echo "# GraphMambaFormer HG002 MENTOR-SPEC train+eval  ($STAMP)"
echo "#   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  d_model=$D_MODEL"
echo "#   epochs=$EPOCHS batch=$BATCH max_read_len=$MAX_READ_LEN"
echo "#   Illumina ~12x (48k reads) + ALL ONT + ALL HiFi (no count-balancing)"
echo "#   manifest=$MANIFEST"
echo "############################################################"

if [ ! -f "$ROOT/.venv/bin/activate" ]; then
  echo "ERROR: $ROOT/.venv not found." >&2; exit 1
fi
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export PYTHONPATH="${PYTHONPATH:-.}"

echo "== visible GPU =="
python -c "import torch; assert torch.cuda.is_available(); print('cuda dev:', torch.cuda.get_device_name(0), '| count', torch.cuda.device_count())" \
  || { echo "ERROR: torch cannot see a CUDA device." >&2; exit 1; }
[ -f "$MANIFEST" ] || { echo "ERROR: manifest $MANIFEST missing." >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 1) TRAIN (mentor composition; NO balance-modalities)
# --------------------------------------------------------------------------- #
echo; echo "### [1/2] train mentor-spec checkpoint -> $OUT_TRAIN"
python -u scripts/train.py \
  --data real \
  --manifest "$MANIFEST" \
  --ref-mode linear \
  --modality-loss-weight \
  --d-model "$D_MODEL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --workers "$WORKERS" \
  --max-read-len "$MAX_READ_LEN" \
  --device cuda --devices auto --require-gpu \
  --monitor macro_locus_accuracy \
  --epoch-val-max-batches "$EPOCH_VAL_MAX_BATCHES" \
  --out "$OUT_TRAIN"

if [ ! -f "$CKPT" ]; then
  if [ -f "$OUT_TRAIN/last.pt" ]; then CKPT="$OUT_TRAIN/last.pt"; else
    echo "ERROR: no checkpoint produced under $OUT_TRAIN" >&2; exit 1; fi
fi
echo "trained checkpoint: $CKPT"

# --------------------------------------------------------------------------- #
# 2) EVAL across all modalities (macro + per-modality)
# --------------------------------------------------------------------------- #
echo; echo "### [2/2] eval mentor-spec (all modalities, two_pass) -> $OUT_EVAL"
python -u scripts/eval.py \
  --data real \
  --manifest "$MANIFEST" \
  --ref-mode linear \
  --mode two_pass \
  --checkpoint "$CKPT" \
  --batch-size "$BATCH" \
  --workers "$WORKERS" \
  --max-read-len "$MAX_READ_LEN" \
  --device cuda --require-gpu \
  --out "$OUT_EVAL"

echo
echo "############################################################"
echo "# DONE (mentor-spec)"
echo "#   checkpoint : $CKPT"
echo "#   eval json  : $OUT_EVAL (macro_locus_accuracy + per-modality)"
echo "############################################################"
