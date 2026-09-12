#!/usr/bin/env bash
# =============================================================================
# HG005 UNIVERSAL (all-modality) train + HELD-OUT TEST for chr21.
#
# Mentor data recipe on an independent sample (HG005):
#   * Illumina short reads -> ~12x coverage (48000 reads over 1Mb; test 24000/500kb)
#   * ONT + PacBio HiFi    -> ALL long reads, at (near) full length
#   * NO --balance-modalities (keep natural composition); --modality-loss-weight
#     so the rare long reads still drive the loss.
#
# Proper 3-way split (fixes the HG002 runs, which had no held-out test):
#   - TRAIN/VAL : chr21:5,000,000-6,000,000  (80/20 split of this window)
#   - TEST      : chr21:6,500,000-7,000,000  (DISJOINT, 500kb gap >> read length)
#
# GPU: the "fourth" GPU by default (CUDA_VISIBLE_DEVICES=3). That env var remaps
#   the chosen physical GPU to logical id 0, so we pass --devices auto.
#
# HOW TO RUN (on the GPU node you already hold):
#   cd /users/PCS0289/sarakhosravi/mambaformer
#   CUDA_VISIBLE_DEVICES=3 setsid bash scripts/gpu_hg005_universal.sh \
#       > .tmp/hg005_universal.log 2>&1 < /dev/null &
#   tail -f .tmp/hg005_universal.log
# =============================================================================

#SBATCH --job-name=gmf_hg005_universal
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

# "fourth" GPU. CUDA_VISIBLE_DEVICES remaps it to logical 0 -> use --devices auto.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

D_MODEL="${D_MODEL:-256}"
EPOCHS="${EPOCHS:-15}"
BATCH="${BATCH:-4}"                 # if OOM at 24k length -> set BATCH=2; if room -> 8
WORKERS="${WORKERS:-4}"
MAX_READ_LEN="${MAX_READ_LEN:-24576}"  # covers HiFi fully + most ONT
# Epoch-boundary validation runs the full aligner over every val batch; cap it.
# Manifest lists long reads FIRST (val is not shuffled) so this prefix cap still
# covers all 3 modalities -> macro_locus_accuracy stays valid.
EPOCH_VAL_MAX_BATCHES="${EPOCH_VAL_MAX_BATCHES:-400}"

MANIFEST="${MANIFEST:-scripts/manifests/hg005_chr21_universal.json}"
TEST_MANIFEST="${TEST_MANIFEST:-scripts/manifests/hg005_chr21_universal.test.json}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_TRAIN="${OUT_TRAIN:-data/training_runs/hg005_chr21_universal_${STAMP}}"
OUT_TEST="${OUT_TEST:-data/eval_runs/hg005_universal_test_${STAMP}}"
CKPT="$OUT_TRAIN/checkpoint.pt"

echo "############################################################"
echo "# GraphMambaFormer HG005 UNIVERSAL train+test  ($STAMP)"
echo "#   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  d_model=$D_MODEL"
echo "#   epochs=$EPOCHS batch=$BATCH max_read_len=$MAX_READ_LEN epoch_val_cap=$EPOCH_VAL_MAX_BATCHES"
echo "#   Illumina ~12x + ALL ONT + ALL HiFi (no count-balancing)"
echo "#   TRAIN/VAL manifest = $MANIFEST   (chr21:5-6Mb)"
echo "#   HELD-OUT TEST      = $TEST_MANIFEST (chr21:6.5-7.0Mb, disjoint)"
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
[ -f "$MANIFEST" ]      || { echo "ERROR: manifest $MANIFEST missing." >&2; exit 1; }
[ -f "$TEST_MANIFEST" ] || { echo "ERROR: test manifest $TEST_MANIFEST missing." >&2; exit 1; }

# --------------------------------------------------------------------------- #
# 1) TRAIN one universal model (mentor composition; NO balance-modalities)
# --------------------------------------------------------------------------- #
echo; echo "### [1/2] train HG005 universal checkpoint -> $OUT_TRAIN"
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
# 2) HELD-OUT TEST on chr21:6.5-7.0Mb, all modalities (two_pass)
# --------------------------------------------------------------------------- #
echo; echo "### [2/2] HELD-OUT TEST (chr21:6.5-7.0Mb, all modalities, two_pass) -> $OUT_TEST"
python -u scripts/eval.py \
  --data real \
  --manifest "$TEST_MANIFEST" \
  --ref-mode linear \
  --mode two_pass \
  --checkpoint "$CKPT" \
  --batch-size "$BATCH" \
  --workers "$WORKERS" \
  --max-read-len "$MAX_READ_LEN" \
  --device cuda --require-gpu \
  --no-bam \
  --out "$OUT_TEST"

echo
echo "############################################################"
echo "# DONE (HG005 universal)"
echo "#   checkpoint      : $CKPT"
echo "#   held-out test   : $OUT_TEST  (macro_locus_accuracy + per-modality)"
echo "############################################################"
