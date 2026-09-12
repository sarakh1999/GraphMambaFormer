#!/usr/bin/env bash
# =============================================================================
# HG005 SHORT-READ (Illumina) train + HELD-OUT TEST for chr21.
#
# WHAT'S DIFFERENT FROM THE HG002 SHORT-READ SCRIPT
#   * Sample = HG005 (an independent individual from HG002).
#   * Proper 3-way split (fixes the HG002 runs, which had no held-out test):
#       - TRAIN/VAL : chr21:5,000,000-6,000,000  (80/20 split of this window)
#       - TEST      : chr21:6,500,000-7,000,000  (DISJOINT, 500kb gap >> read
#                     length, so no read leakage -> a real generalization number)
#   * Illumina capped to ~12x coverage (48000 reads over 1Mb; test 24000/500kb).
#   * Epoch-boundary validation is capped (--epoch-val-max-batches) so it does
#     not stall ~1h/epoch running the full aligner over the whole val set.
#
# GPU: the "third" GPU by default (CUDA_VISIBLE_DEVICES=2). Because that env var
#   remaps the chosen physical GPU to logical id 0, we pass --devices auto.
#
# HOW TO RUN (on the GPU node you already hold):
#   cd /users/PCS0289/sarakhosravi/mambaformer
#   CUDA_VISIBLE_DEVICES=2 setsid bash scripts/gpu_hg005_shortread.sh \
#       > .tmp/hg005_shortread.log 2>&1 < /dev/null &
#   tail -f .tmp/hg005_shortread.log
# =============================================================================

#SBATCH --job-name=gmf_hg005_shortread
#SBATCH --account=PCS0289
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# "third" GPU. CUDA_VISIBLE_DEVICES remaps it to logical 0 -> use --devices auto.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

D_MODEL="${D_MODEL:-256}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-4}"
# Epoch-boundary validation runs the full aligner over every val batch; cap it.
EPOCH_VAL_MAX_BATCHES="${EPOCH_VAL_MAX_BATCHES:-300}"

MANIFEST="${MANIFEST:-scripts/manifests/hg005_chr21_shortread.json}"
TEST_MANIFEST="${TEST_MANIFEST:-scripts/manifests/hg005_chr21_shortread.test.json}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_TRAIN="${OUT_TRAIN:-data/training_runs/hg005_chr21_shortread_${STAMP}}"
OUT_TEST="${OUT_TEST:-data/eval_runs/hg005_shortread_test_${STAMP}}"
CKPT="$OUT_TRAIN/checkpoint.pt"

echo "############################################################"
echo "# GraphMambaFormer HG005 SHORT-READ train+test  ($STAMP)"
echo "#   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  d_model=$D_MODEL"
echo "#   epochs=$EPOCHS batch=$BATCH  epoch_val_cap=$EPOCH_VAL_MAX_BATCHES"
echo "#   TRAIN/VAL manifest = $MANIFEST   (Illumina ~12x, chr21:5-6Mb)"
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
# 1) TRAIN on chr21:5-6Mb (Illumina, ~12x) -> train/val 80/20
# --------------------------------------------------------------------------- #
echo; echo "### [1/2] train HG005 short-read checkpoint -> $OUT_TRAIN"
python -u scripts/train.py \
  --data real \
  --manifest "$MANIFEST" \
  --ref-mode linear \
  --d-model "$D_MODEL" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH" \
  --workers "$WORKERS" \
  --device cuda --devices auto --require-gpu \
  --monitor locus_accuracy \
  --epoch-val-max-batches "$EPOCH_VAL_MAX_BATCHES" \
  --out "$OUT_TRAIN"

if [ ! -f "$CKPT" ]; then
  if [ -f "$OUT_TRAIN/last.pt" ]; then CKPT="$OUT_TRAIN/last.pt"; else
    echo "ERROR: no checkpoint produced under $OUT_TRAIN" >&2; exit 1; fi
fi
echo "trained checkpoint: $CKPT"

# --------------------------------------------------------------------------- #
# 2) HELD-OUT TEST on the disjoint region chr21:6.5-7.0Mb (two_pass)
# --------------------------------------------------------------------------- #
echo; echo "### [2/2] HELD-OUT TEST (chr21:6.5-7.0Mb, two_pass) -> $OUT_TEST"
python -u scripts/eval.py \
  --data real \
  --manifest "$TEST_MANIFEST" \
  --ref-mode linear \
  --mode two_pass \
  --checkpoint "$CKPT" \
  --batch-size "$BATCH" \
  --workers "$WORKERS" \
  --device cuda --require-gpu \
  --no-bam \
  --out "$OUT_TEST"

echo
echo "############################################################"
echo "# DONE (HG005 short-read)"
echo "#   checkpoint      : $CKPT"
echo "#   held-out test   : $OUT_TEST  (locus_accuracy on chr21:6.5-7.0Mb)"
echo "############################################################"
