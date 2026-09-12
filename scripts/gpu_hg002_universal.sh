#!/usr/bin/env bash
# =============================================================================
# UNIVERSAL (multi-modality) GPU train + eval for HG002 chr21.
#
# Unlike scripts/gpu_hg002_train_eval.sh (Illumina short-read only), this trains
# ONE model across ALL THREE real modalities together and reports the macro
# ("universal") locus accuracy plus a per-modality breakdown:
#     Illumina (short)  +  ONT (long)  +  PacBio HiFi (long)
# via a 3-entry manifest and --balance-modalities so the ~100k short reads do
# not drown the far rarer long reads.
#
# Runs on a SINGLE GPU chosen by CUDA_VISIBLE_DEVICES (default 1) so it can run
# alongside the short-read job on GPU 0 without colliding.
#
# HOW TO RUN (interactive, on the GPU node you already hold):
#   cd /users/PCS0289/sarakhosravi/mambaformer
#   CUDA_VISIBLE_DEVICES=1 bash scripts/gpu_hg002_universal.sh
#
# Override anything:
#   CUDA_VISIBLE_DEVICES=2 EPOCHS=30 MAX_READ_LEN=8192 bash scripts/gpu_hg002_universal.sh
# =============================================================================

#SBATCH --job-name=gmf_hg002_universal
#SBATCH --account=PCS0289
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

# --------------------------------------------------------------------------- #
# Config (override via environment)
# --------------------------------------------------------------------------- #
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Which physical GPU to use. Default 1 (GPU 0 is the short-read run).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

D_MODEL="${D_MODEL:-256}"
EPOCHS="${EPOCHS:-20}"
BATCH="${BATCH:-4}"                 # smaller: long reads are memory-heavy
WORKERS="${WORKERS:-4}"            # modest: share host cores with the GPU-0 run
MAX_READ_LEN="${MAX_READ_LEN:-4096}"  # REQUIRED for ONT/HiFi to bound memory
# The epoch-boundary validation runs the FULL end-to-end aligner over every val
# batch (~1.5s/batch); uncapped that is ~2h/epoch on this val set. The manifest
# lists the rare long-read modalities FIRST (val is not shuffled), so this prefix
# cap still covers all 3 modalities -> macro_locus_accuracy stays valid. 0 = full.
EPOCH_VAL_MAX_BATCHES="${EPOCH_VAL_MAX_BATCHES:-400}"

MANIFEST="${MANIFEST:-scripts/manifests/hg002_chr21_universal.json}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_TRAIN="${OUT_TRAIN:-data/training_runs/hg002_chr21_universal_${STAMP}}"
OUT_EVAL="${OUT_EVAL:-data/eval_runs/hg002_universal_${STAMP}}"
CKPT="$OUT_TRAIN/checkpoint.pt"

# --------------------------------------------------------------------------- #
# Preflight
# --------------------------------------------------------------------------- #
echo "############################################################"
echo "# GraphMambaFormer HG002 UNIVERSAL train+eval  ($STAMP)"
echo "#   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  d_model=$D_MODEL"
echo "#   epochs=$EPOCHS batch=$BATCH max_read_len=$MAX_READ_LEN"
echo "#   manifest=$MANIFEST  (illumina + ont + pacbio_hifi)"
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
# 1) TRAIN one universal model across all modalities (new 2:1:1 arch)
# --------------------------------------------------------------------------- #
echo; echo "### [1/2] train universal checkpoint -> $OUT_TRAIN"
python -u scripts/train.py \
  --data real \
  --manifest "$MANIFEST" \
  --ref-mode linear \
  --balance-modalities \
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
# 2) EVAL the universal model across all modalities (macro + per-modality)
# --------------------------------------------------------------------------- #
echo; echo "### [2/2] eval universal (all modalities, two_pass) -> $OUT_EVAL"
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

# --------------------------------------------------------------------------- #
# Done
# --------------------------------------------------------------------------- #
echo
echo "############################################################"
echo "# DONE (universal)"
echo "#   checkpoint : $CKPT"
echo "#   eval json  : $OUT_EVAL/sweep_metrics.json (or per-mode metrics.json)"
echo "############################################################"
echo "Look for macro_locus_accuracy (the 'universal' number) and the"
echo "per-modality locus/<illumina|ont|pacbio_hifi> breakdown in the eval output."
