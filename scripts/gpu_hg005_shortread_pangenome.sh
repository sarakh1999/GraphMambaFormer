#!/usr/bin/env bash
# =============================================================================
# HG005 SHORT-READ (Illumina) train + HELD-OUT TEST on the PANGENOME graph.
#
# Identical experiment to scripts/gpu_hg005_shortread.sh (same sample, same
# Illumina ~12x data, same disjoint train/val/TEST regions) BUT trained and
# tested in --ref-mode pangenome instead of linear:
#   * The GFA variation graph (chr21.gfa, in the manifest) is attached, so
#     seeding/chaining/extension become graph-aware and the 3-layer GATv2 tower
#     is actually exercised (idle in linear mode).
#   * Run this alongside the linear short-read run to get a linear-vs-graph
#     comparison on the SAME held-out test region (chr21:6.5-7.0Mb).
#
# SPLIT (unchanged, the fix from the HG002 runs):
#   - TRAIN/VAL : chr21:5,000,000-6,000,000  (80/20)
#   - TEST      : chr21:6,500,000-7,000,000  (DISJOINT, 500kb gap >> read length)
#
# NOTE (pangenome cost): the first run builds a graph-aware seed index for the
#   region (slower than linear, and the chr21.gfa is ~719MB); it is cached, so
#   later runs/epochs are faster. Pangenome also typically needs more epochs to
#   converge than the near-ceiling linear run -- early stopping (patience) will
#   still cut it off when val locus_accuracy plateaus.
#
# GPU: set CUDA_VISIBLE_DEVICES to a FREE GPU (GPUs 0-3 may be busy with the
#   other four runs -- check `nvidia-smi` first). That env var remaps the chosen
#   physical GPU to logical id 0, so we pass --devices auto.
#
# HOW TO RUN (on the GPU node you already hold):
#   cd /users/PCS0289/sarakhosravi/mambaformer
#   CUDA_VISIBLE_DEVICES=<free-gpu> setsid bash scripts/gpu_hg005_shortread_pangenome.sh \
#       > .tmp/hg005_shortread_pangenome.log 2>&1 < /dev/null &
#   tail -f .tmp/hg005_shortread_pangenome.log
# =============================================================================

#SBATCH --job-name=gmf_hg005_shortread_pango
#SBATCH --account=PCS0289
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Point at a FREE GPU. CUDA_VISIBLE_DEVICES remaps it to logical 0 -> --devices auto.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

D_MODEL="${D_MODEL:-256}"
EPOCHS="${EPOCHS:-25}"             # pangenome converges slower than linear
BATCH="${BATCH:-8}"
WORKERS="${WORKERS:-4}"
# Epoch-boundary validation runs the full (graph-aware) aligner over every val
# batch; cap it. Illumina-only -> a simple prefix cap is representative.
EPOCH_VAL_MAX_BATCHES="${EPOCH_VAL_MAX_BATCHES:-300}"

MANIFEST="${MANIFEST:-scripts/manifests/hg005_chr21_shortread.json}"
TEST_MANIFEST="${TEST_MANIFEST:-scripts/manifests/hg005_chr21_shortread.test.json}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_TRAIN="${OUT_TRAIN:-data/training_runs/hg005_chr21_shortread_pango_${STAMP}}"
OUT_TEST="${OUT_TEST:-data/eval_runs/hg005_shortread_pango_test_${STAMP}}"
CKPT="$OUT_TRAIN/checkpoint.pt"

echo "############################################################"
echo "# GraphMambaFormer HG005 SHORT-READ PANGENOME train+test  ($STAMP)"
echo "#   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  d_model=$D_MODEL"
echo "#   epochs=$EPOCHS batch=$BATCH  epoch_val_cap=$EPOCH_VAL_MAX_BATCHES"
echo "#   ref-mode=pangenome (GFA graph attached; GATv2 tower active)"
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
# pangenome mode needs the GFA the manifests point at.
GFA_PATH="$(python -c "import json,sys; print(json.load(open('$MANIFEST')).get('gfa',''))")"
if [ -z "$GFA_PATH" ] || [ ! -e "$GFA_PATH" ]; then
  echo "ERROR: pangenome mode needs a GFA; manifest 'gfa' = '$GFA_PATH' not found." >&2
  exit 1
fi
echo "pangenome graph: $GFA_PATH"

# --------------------------------------------------------------------------- #
# 1) TRAIN on chr21:5-6Mb (Illumina ~12x), PANGENOME graph -> train/val 80/20
# --------------------------------------------------------------------------- #
echo; echo "### [1/2] train HG005 short-read PANGENOME checkpoint -> $OUT_TRAIN"
python -u scripts/train.py \
  --data real \
  --manifest "$MANIFEST" \
  --ref-mode pangenome \
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
# 2) HELD-OUT TEST on chr21:6.5-7.0Mb, pangenome (two_pass)
# --------------------------------------------------------------------------- #
echo; echo "### [2/2] HELD-OUT TEST (chr21:6.5-7.0Mb, pangenome, two_pass) -> $OUT_TEST"
python -u scripts/eval.py \
  --data real \
  --manifest "$TEST_MANIFEST" \
  --ref-mode pangenome \
  --mode two_pass \
  --checkpoint "$CKPT" \
  --batch-size "$BATCH" \
  --workers "$WORKERS" \
  --device cuda --require-gpu \
  --no-bam \
  --out "$OUT_TEST"

echo
echo "############################################################"
echo "# DONE (HG005 short-read, PANGENOME)"
echo "#   checkpoint      : $CKPT"
echo "#   held-out test   : $OUT_TEST  (locus_accuracy on chr21:6.5-7.0Mb, graph-aware)"
echo "#   Compare vs the LINEAR short-read run on the same test region."
echo "############################################################"
