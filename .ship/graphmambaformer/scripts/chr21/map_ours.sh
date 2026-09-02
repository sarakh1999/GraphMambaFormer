#!/usr/bin/env bash
# GraphMambaFormer ("our implementation") mapping arm.
#
# Runs the seed->chain->extend->score pipeline (graphmambaformer.alignment) on
# the fetched chr21 reads and writes a sorted+indexed BAM whose @SQ name matches
# the reference FASTA, so DeepVariant/hap.py accept it like the Giraffe BAM.
#
# This arm runs entirely in Python via pysam — no Docker required.
#
# Short and long reads can be aligned in one command. Default BAM output when
# both are present is a single combined file with per-modality @RG/XM tags.
# Set OURS_BAM_MODE=separate for one BAM per modality.
#
# Resolution order:
#   1) OURS_BAM=/path/to/sorted.bam   -> install that BAM (external aligner)
#   2) existing BAM at the default path -> reuse
#   3) otherwise run align_ours.py on short and/or long reads
#
# Knobs:
#   OURS_MODE=fast|hybrid|two_pass   (default fast)
#   OURS_MAX_READS=N                 (cap reads for a quick pass; 0/unset = all)
#   OURS_CHECKPOINT=/path/checkpoint.pt  (for hybrid/two_pass neural scoring)
#   OURS_FORCE=1                     (rebuild even if BAM exists)
#   OURS_LONG_READS=/path/hifi.fq[.gz]   (optional long reads; also tries HIFI_FASTQ)
#   OURS_LONG_MODALITY=pacbio_hifi|ont   (default pacbio_hifi)
#   OURS_BAM_MODE=combined|separate|auto (default combined)
#   PYTHON=/path/to/python           (default: repo .venv, else python3)
#
# usage:
#   ./scripts/chr21/map_ours.sh
#   OURS_MAX_READS=50000 ./scripts/chr21/map_ours.sh
#   OURS_BAM=/path/to/ours.sorted.bam ./scripts/chr21/map_ours.sh
#   OURS_LONG_READS=/path/hifi.fq.gz OURS_BAM_MODE=combined ./scripts/chr21/map_ours.sh
#   OURS_MODE=hybrid OURS_CHECKPOINT=data/training_runs/latest/checkpoint.pt ./scripts/chr21/map_ours.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
D="$(dirname "${BASH_SOURCE[0]}")"

OUT="$BAM_DIR/${SAMPLE}.${CHR}.ours.sorted.bam"
SRC="${OURS_BAM:-}"

# Pick a Python interpreter: prefer the repo venv, then $PYTHON, then python3.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"
  else PYTHON="python3"; fi
fi

py_index() {  # index a BAM with pysam (no external samtools / docker)
  "$PYTHON" -c "import pysam,sys; pysam.index(sys.argv[1])" "$1"
}

# 1) External BAM provided.
if [ -n "$SRC" ]; then
  require "$SRC"
  cp -f "$SRC" "$OUT"
  if   [ -f "${SRC}.bai" ];        then cp -f "${SRC}.bai" "${OUT}.bai"
  elif [ -f "${SRC%.bam}.bam.bai" ]; then cp -f "${SRC%.bam}.bam.bai" "${OUT}.bai"
  fi
  [ -f "${OUT}.bai" ] || py_index "$OUT"
  echo "Installed our BAM: $OUT"
  exit 0
fi

# 2) Reuse an existing BAM.
if [ -f "$OUT" ] && [ "${OURS_FORCE:-0}" != "1" ]; then
  echo "Found existing our BAM: $OUT  (set OURS_FORCE=1 to rebuild)"
  [ -f "${OUT}.bai" ] || py_index "$OUT"
  exit 0
fi

# 3) Run the aligner (short and/or long reads in one command).
require "$REF_FA" "run fetch_reference.sh first"
R1="$READS_DIR/${SAMPLE}.${CHR}.R1.fastq.gz"
R2="$READS_DIR/${SAMPLE}.${CHR}.R2.fastq.gz"
LONG="${OURS_LONG_READS:-${HIFI_FASTQ:-}}"

READS_ARGS=()
if [ -f "$R1" ]; then
  READS_ARGS+=(--reads "$R1")
  [ -f "$R2" ] && READS_ARGS+=(--reads "$R2")
elif [ -z "$LONG" ]; then
  echo "ERROR: no short reads at $R1 and no OURS_LONG_READS/HIFI_FASTQ" >&2
  echo "run fetch_reads.sh first, or set OURS_LONG_READS=/path/to/long.fastq.gz" >&2
  exit 1
fi

LONG_ARGS=()
if [ -n "$LONG" ]; then
  require "$LONG"
  LONG_ARGS+=(--long-reads "$LONG")
  LONG_ARGS+=(--long-modality "${OURS_LONG_MODALITY:-pacbio_hifi}")
fi

if ! PYTHONPATH="$ROOT" "$PYTHON" -c "import torch, numpy, pysam, graphmambaformer" 2>/dev/null; then
  cat >&2 <<EOF
ERROR: Python deps for the GraphMambaFormer arm are missing.
Install them, e.g.:
  python3 -m venv .venv && . .venv/bin/activate
  pip install -r requirements.txt
Then re-run: $0
EOF
  exit 1
fi

BAM_MODE="${OURS_BAM_MODE:-combined}"
echo "GraphMambaFormer mapping ($CHR, mode=${OURS_MODE:-fast}, bam-mode=$BAM_MODE) -> $OUT"
[ -n "${OURS_MAX_READS:-}" ] && [ "${OURS_MAX_READS}" != "0" ] \
  && echo "note: OURS_MAX_READS=${OURS_MAX_READS} (partial pass for speed)"
[ -n "${OURS_CHECKPOINT:-}" ] && echo "note: OURS_CHECKPOINT=${OURS_CHECKPOINT}"
[ -n "$LONG" ] && echo "note: long reads=$LONG (${OURS_LONG_MODALITY:-pacbio_hifi})"
[ -n "${OURS_INDEX_CACHE:-}" ] && echo "note: index-cache=${OURS_INDEX_CACHE}"

CKPT_ARGS=()
[ -n "${OURS_CHECKPOINT:-}" ] && CKPT_ARGS+=(--checkpoint "$OURS_CHECKPOINT")
[ -n "${OURS_D_MODEL:-}" ] && CKPT_ARGS+=(--d-model "$OURS_D_MODEL")

SPEED_ARGS=()
[ -n "${OURS_WORKERS:-${GMF_NUM_WORKERS:-}}" ] && SPEED_ARGS+=(--workers "${OURS_WORKERS:-$GMF_NUM_WORKERS}")
[ -n "${OURS_BATCH_SIZE:-}" ] && SPEED_ARGS+=(--batch-size "$OURS_BATCH_SIZE")
[ -n "${OURS_INDEX_CACHE:-}" ] && SPEED_ARGS+=(--index-cache "$OURS_INDEX_CACHE")
[ -n "${OURS_DEVICE:-}" ] && SPEED_ARGS+=(--device "$OURS_DEVICE")
[ "${OURS_SAM:-0}" = "1" ] && SPEED_ARGS+=(--sam)

"$PYTHON" "$D/align_ours.py" \
  --ref "$REF_FA" \
  "${READS_ARGS[@]}" \
  "${LONG_ARGS[@]}" \
  --out "$OUT" \
  --mode "${OURS_MODE:-fast}" \
  --bam-mode "$BAM_MODE" \
  "${SPEED_ARGS[@]}" \
  "${CKPT_ARGS[@]}"

# DeepVariant / compare.sh look for the default OUT path. When we wrote
# separate modality BAMs, point OUT at the short-read file so callers keep working.
if [ "$BAM_MODE" = "separate" ] && [ "${#READS_ARGS[@]}" -gt 0 ]; then
  SHORT_OUT="${OUT%.bam}.illumina.bam"
  if [ -f "$SHORT_OUT" ]; then
    cp -f "$SHORT_OUT" "$OUT"
    if [ -f "${SHORT_OUT}.bai" ]; then cp -f "${SHORT_OUT}.bai" "${OUT}.bai"
    else py_index "$OUT"; fi
    echo "note: installed short-read BAM as $OUT for DeepVariant (from $SHORT_OUT)"
  fi
fi

echo "Done: $OUT"
if [ "$BAM_MODE" = "separate" ]; then
  echo "note: separate modality BAMs use stem ${OUT%.bam}.<modality>.bam"
elif [ -n "$LONG" ] && [ "${#READS_ARGS[@]}" -gt 0 ]; then
  echo "note: combined BAM mixes short+long (@RG tags). For DeepVariant-only short reads, re-run with OURS_BAM_MODE=separate"
fi
