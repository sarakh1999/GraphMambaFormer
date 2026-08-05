#!/usr/bin/env bash
# GraphMambaFormer ("our implementation") mapping arm.
#
# Runs the seed->chain->extend->score pipeline (graphmambaformer.alignment) on
# the fetched chr21 reads and writes a sorted+indexed BAM whose @SQ name matches
# the reference FASTA, so DeepVariant/hap.py accept it like the Giraffe BAM.
#
# This arm runs entirely in Python via pysam — no Docker required.
#
# Resolution order:
#   1) OURS_BAM=/path/to/sorted.bam   -> install that BAM (external aligner)
#   2) existing BAM at the default path -> reuse
#   3) otherwise run align_ours.py on data/chr21/<SAMPLE>/reads/*.fastq.gz
#
# Knobs:
#   OURS_MODE=fast|hybrid|two_pass   (default fast)
#   OURS_MAX_READS=N                 (cap reads for a quick pass; 0/unset = all)
#   PYTHON=/path/to/python           (default: repo .venv, else python3)
#
# usage:
#   ./scripts/chr21/map_ours.sh
#   OURS_MAX_READS=50000 ./scripts/chr21/map_ours.sh
#   OURS_BAM=/path/to/ours.sorted.bam ./scripts/chr21/map_ours.sh
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

# 3) Run the aligner.
require "$REF_FA" "run fetch_reference.sh first"
R1="$READS_DIR/${SAMPLE}.${CHR}.R1.fastq.gz"
R2="$READS_DIR/${SAMPLE}.${CHR}.R2.fastq.gz"
require "$R1" "run fetch_reads.sh first"

READS_ARGS=(--reads "$R1")
[ -f "$R2" ] && READS_ARGS+=(--reads "$R2")

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

echo "GraphMambaFormer mapping ($CHR, mode=${OURS_MODE:-fast}) -> $OUT"
[ -n "${OURS_MAX_READS:-}" ] && [ "${OURS_MAX_READS}" != "0" ] \
  && echo "note: OURS_MAX_READS=${OURS_MAX_READS} (partial pass for speed)"

"$PYTHON" "$D/align_ours.py" \
  --ref "$REF_FA" \
  "${READS_ARGS[@]}" \
  --out "$OUT" \
  --mode "${OURS_MODE:-fast}"

echo "Done: $OUT"
