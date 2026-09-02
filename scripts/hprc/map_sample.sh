#!/usr/bin/env bash
# Map one HPRC sample (HiFi FASTQ.gz / Illumina CRAM / ONT BAM) with align_ours.py.
#
# Layout expected:
#   ${READS_ROOT}/${SAMPLE}/{hifi,illumina,ont}/
#
# Knobs:
#   SAMPLE=HG00438              (required)
#   REF=/path/to/ref.fa         (required unless OURS_REF set)
#   READS_ROOT=data/hprc/reads
#   MODALITIES=illumina,hifi,ont   # or a subset: hifi | illumina | ont
#   BAM_MODE=combined|separate
#   OUT=...                     # default data/hprc/bam/<SAMPLE>.ours.sorted.bam
#   OURS_INDEX_CACHE=...        # reuse Stage-1 index across runs
#   OURS_MAX_READS=N            # smoke cap
#   OURS_WORKERS / OURS_BATCH_SIZE / OURS_DEVICE / OURS_REGION / OURS_SAM
#
# usage:
#   SAMPLE=HG00438 REF=data/chr21/HG002/ref/GRCh38.chr21.fa ./scripts/hprc/map_sample.sh
#   SAMPLE=HG00621 MODALITIES=hifi ./scripts/hprc/map_sample.sh
#   SAMPLE=HG00673 BAM_MODE=separate ./scripts/hprc/map_sample.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SAMPLE="${SAMPLE:?set SAMPLE=HG00438|HG00621|HG00673|...}"
REF="${REF:-${OURS_REF:-}}"
if [ -z "$REF" ]; then
  echo "ERROR: set REF=/path/to/reference.fasta" >&2
  exit 1
fi
READS_ROOT="${READS_ROOT:-${OURS_READS_ROOT:-data/hprc/reads}}"
MODALITIES="${MODALITIES:-${OURS_MODALITIES:-illumina,hifi,ont}}"
BAM_MODE="${BAM_MODE:-${OURS_BAM_MODE:-combined}}"
OUT="${OUT:-data/hprc/bam/${SAMPLE}.ours.sorted.bam}"
INDEX_CACHE="${OURS_INDEX_CACHE:-data/hprc/index_cache}"

PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
  if [ -x "$ROOT/.venv/bin/python" ]; then PYTHON="$ROOT/.venv/bin/python"
  else PYTHON="python3"; fi
fi

mkdir -p "$(dirname "$OUT")" "$INDEX_CACHE"

SPEED_ARGS=(--index-cache "$INDEX_CACHE")
[ -n "${OURS_WORKERS:-${GMF_NUM_WORKERS:-}}" ] && SPEED_ARGS+=(--workers "${OURS_WORKERS:-$GMF_NUM_WORKERS}")
[ -n "${OURS_BATCH_SIZE:-}" ] && SPEED_ARGS+=(--batch-size "$OURS_BATCH_SIZE")
[ -n "${OURS_DEVICE:-}" ] && SPEED_ARGS+=(--device "$OURS_DEVICE")
[ -n "${OURS_REGION:-}" ] && SPEED_ARGS+=(--region "$OURS_REGION")
[ -n "${OURS_MAX_READS:-}" ] && [ "${OURS_MAX_READS}" != "0" ] && SPEED_ARGS+=(--max-reads "$OURS_MAX_READS")
[ "${OURS_SAM:-0}" = "1" ] && SPEED_ARGS+=(--sam)

echo "HPRC map: sample=$SAMPLE modalities=$MODALITIES bam-mode=$BAM_MODE"
echo "  ref=$REF"
echo "  reads-root=$READS_ROOT"
echo "  out=$OUT"

PYTHONPATH="$ROOT" "$PYTHON" "$ROOT/scripts/chr21/align_ours.py" \
  --ref "$REF" \
  --sample "$SAMPLE" \
  --reads-root "$READS_ROOT" \
  --modalities "$MODALITIES" \
  --bam-mode "$BAM_MODE" \
  --out "$OUT" \
  --mode "${OURS_MODE:-fast}" \
  "${SPEED_ARGS[@]}"

echo "Done: $OUT"
[ "$BAM_MODE" = "separate" ] && echo "note: per-modality BAMs use stem ${OUT%.bam}.<modality>.bam"
