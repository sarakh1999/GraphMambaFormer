#!/usr/bin/env bash
# GraphMambaFormer ("our implementation") mapping arm.
#
# The cross-attention alignment decoder is not shipped yet, so this script
# accepts an externally produced BAM (set OURS_BAM=...) or fails with a clear
# message. Once the decoder emits BAM, drop it at:
#   data/chr21/<SAMPLE>/bam/<SAMPLE>.chr21.ours.sorted.bam
# and re-run DeepVariant with label "ours".
#
# usage:
#   OURS_BAM=/path/to/ours.sorted.bam ./scripts/chr21/map_ours.sh
#   # or place the BAM at the default path and run: ./scripts/chr21/map_ours.sh
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

OUT="$BAM_DIR/${SAMPLE}.${CHR}.ours.sorted.bam"
SRC="${OURS_BAM:-}"

if [ -n "$SRC" ]; then
  require "$SRC"
  cp -f "$SRC" "$OUT"
  [ -f "${SRC}.bai" ] && cp -f "${SRC}.bai" "${OUT}.bai"
  [ -f "${SRC%.bam}.bam.bai" ] && cp -f "${SRC%.bam}.bam.bai" "${OUT}.bai"
  if [ ! -f "${OUT}.bai" ]; then
    dockvg samtools index "$(inwork "$OUT")"
  fi
  echo "Installed our BAM: $OUT"
  exit 0
fi

if [ -f "$OUT" ]; then
  echo "Found existing our BAM: $OUT"
  [ -f "${OUT}.bai" ] || dockvg samtools index "$(inwork "$OUT")"
  exit 0
fi

cat >&2 <<EOF
ERROR: GraphMambaFormer BAM not found for SAMPLE=$SAMPLE CHR=$CHR

The alignment decoder (Figure 1C) is not implemented yet, so this arm is a
hook. Produce a sorted+indexed BAM and either:

  1) place it at:
       $OUT
  2) or run:
       OURS_BAM=/path/to/sorted.bam $0

Then continue with:
  ./scripts/chr21/call_deepvariant.sh ours
EOF
exit 1
