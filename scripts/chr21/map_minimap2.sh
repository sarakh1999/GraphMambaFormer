#!/usr/bin/env bash
# Map $SAMPLE $CHR LONG reads to the linear reference with minimap2 -> sorted BAM.
# Long-read baseline arm (the tool your NN competes with on HiFi/ONT).
#
# usage: map_minimap2.sh [pacbio_hifi|ont] [reads.fastq.gz]
#   modality picks the minimap2 preset: pacbio_hifi -> map-hifi, ont -> map-ont
# Produces: data/chr21/$SAMPLE/bam/${SAMPLE}.${CHR}.minimap2.<modality>.sorted.bam
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

MODALITY="${1:-pacbio_hifi}"
case "$MODALITY" in
  pacbio_hifi|hifi) PRESET="map-hifi"; TAG="pacbio_hifi" ;;
  ont|nanopore)     PRESET="map-ont";  TAG="ont" ;;
  *) echo "ERROR: modality must be pacbio_hifi or ont (got '$MODALITY')" >&2; exit 2 ;;
esac

READS="${2:-$READS_DIR/${SAMPLE}.${CHR}.${TAG}.fastq.gz}"
SAM="$BAM_DIR/${SAMPLE}.${CHR}.minimap2.${TAG}.sam"
OUT="$BAM_DIR/${SAMPLE}.${CHR}.minimap2.${TAG}.sorted.bam"
RG="@RG\\tID:${SAMPLE}\\tSM:${SAMPLE}\\tPL:$([ "$TAG" = ont ] && echo ONT || echo PACBIO)\\tLB:${SAMPLE}"

if [ -s "$OUT" ] && [ -s "$OUT.bai" ]; then
  echo "minimap2 BAM already present — skipping."; ls -lh "$OUT" "$OUT.bai"; exit 0
fi

require "$REF_FA" "run fetch_reference.sh first (or provide ref/${CHR}.fa)"
require "$READS"  "long reads for $TAG (run fetch/prepare long reads first)"

REF_W="$(inwork "$REF_FA")"; READSW="$(inwork "$READS")"
SAMW="$(inwork "$SAM")"; OUTW="$(inwork "$OUT")"

# minimap2 image may not bundle samtools, so emit SAM here and sort via the vg
# image's samtools (same pattern as map_bwa.sh).
echo "[1/2] minimap2 -ax $PRESET ($SAMPLE $CHR $TAG) ..."
dock "$MINIMAP_IMAGE" minimap2 -ax "$PRESET" -R "$RG" -t "$THREADS" "$REF_W" "$READSW" > "$SAM"

echo "[2/2] sort + index ..."
dockvg bash -c "samtools sort -@ $THREADS '$SAMW' -o '$OUTW' && samtools index '$OUTW'"
rm -f "$SAM"

echo "Done: $OUT"
dockvg samtools flagstat "$OUTW" || true
