#!/usr/bin/env bash
# Map $SAMPLE $CHR reads to linear GRCh38 with BWA-MEM -> sorted BAM.
# Arm: "BWAMEM" in Figure 6a.
# Produces: data/fig6/$SAMPLE/bam/${SAMPLE}.${CHR}.bwa.sorted.bam
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

SAM="$BAM_DIR/${SAMPLE}.${CHR}.bwa.sam"
OUT="$BAM_DIR/${SAMPLE}.${CHR}.bwa.sorted.bam"
# BWA expects the two characters "\t", not literal tab bytes.
RG="@RG\\tID:${SAMPLE}\\tSM:${SAMPLE}\\tPL:ILLUMINA\\tLB:${SAMPLE}"

if [ -s "$OUT" ] && [ -s "$OUT.bai" ]; then
  echo "BWA BAM already present — skipping."
  ls -lh "$OUT" "$OUT.bai"
  exit 0
fi

require "$REF_FA.bwt" "run index_bwa.sh first"
require "$R1"; require "$R2"

REF_W="$(inwork "$REF_FA")"; R1W="$(inwork "$R1")"; R2W="$(inwork "$R2")"
SAMW="$(inwork "$SAM")"; OUTW="$(inwork "$OUT")"

echo "[1/2] bwa mem ($SAMPLE $CHR) ..."
dock "$BWA_IMAGE" bwa mem -t "$THREADS" -R "$RG" "$REF_W" "$R1W" "$R2W" > "$SAM"

echo "[2/2] sort + index ..."
dockvg bash -c "samtools sort -@ $THREADS '$SAMW' -o '$OUTW' && samtools index '$OUTW'"
rm -f "$SAM"

echo "Done: $OUT"
dockvg samtools flagstat "$OUTW" || true
