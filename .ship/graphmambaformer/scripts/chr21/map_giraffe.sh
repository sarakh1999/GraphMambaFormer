#!/usr/bin/env bash
# Map SAMPLE short reads with vg Giraffe -> sorted BAM.
# Produces: data/chr21/<SAMPLE>/bam/<SAMPLE>.chr21.giraffe.sorted.bam
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

R1="${1:-$READS_DIR/${SAMPLE}.${CHR}.R1.fastq.gz}"
R2="${2:-$READS_DIR/${SAMPLE}.${CHR}.R2.fastq.gz}"
OUT="$BAM_DIR/${SAMPLE}.${CHR}.giraffe.sorted.bam"

for f in "$CHR_PREFIX.giraffe.gbz" "$CHR_PREFIX.min" "$CHR_PREFIX.dist"; do
  require "$f" "run build_giraffe.sh first"
done
require "$R1"; require "$R2"

GBZ="$(inwork "$CHR_PREFIX.giraffe.gbz")"
MIN="$(inwork "$CHR_PREFIX.min")"
DIST="$(inwork "$CHR_PREFIX.dist")"
R1W="$(inwork "$R1")"; R2W="$(inwork "$R2")"; OUTW="$(inwork "$OUT")"

echo "Giraffe mapping -> sorted BAM ..."
dockvg bash -c "vg giraffe -Z '$GBZ' -m '$MIN' -d '$DIST' -f '$R1W' -f '$R2W' -o BAM -t $THREADS -p | samtools sort -@ $THREADS -o '$OUTW' - && samtools index '$OUTW'"

echo "Done: $OUT"
dockvg samtools flagstat "$OUTW" || true
