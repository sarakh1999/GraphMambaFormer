#!/usr/bin/env bash
# Download GIAB v4.2.1 GRCh38 benchmark for $SAMPLE and subset to $CHR.
# Produces:
#   data/fig6/$SAMPLE/truth/${SAMPLE}.${CHR}.benchmark.vcf.gz (+.tbi)
#   data/fig6/$SAMPLE/truth/${SAMPLE}.${CHR}.benchmark.bed
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

if [ -z "${GIAB_TRUTH_BASE:-}" ] || [ -z "${GIAB_TRUTH_VCF:-}" ] || [ -z "${GIAB_TRUTH_BED:-}" ]; then
  echo "ERROR: no GIAB truth defaults for SAMPLE=$SAMPLE" >&2
  echo "       set GIAB_TRUTH_BASE / GIAB_TRUTH_VCF / GIAB_TRUTH_BED" >&2
  exit 1
fi

cd "$TRUTH_DIR"
for f in "$GIAB_TRUTH_VCF" "$GIAB_TRUTH_VCF.tbi" "$GIAB_TRUTH_BED"; do
  [ -f "$f" ] || { echo "Downloading $f ..."; curl -L --fail -O "$GIAB_TRUTH_BASE/$f"; }
done

echo "Subsetting to $CHR ..."
dock "$BCFTOOLS_IMAGE" bcftools view -r "$CHR" "$(inwork "$TRUTH_DIR/$GIAB_TRUTH_VCF")" -Oz \
  -o "$(inwork "$TRUTH_VCF")"
dock "$BCFTOOLS_IMAGE" bcftools index -t "$(inwork "$TRUTH_VCF")"
awk -v c="$CHR" '$1==c' "$TRUTH_DIR/$GIAB_TRUTH_BED" > "$TRUTH_BED"

echo "Done:"
ls -lh "$TRUTH_VCF" "$TRUTH_VCF.tbi" "$TRUTH_BED"
wc -l "$TRUTH_BED"
