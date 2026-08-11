#!/usr/bin/env bash
# Download GIAB benchmark for SAMPLE and subset to $CHR.
# Built-in truth defaults are HG002-only; other samples need GIAB_TRUTH_* overrides.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RESOLVE="$(dirname "${BASH_SOURCE[0]}")/resolve_sample.py"

if [ "${SKIP_TRUTH:-0}" = "1" ]; then
  echo "SKIP_TRUTH=1 — skipping GIAB truth download for SAMPLE=$SAMPLE"
  exit 0
fi

BASE="${GIAB_TRUTH_BASE:-$(python3 "$RESOLVE" "$SAMPLE" truth_base 2>/dev/null || true)}"
VCF="${GIAB_TRUTH_VCF:-$(python3 "$RESOLVE" "$SAMPLE" truth_vcf 2>/dev/null || true)}"
BED="${GIAB_TRUTH_BED:-$(python3 "$RESOLVE" "$SAMPLE" truth_bed 2>/dev/null || true)}"

if [ -z "$BASE" ] || [ -z "$VCF" ] || [ -z "$BED" ]; then
  echo "NOTE: no built-in GIAB truth for SAMPLE=$SAMPLE." >&2
  echo "      Set GIAB_TRUTH_BASE / GIAB_TRUTH_VCF / GIAB_TRUTH_BED," >&2
  echo "      or run with SKIP_TRUTH=1 for HPRC-only samples like HG00438." >&2
  exit 1
fi

cd "$TRUTH_DIR"
for f in "$VCF" "$VCF.tbi" "$BED"; do
  [ -f "$f" ] || { echo "Downloading $f ..."; curl -L --fail -O "$BASE/$f"; }
done

echo "Subsetting to $CHR ..."
dock "$BCFTOOLS_IMAGE" bcftools view -r "$CHR" "$(inwork "$TRUTH_DIR/$VCF")" -Oz \
  -o "$(inwork "$TRUTH_VCF")"
dock "$BCFTOOLS_IMAGE" bcftools index -t "$(inwork "$TRUTH_VCF")"
awk -v c="$CHR" '$1==c' "$TRUTH_DIR/$BED" > "$TRUTH_BED"

echo "Done:"; ls -lh "$TRUTH_VCF" "$TRUTH_VCF.tbi" "$TRUTH_BED"
