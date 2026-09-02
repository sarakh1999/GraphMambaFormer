#!/usr/bin/env bash
# Score a query VCF against GIAB truth with hap.py (vcfeval) -> ROC for PR curves.
# usage: eval_happy.sh <label>
# Produces: data/fig6/$SAMPLE/eval/<label>/${SAMPLE}.${CHR}.<label>.summary.csv + .roc.all.csv.gz
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

LABEL="${1:?usage: $0 <label:giraffe|bwa>}"
QUERY="${2:-$VCF_DIR/${SAMPLE}.${CHR}.${LABEL}.dv.vcf.gz}"
OUTDIR="$EVAL_DIR/$LABEL"
PREFIX="$OUTDIR/${SAMPLE}.${CHR}.${LABEL}"
mkdir -p "$OUTDIR"

require "$TRUTH_VCF" "run fetch_truth.sh"
require "$TRUTH_BED" "run fetch_truth.sh"
require "$REF_FA"    "run fetch_reference.sh"
require "$QUERY"     "run call_deepvariant.sh $LABEL"

echo "hap.py: $LABEL vs GIAB $SAMPLE truth on $CHR ..."
dock "$HAPPY_IMAGE" /opt/hap.py/bin/hap.py \
  "$(inwork "$TRUTH_VCF")" "$(inwork "$QUERY")" \
  -f "$(inwork "$TRUTH_BED")" \
  -r "$(inwork "$REF_FA")" \
  -o "$(inwork "$PREFIX")" \
  -l "$CHR" \
  --engine=vcfeval --threads "$THREADS"

echo "Done. Summary:"
cat "$PREFIX.summary.csv" 2>/dev/null || true
