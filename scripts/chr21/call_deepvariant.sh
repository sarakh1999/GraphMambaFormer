#!/usr/bin/env bash
# DeepVariant small-variant calling on a mapped BAM (chr21).
# usage: call_deepvariant.sh <label> [bam]
#   label: giraffe | ours | bwa | ...
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

LABEL="${1:?usage: $0 <label:giraffe|ours|...> [bam]}"
BAM="${2:-$BAM_DIR/${SAMPLE}.${CHR}.${LABEL}.sorted.bam}"
OUT="$VCF_DIR/${SAMPLE}.${CHR}.${LABEL}.dv.vcf.gz"

require "$REF_FA" "run fetch_reference.sh"
require "$BAM"   "run map_${LABEL}.sh (or provide BAM)"

echo "DeepVariant ($LABEL) on $SAMPLE $CHR ..."
dock "$DV_IMAGE" /opt/deepvariant/bin/run_deepvariant \
  --model_type=WGS \
  --ref="$(inwork "$REF_FA")" \
  --reads="$(inwork "$BAM")" \
  --regions="$REGION" \
  --output_vcf="$(inwork "$OUT")" \
  --num_shards="$THREADS"

echo "Done: $OUT"
