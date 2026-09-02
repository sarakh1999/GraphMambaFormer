#!/usr/bin/env bash
# Build the BWA-MEM index for the linear $CHR reference (baseline arm).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require "$REF_FA" "run fetch_reference.sh first (or provide ref/${CHR}.fa)"

if [ -f "$REF_FA.bwt" ]; then
  echo "BWA index already present — skipping."
  ls -lh "$REF_FA".{amb,ann,bwt,pac,sa} 2>/dev/null
  exit 0
fi

echo "bwa index $REF_FA ..."
dock "$BWA_IMAGE" bwa index "$(inwork "$REF_FA")"
echo "Done:"
ls -lh "$REF_FA".{amb,ann,bwt,pac,sa} 2>/dev/null
