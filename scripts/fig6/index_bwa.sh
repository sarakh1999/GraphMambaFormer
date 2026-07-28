#!/usr/bin/env bash
# Build the BWA index for the linear $CHR reference (shared).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require "$REF_FA" "run fetch_reference.sh first"

if [ -f "$REF_FA.bwt" ]; then
  echo "BWA index already present — skipping."
  ls -lh "$REF_FA".{amb,ann,bwt,pac,sa}
  exit 0
fi

echo "bwa index $REF_FA ..."
dock "$BWA_IMAGE" bwa index "$(inwork "$REF_FA")"
echo "Done:"
ls -lh "$REF_FA".{amb,ann,bwt,pac,sa} 2>/dev/null
