#!/usr/bin/env bash
# Sanity-check the downloaded full-genome HPRC GRCh38 Giraffe indexes:
# confirm vg runs, the GBZ loads, and show the chr20 reference path name.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require "$FULL_GBZ"

echo "== vg version =="
dockvg vg version
echo
echo "== graph stats (loads the ~4 GB GBZ; slow under emulation) =="
dockvg vg stats -z "$(inwork "$FULL_GBZ")"
echo
echo "== path names containing '$CHR' =="
dockvg vg paths -L -x "$(inwork "$FULL_GBZ")" | grep -i "$CHR" | head -20
