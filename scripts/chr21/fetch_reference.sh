#!/usr/bin/env bash
# Download GRCh38 and extract $CHR FASTA.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

URL="${GRCH38_URL:-https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/references/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta.gz}"

echo "Downloading/resuming GRCh38 (~900 MB) ..."
curl -L --fail -C - -o "$REF_FULL_GZ" "$URL" || curl -L --fail -o "$REF_FULL_GZ" "$URL"

REF_FULL_W="$(inwork "$REF_FULL_GZ")"
REF_FA_W="$(inwork "$REF_FA")"

echo "Indexing whole-genome FASTA ..."
dockvg samtools faidx "$REF_FULL_W"

echo "Extracting $CHR ..."
dockvg samtools faidx "$REF_FULL_W" "$CHR" > "$REF_FA"
dockvg samtools faidx "$REF_FA_W"

echo "Done:"; ls -lh "$REF_FA" "$REF_FA.fai"
