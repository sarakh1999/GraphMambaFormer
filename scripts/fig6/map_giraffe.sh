#!/usr/bin/env bash
# Map $SAMPLE $CHR reads to the pangenome with vg Giraffe -> sorted BAM.
# Arm: "HPRC Giraffe" in Figure 6a.
# Produces: data/fig6/$SAMPLE/bam/${SAMPLE}.${CHR}.giraffe.sorted.bam
#
# Newer vg autoindex emits *.shortread.withzip.min (+ optional *.zipcodes);
# older builds emit *.min. Accept either.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

OUT="$BAM_DIR/${SAMPLE}.${CHR}.giraffe.sorted.bam"
COMPAT_MARKER="$OUT.grch38-compatible"
TMP="$BAM_DIR/.${SAMPLE}.${CHR}.giraffe.tmp.bam"
REF_PATHS="$IDX_DIR/${CHR}.grch38.paths.txt"

if [ -s "$OUT" ] && [ -s "$OUT.bai" ] && [ -f "$COMPAT_MARKER" ]; then
  echo "Giraffe BAM already present — skipping."
  ls -lh "$OUT" "$OUT.bai"
  exit 0
fi

require "$CHR_PREFIX.giraffe.gbz" "run build_chr20_giraffe.sh first"
require "$CHR_PREFIX.dist"        "run build_chr20_giraffe.sh first"
require "$R1"; require "$R2"

MIN_HOST=""
for cand in "$CHR_PREFIX.min" "$CHR_PREFIX.shortread.withzip.min" "$CHR_PREFIX.shortread.min"; do
  if [ -f "$cand" ]; then MIN_HOST="$cand"; break; fi
done
[ -n "$MIN_HOST" ] || {
  echo "ERROR: missing minimizer index under $IDX_DIR (expected *.min or *.shortread.withzip.min)" >&2
  ls -lh "$IDX_DIR" >&2 || true
  exit 1
}

ZIP_HOST=""
for cand in "$CHR_PREFIX.shortread.zipcodes" "$CHR_PREFIX.zipcodes"; do
  if [ -f "$cand" ]; then ZIP_HOST="$cand"; break; fi
done

GBZ="$(inwork "$CHR_PREFIX.giraffe.gbz")"
MIN="$(inwork "$MIN_HOST")"
DIST="$(inwork "$CHR_PREFIX.dist")"
R1W="$(inwork "$R1")"; R2W="$(inwork "$R2")"
TMPW="$(inwork "$TMP")"; OUTW="$(inwork "$OUT")"
ZIP_ARG=""
[ -n "$ZIP_HOST" ] && ZIP_ARG="-z $(inwork "$ZIP_HOST")"

# HPRC chr graphs include both CHM13 and GRCh38. Restrict BAM surjection to
# GRCh38, then remove the PanSN prefix so the BAM matches the chr20 FASTA.
printf 'GRCh38#0#%s\n' "$CHR" > "$REF_PATHS"
REF_PATHSW="$(inwork "$REF_PATHS")"
RG="ID:${SAMPLE} SM:${SAMPLE} PL:ILLUMINA LB:${SAMPLE}"

echo "Giraffe mapping $SAMPLE $CHR -> sorted BAM ..."
echo "  gbz=$GBZ"
echo "  min=$MIN"
echo "  dist=$DIST"
[ -n "$ZIP_HOST" ] && echo "  zip=$ZIP_HOST"

rm -f "$TMP" "$TMP.bai"
dockvg bash -c "set -o pipefail; vg giraffe -Z '$GBZ' -m '$MIN' -d '$DIST' $ZIP_ARG --ref-paths '$REF_PATHSW' --sample '$SAMPLE' --read-group '$RG' -f '$R1W' -f '$R2W' -o BAM -t $THREADS -p | samtools view -h - | sed 's/GRCh38#0#$CHR/$CHR/g' | samtools sort -@ $THREADS -o '$TMPW' - && samtools index '$TMPW'"

# Verify reference compatibility before replacing an earlier BAM.
dockvg samtools view -H "$TMPW" | grep -q $'\tSN:'"$CHR"$'\t'
dockvg samtools view -H "$TMPW" | grep -q $'\tSM:'"$SAMPLE"
mv "$TMP" "$OUT"
mv "$TMP.bai" "$OUT.bai"
touch "$COMPAT_MARKER"

echo "Done: $OUT"
dockvg samtools flagstat "$OUTW" || true
