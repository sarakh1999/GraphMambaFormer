#!/usr/bin/env bash
# Call variants with DeepVariant on a mapped BAM ($CHR only).
# usage: call_deepvariant.sh <label> [bam]   e.g. call_deepvariant.sh giraffe
# Produces: data/fig6/$SAMPLE/vcf/${SAMPLE}.${CHR}.<label>.dv.vcf.gz
#
# Notes for Apple Silicon / Docker Desktop:
#   - DeepVariant's call_variants multiprocessing needs a large /dev/shm.
#     Without --shm-size it often dies with `_queue.Empty` after make_examples.
#   - Intermediate TFRecords are kept under data/fig6/$SAMPLE/dv/<label>/ so a
#     failed call_variants does not force a full multi-hour redo.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

LABEL="${1:?usage: $0 <label:giraffe|bwa> [bam]}"
BAM="${2:-$BAM_DIR/${SAMPLE}.${CHR}.${LABEL}.sorted.bam}"
OUT="$VCF_DIR/${SAMPLE}.${CHR}.${LABEL}.dv.vcf.gz"
DV_DIR="$RUN_DIR/dv/${LABEL}"
EXAMPLES="$DV_DIR/make_examples.tfrecord@${THREADS}.gz"
CALL_REQUEST="$DV_DIR/call_variants_output.tfrecord.gz"
# DeepVariant writes a sharded filename even when only one output shard exists.
CALL_OUT="$DV_DIR/call_variants_output-00000-of-00001.tfrecord.gz"
CALL_INPUT_SPEC="$DV_DIR/call_variants_output@1.tfrecord.gz"
SHM_SIZE="${DV_SHM_SIZE:-8g}"

mkdir -p "$DV_DIR"

if [ -s "$OUT" ] && { [ -s "$OUT.tbi" ] || [ -s "$OUT.csi" ]; }; then
  echo "DeepVariant output already present — skipping."
  ls -lh "$OUT" "$OUT.tbi" "$OUT.csi" 2>/dev/null || true
  exit 0
fi

require "$REF_FA" "run fetch_reference.sh"
require "$BAM"   "run map_${LABEL}.sh"

# DeepVariant needs larger shared memory than Docker's default 64MB. In the
# runner image the binaries are local and /dev/shm was already sized by
# docker/run.sh.
dock_dv() {
  local img="$1"; shift
  if [ "$FIG6_NATIVE" = "1" ] && have_native "$1"; then
    "$@"
    return
  fi
  docker run --rm --platform "$PLATFORM" --shm-size="$SHM_SIZE" \
    -v "$FIG6_HOST_ROOT:/work" -w /work "$img" "$@"
}

examples_ready() {
  # All shard files present and non-empty.
  local i f
  for ((i=0; i<THREADS; i++)); do
    f="$(printf '%s/make_examples.tfrecord-%05d-of-%05d.gz' "$DV_DIR" "$i" "$THREADS")"
    [ -s "$f" ] || return 1
  done
  return 0
}

REF_W="$(inwork "$REF_FA")"
BAM_W="$(inwork "$BAM")"
OUT_W="$(inwork "$OUT")"
DV_W="$(inwork "$DV_DIR")"
EX_W="$(inwork "$DV_DIR")/make_examples.tfrecord@${THREADS}.gz"
CALL_REQUEST_W="$(inwork "$CALL_REQUEST")"
CALL_W="$(inwork "$CALL_OUT")"
CALL_INPUT_W="$(inwork "$CALL_INPUT_SPEC")"

if examples_ready; then
  echo "Reusing existing make_examples shards in $DV_DIR"
else
  echo "DeepVariant make_examples ($LABEL) — slow under x86 emulation ..."
  # Clear partial shards from a prior crash.
  rm -f "$DV_DIR"/make_examples.tfrecord-*.gz "$DV_DIR"/make_examples.tfrecord-*.json
  dock_dv "$DV_IMAGE" bash -c "
    set -euo pipefail
    seq 0 $((THREADS-1)) | parallel -q --halt 2 --line-buffer \
      /opt/deepvariant/bin/make_examples \
        --mode calling \
        --ref '$REF_W' \
        --reads '$BAM_W' \
        --examples '$EX_W' \
        --channels insert_size \
        --regions '$CHR' \
        --sample_name '$SAMPLE' \
        --task {}
  "
  examples_ready || { echo "ERROR: make_examples did not finish all shards" >&2; exit 1; }
fi

if [ -s "$CALL_OUT" ]; then
  echo "Reusing existing call_variants output: $CALL_OUT"
else
  echo "DeepVariant call_variants ($LABEL) ..."
  rm -f "$CALL_REQUEST" "$CALL_OUT"
  dock_dv "$DV_IMAGE" /opt/deepvariant/bin/call_variants \
    --outfile "$CALL_REQUEST_W" \
    --examples "$EX_W" \
    --checkpoint /opt/models/wgs
fi

require "$CALL_OUT" "call_variants produced no output"

echo "DeepVariant postprocess_variants ($LABEL) ..."
dock_dv "$DV_IMAGE" /opt/deepvariant/bin/postprocess_variants \
  --ref "$REF_W" \
  --infile "$CALL_INPUT_W" \
  --outfile "$OUT_W"

require "$OUT" "DeepVariant did not produce a VCF"
# Index if the binary did not.
if [ ! -s "$OUT.tbi" ] && [ ! -s "$OUT.csi" ]; then
  dock "$BCFTOOLS_IMAGE" bcftools index -t "$(inwork "$OUT")"
fi
require "$OUT.tbi" "DeepVariant VCF index missing"
echo "Done: $OUT"
ls -lh "$OUT" "$OUT.tbi"
