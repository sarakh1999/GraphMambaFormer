#!/usr/bin/env bash
# Stream $SAMPLE short reads for $CHR from a remote GIAB GRCh38 BAM/CRAM
# (HTTP range requests) -> paired FASTQ. Does NOT download the whole genome.
#
# Uses biocontainers/samtools (libcurl/HTTPS). The vg image's samtools often
# fails with "Protocol not supported" on https:// URLs.
#
# Produces:
#   data/fig6/$SAMPLE/reads/${SAMPLE}.${CHR}.R{1,2}.fastq.gz
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

ALN="${READS_ALN_URL:-}"
if [ -z "$ALN" ]; then
  echo "ERROR: set READS_ALN_URL (or use SAMPLE=HG002 / HG005)." >&2
  exit 1
fi

if [ -s "$R1" ] && [ -s "$R2" ]; then
  echo "FASTQs already present — skipping."
  ls -lh "$R1" "$R2"
  exit 0
fi

R1W="$(inwork "$R1")"; R2W="$(inwork "$R2")"

TARG=""
case "$ALN" in
  *.cram) require "$REF_FULL_GZ" "run fetch_reference.sh first"
          TARG="-T $(inwork "$REF_FULL_GZ")";;
esac

# High-coverage source? DeepVariant WGS model wants ~30x.
if [ -z "${DOWNSAMPLE:-}" ] && [[ "$ALN" == *300x* ]]; then
  DOWNSAMPLE=0.1
  echo "note: 300x source -> DOWNSAMPLE=0.1 (~30x)"
fi
SUB=""
[ -n "${DOWNSAMPLE:-}" ] && SUB="| samtools view -u -s $DOWNSAMPLE -"

echo "Streaming $CHR reads for $SAMPLE from:"
echo "  $ALN"
# biocontainers samtools has HTTPS; vg's often does not
dock "$SAMTOOLS_IMAGE" bash -c \
  "samtools view -h -u $TARG '$ALN' '$CHR' $SUB | samtools collate -Ou - | samtools fastq -1 '$R1W' -2 '$R2W' -0 /dev/null -s /dev/null -n -"

echo "Done:"
ls -lh "$R1" "$R2"
