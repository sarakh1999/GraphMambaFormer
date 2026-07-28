#!/usr/bin/env bash
# Stream SAMPLE short reads for $CHR from a remote BAM/CRAM -> paired FASTQ.
#
# Resolution order:
#   1) READS_ALN_URL env var
#   2) GIAB defaults for HG005 / HG002
#   3) HPRC Illumina CRAM from data/hprc/sample_links.json (e.g. HG00438)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RESOLVE="$(dirname "${BASH_SOURCE[0]}")/resolve_sample.py"

ALN="${READS_ALN_URL:-}"
if [ -z "$ALN" ]; then
  ALN="$(python3 "$RESOLVE" "$SAMPLE" reads_aln_url 2>/dev/null || true)"
fi
if [ -z "$ALN" ]; then
  echo "ERROR: no short-read alignment URL for SAMPLE=$SAMPLE" >&2
  echo "       set READS_ALN_URL or add the sample to data/hprc/sample_links.json" >&2
  exit 1
fi

R1="$READS_DIR/${SAMPLE}.${CHR}.R1.fastq.gz"
R2="$READS_DIR/${SAMPLE}.${CHR}.R2.fastq.gz"
R1W="$(inwork "$R1")"; R2W="$(inwork "$R2")"

TARG=""
case "$ALN" in
  *.cram) require "$REF_FULL_GZ" "run fetch_reference.sh first"
          TARG="-T $(inwork "$REF_FULL_GZ")";;
esac

if [ -z "${DOWNSAMPLE:-}" ] && [[ "$ALN" == *300x* ]]; then
  DOWNSAMPLE=0.1
  echo "note: 300x source -> DOWNSAMPLE=0.1 (~30x)"
fi
SUB=""
[ -n "${DOWNSAMPLE:-}" ] && SUB="| samtools view -u -s $DOWNSAMPLE -"

echo "Streaming $CHR short reads from:"
echo "  $ALN"

if [[ "$ALN" == s3://* ]]; then
  require_cmd aws "install AWS CLI (brew install awscli) for HPRC S3 CRAM/BAM"
  dockvg bash -c "export AWS_DEFAULT_REGION=us-west-2; aws s3 cp --no-sign-request '$ALN' - | samtools view -h -u $TARG - '$CHR' $SUB | samtools collate -Ou - | samtools fastq -1 '$R1W' -2 '$R2W' -0 /dev/null -s /dev/null -n -"
else
  dockvg bash -c "samtools view -h -u $TARG '$ALN' '$CHR' $SUB | samtools collate -Ou - | samtools fastq -1 '$R1W' -2 '$R2W' -0 /dev/null -s /dev/null -n -"
fi

echo "Done:"; ls -lh "$R1" "$R2"
