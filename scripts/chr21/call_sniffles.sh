#!/usr/bin/env bash
# Map PacBio HiFi / ONT long reads for $CHR with minimap2 -> sorted BAM,
# then call structural variants with Sniffles2.
#
# https://github.com/fritzsedlazeck/Sniffles
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
RESOLVE="$(dirname "${BASH_SOURCE[0]}")/resolve_sample.py"

require "$REF_FA" "run fetch_reference.sh"

HIFI_BAM_LOCAL="$BAM_DIR/${SAMPLE}.${CHR}.hifi.minimap.sorted.bam"
SV_VCF="$SV_DIR/${SAMPLE}.${CHR}.sniffles.vcf.gz"

if [ -z "${HIFI_BAM_URL:-}" ] && [ -z "${HIFI_BAM:-}" ] && [ -z "${HIFI_FASTQ:-}" ]; then
  HIFI_BAM_URL="$(python3 "$RESOLVE" "$SAMPLE" hifi_bam_url 2>/dev/null || true)"
fi

prepare_bam_from_fastq() {
  local fq="$1"
  require "$fq"
  echo "minimap2 map-hifi -> $HIFI_BAM_LOCAL ..."
  dock "$MINIMAP_IMAGE" minimap2 -ayYL --MD -x map-hifi -t "$THREADS" \
    "$(inwork "$REF_FA")" "$(inwork "$fq")" \
    > "$RUN_DIR/${SAMPLE}.${CHR}.hifi.sam"
  dockvg bash -c "samtools sort -@ $THREADS -o '$(inwork "$HIFI_BAM_LOCAL")' '$(inwork "$RUN_DIR/${SAMPLE}.${CHR}.hifi.sam")' && samtools index '$(inwork "$HIFI_BAM_LOCAL")'"
  rm -f "$RUN_DIR/${SAMPLE}.${CHR}.hifi.sam"
}

prepare_bam_from_url() {
  local url="$1"
  echo "Streaming $CHR long reads from: $url"
  local TARG=""
  case "$url" in
    *.cram) require "$REF_FULL_GZ"; TARG="-T $(inwork "$REF_FULL_GZ")";;
  esac
  if [[ "$url" == s3://* ]]; then
    require_cmd aws "install AWS CLI (brew install awscli) for HPRC S3 HiFi BAM"
    dockvg bash -c "export AWS_DEFAULT_REGION=us-west-2; aws s3 cp --no-sign-request '$url' - | samtools view -h -u $TARG - '$CHR' | samtools sort -@ $THREADS -o '$(inwork "$HIFI_BAM_LOCAL")' - && samtools index '$(inwork "$HIFI_BAM_LOCAL")'"
  else
    dockvg bash -c "samtools view -h -u $TARG '$url' '$CHR' | samtools sort -@ $THREADS -o '$(inwork "$HIFI_BAM_LOCAL")' - && samtools index '$(inwork "$HIFI_BAM_LOCAL")'"
  fi
}

if [ -n "${HIFI_BAM:-}" ]; then
  require "$HIFI_BAM"
  cp -f "$HIFI_BAM" "$HIFI_BAM_LOCAL"
  [ -f "${HIFI_BAM}.bai" ] && cp -f "${HIFI_BAM}.bai" "${HIFI_BAM_LOCAL}.bai"
  [ -f "${HIFI_BAM_LOCAL}.bai" ] || dockvg samtools index "$(inwork "$HIFI_BAM_LOCAL")"
elif [ -n "${HIFI_BAM_URL:-}" ]; then
  prepare_bam_from_url "$HIFI_BAM_URL"
elif [ -n "${HIFI_FASTQ:-}" ]; then
  prepare_bam_from_fastq "$HIFI_FASTQ"
elif [ -f "$HIFI_BAM_LOCAL" ]; then
  echo "Using existing $HIFI_BAM_LOCAL"
else
  cat >&2 <<EOF
ERROR: no long-read input for Sniffles.

Provide one of:
  HIFI_FASTQ=/path/to/reads.fastq.gz
  HIFI_BAM=/path/to/sorted.bam
  HIFI_BAM_URL=s3://.../sample.bam

Or ensure the sample is listed in data/hprc/sample_links.json.

List HiFi files for a Year-1 sample:
  ./scripts/chr21/fetch_hprc_sample.sh HG00438
  ./scripts/chr21/fetch_hprc_sample.sh HG00438 hifi
EOF
  exit 1
fi

echo "Sniffles2 SV calling ..."
dock "$SNIFFLES_IMAGE" sniffles \
  --input "$(inwork "$HIFI_BAM_LOCAL")" \
  --vcf "$(inwork "${SV_VCF%.gz}")" \
  --reference "$(inwork "$REF_FA")" \
  --threads "$THREADS" \
  --allow-overwrite

if [ -f "${SV_VCF%.gz}" ] && [ ! -f "$SV_VCF" ]; then
  dock "$BCFTOOLS_IMAGE" bgzip -f "$(inwork "${SV_VCF%.gz}")"
fi
if [ -f "$SV_VCF" ]; then
  dock "$BCFTOOLS_IMAGE" bcftools index -t "$(inwork "$SV_VCF")" || true
fi

echo "Done: $SV_VCF (or ${SV_VCF%.gz})"
ls -lh "$SV_DIR"
