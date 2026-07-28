#!/usr/bin/env bash
# Build laptop-friendly $CHR Giraffe indexes.
#
# Default (recommended): download prebuilt ${CHR}.d9.vg (~1 GB) from HPRC S3,
# then vg autoindex. Avoids loading the full ~4 GB GBZ just to chunk one chrom.
#
# Fallback: USE_FULL_GBZ=1 chunks from data/downloaded_data/.../*.gbz
#
# Produces (shared): data/fig6/indexes/${CHR}.giraffe.gbz / .min / .dist
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CHR_VG_URL="${CHR_VG_URL:-https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/freeze1/minigraph-cactus/hprc-v1.1-mc-grch38/hprc-v1.1-mc-grch38.chroms/${CHR}.d9.vg}"
CHR_VG="$IDX_DIR/${CHR}.d9.vg"
CHR_GFA="$IDX_DIR/${CHR}.gfa"

have_min() {
  [ -f "$CHR_PREFIX.min" ] || [ -f "$CHR_PREFIX.shortread.withzip.min" ] || [ -f "$CHR_PREFIX.shortread.min" ]
}

if [ -f "$CHR_PREFIX.giraffe.gbz" ] && have_min && [ -f "$CHR_PREFIX.dist" ]; then
  echo "Giraffe $CHR indexes already present — skipping."
  ls -lh "$CHR_PREFIX".giraffe.gbz "$CHR_PREFIX".dist \
    "$CHR_PREFIX".min "$CHR_PREFIX".shortread.withzip.min "$CHR_PREFIX".shortread.zipcodes 2>/dev/null || true
  exit 0
fi

# Migrate old flat layout if present
if [ -f "$FIG6_DIR/${CHR}.giraffe.gbz" ] && [ ! -f "$CHR_PREFIX.giraffe.gbz" ]; then
  echo "Moving existing $CHR Giraffe indexes into indexes/ ..."
  mv "$FIG6_DIR/${CHR}".giraffe.gbz "$FIG6_DIR/${CHR}".dist "$IDX_DIR/" 2>/dev/null || true
  mv "$FIG6_DIR/${CHR}".min "$FIG6_DIR/${CHR}".shortread.withzip.min \
     "$FIG6_DIR/${CHR}".shortread.zipcodes "$IDX_DIR/" 2>/dev/null || true
  if [ -f "$CHR_PREFIX.giraffe.gbz" ] && have_min && [ -f "$CHR_PREFIX.dist" ]; then
    ls -lh "$IDX_DIR"
    exit 0
  fi
fi

if [ "${USE_FULL_GBZ:-0}" = "1" ]; then
  require "$FULL_GBZ"
  GBZ_IN="$(inwork "$FULL_GBZ")"
  echo "[full-gbz] Detecting GRCh38 $CHR path ..."
  REFPATH="$(dockvg vg paths -L -x "$GBZ_IN" | grep -E "^GRCh38(#0)?#${CHR}\$" | head -1 || true)"
  [ -z "$REFPATH" ] && REFPATH="$(dockvg vg paths -L -x "$GBZ_IN" | grep -iE "${CHR}\$" | grep -i grch38 | head -1 || true)"
  REFPATH="${REFPATH_OVERRIDE:-$REFPATH}"
  if [ -z "$REFPATH" ]; then
    echo "ERROR: could not auto-detect a GRCh38 $CHR path. Candidates:" >&2
    dockvg vg paths -L -x "$GBZ_IN" | grep -i "$CHR" | head -20 >&2 || true
    echo "Re-run with REFPATH_OVERRIDE=<name>, or omit USE_FULL_GBZ to use the prebuilt chrom VG." >&2
    exit 1
  fi
  echo "      -> $REFPATH"
  echo "[full-gbz] Extracting $CHR component -> GFA ..."
  dockvg bash -c "vg chunk -x '$GBZ_IN' -C -p '$REFPATH' | vg convert - -f > '$(inwork "$CHR_GFA")'"
else
  if [ ! -f "$CHR_VG" ]; then
    echo "Downloading prebuilt $CHR graph (~1 GB) ..."
    echo "  $CHR_VG_URL"
    curl -L --fail -C - -o "$CHR_VG" "$CHR_VG_URL" || curl -L --fail -o "$CHR_VG" "$CHR_VG_URL"
  else
    echo "Using existing $CHR_VG"
  fi
  echo "Converting ${CHR}.d9.vg -> GFA ..."
  dockvg vg convert "$(inwork "$CHR_VG")" -f > "$CHR_GFA"
fi

echo "Building $CHR Giraffe indexes ..."
dockvg vg autoindex --workflow giraffe -g "$(inwork "$CHR_GFA")" -p "$(inwork "$CHR_PREFIX")" -t "$THREADS"

# Normalize older *.min name if only the new autoindex names exist
if [ ! -f "$CHR_PREFIX.min" ] && [ -f "$CHR_PREFIX.shortread.withzip.min" ]; then
  ln -sfn "$(basename "$CHR_PREFIX.shortread.withzip.min")" "$CHR_PREFIX.min"
fi

echo "Done:"
ls -lh "$CHR_PREFIX".giraffe.gbz "$CHR_PREFIX".dist \
  "$CHR_PREFIX".min "$CHR_PREFIX".shortread.withzip.min "$CHR_PREFIX".shortread.zipcodes 2>/dev/null \
  || ls -lh "$IDX_DIR"
