#!/usr/bin/env bash
# Build chr21 Giraffe indexes for the HPRC v1.1 GRCh38 pangenome.
#
# Default path (recommended): download the prebuilt chr21.d9.vg (~1 GB) and run
# vg autoindex locally. This avoids downloading the full ~100 GB GBZ.
#
# Optional fallback: if data/downloaded_data/.../hprc-v1.1-mc-grch38.d9.gbz exists,
# chunk chr21 from the full graph instead (USE_FULL_GBZ=1).
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"

CHR_VG_URL="${CHR_VG_URL:-https://s3-us-west-2.amazonaws.com/human-pangenomics/pangenomes/freeze/freeze1/minigraph-cactus/hprc-v1.1-mc-grch38/hprc-v1.1-mc-grch38.chroms/${CHR}.d9.vg}"
CHR_VG="$RUN_DIR/${CHR}.d9.vg"
CHR_GFA="$RUN_DIR/${CHR}.gfa"

# Newer vg names the minimizer "<prefix>.shortread.withzip.min"; older vg used
# "<prefix>.min". Treat either as "already built".
MIN_GLOB=("$CHR_PREFIX".shortread.withzip.min "$CHR_PREFIX".min)
EXISTING_MIN=""
for m in "${MIN_GLOB[@]}"; do [ -f "$m" ] && { EXISTING_MIN="$m"; break; }; done
if [ -f "$CHR_PREFIX.giraffe.gbz" ] && [ -n "$EXISTING_MIN" ] && [ -f "$CHR_PREFIX.dist" ]; then
  echo "Found existing chr21 Giraffe indexes:"
  ls -lh "$CHR_PREFIX".giraffe.gbz "$EXISTING_MIN" "$CHR_PREFIX".dist "$CHR_PREFIX".shortread.zipcodes 2>/dev/null
  exit 0
fi

if [ "${USE_FULL_GBZ:-0}" = "1" ] && [ -f "$FULL_GBZ" ]; then
  GBZ_IN="$(inwork "$FULL_GBZ")"
  echo "[full-gbz] Detecting GRCh38 $CHR path ..."
  REFPATH="$(dockvg vg paths -L -x "$GBZ_IN" | grep -E "^GRCh38(#0)?#${CHR}\$" | head -1 || true)"
  [ -z "$REFPATH" ] && REFPATH="$(dockvg vg paths -L -x "$GBZ_IN" | grep -iE "${CHR}\$" | grep -i grch38 | head -1 || true)"
  REFPATH="${REFPATH_OVERRIDE:-$REFPATH}"
  [ -z "$REFPATH" ] && { echo "ERROR: could not auto-detect GRCh38 $CHR path." >&2; exit 1; }
  echo "[full-gbz] Extracting $CHR component -> GFA ..."
  dockvg bash -c "vg chunk -x '$GBZ_IN' -C -p '$REFPATH' | vg convert - -f > '$(inwork "$CHR_GFA")'"
else
  # Reuse an already-present graph (the NN prep left chr21.d9.vg ~1 GB). Only
  # download when it's missing or clearly truncated — never resume, which can
  # append an S3 error page onto a complete file and corrupt it.
  if [ -s "$CHR_VG" ] && [ "$(stat -c%s "$CHR_VG" 2>/dev/null || echo 0)" -gt 100000000 ]; then
    echo "Using existing $CHR_VG ($(du -h "$CHR_VG" | cut -f1)) — skipping download."
  else
    echo "Downloading prebuilt $CHR graph (~1 GB) ..."
    echo "  $CHR_VG_URL"
    curl -L --fail -o "$CHR_VG" "$CHR_VG_URL"
  fi
  # Reuse an existing non-trivial GFA rather than regenerating (the redirect '>'
  # would otherwise truncate a good chr21.gfa before vg even runs).
  if [ -s "$CHR_GFA" ] && [ "$(stat -c%s "$CHR_GFA" 2>/dev/null || echo 0)" -gt 1000000 ]; then
    echo "Using existing $CHR_GFA ($(du -h "$CHR_GFA" | cut -f1)) — skipping conversion."
  else
    echo "Converting $CHR.d9.vg -> GFA ..."
    dockvg vg convert "$(inwork "$CHR_VG")" -f > "$CHR_GFA"
  fi
fi

echo "Building $CHR Giraffe indexes ..."
dockvg vg autoindex --workflow giraffe -g "$(inwork "$CHR_GFA")" -p "$(inwork "$CHR_PREFIX")" -t "$THREADS"

echo "Done:"
ls -lh "$CHR_PREFIX".giraffe.gbz "$CHR_PREFIX".dist \
       "$CHR_PREFIX".shortread.withzip.min "$CHR_PREFIX".shortread.zipcodes \
       "$CHR_PREFIX".min 2>/dev/null
