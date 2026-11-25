#!/bin/bash
set -euo pipefail
# RAW_DIR: where uncompressed intermediate tiles go. Point it at a tmpfs
# (docker run --tmpfs /scratch:size=12g -e RAW_DIR=/scratch) so ~6 GB of
# uncompressed GLBs never touch disk. Defaults to writing straight into output/.
OUT=output/terrain_tiles
RAW="${RAW_DIR:-$OUT}"
python generate_tiles.py \
  --dem data/ETOPO1_Bed_g_geotiff.tif \
  --land data/ne_10m_land.shp \
  --max-level "${MAX_LEVEL:-5}" \
  --cols "${COLS:-704}" \
  --out "$RAW"
if [ "$RAW" != "$OUT" ]; then
  mkdir -p "$OUT"
  rm -rf "$OUT/tiles"
  cp "$RAW/tileset.json" "$OUT/tileset.json"
fi
# Draco-compress tiles (set COMPRESS=0 to skip)
if [ "${COMPRESS:-1}" = "1" ]; then
  node compress.mjs "$RAW/tiles" "$OUT/tiles"
elif [ "$RAW" != "$OUT" ]; then
  cp -r "$RAW/tiles" "$OUT/tiles"
fi
