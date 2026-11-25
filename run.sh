#!/bin/bash
set -euo pipefail
python generate_tiles.py \
  --dem data/ETOPO1_Bed_g_geotiff.tif \
  --land data/ne_10m_land.shp \
  --max-level "${MAX_LEVEL:-7}" \
  --out output/terrain_tiles
# Draco-compress tiles in place (set COMPRESS=0 to skip)
if [ "${COMPRESS:-1}" = "1" ]; then
  node compress.mjs output/terrain_tiles/tiles
fi
