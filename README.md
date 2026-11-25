# lean3dtiles

Builds a global, land + sea terrain **3D Tiles 1.1** tileset (Draco-compressed GLB tiles,
green elevation-graded land, flat blue sea) for CesiumJS / Cesium for Unity.

The built tileset is in `output/terrain_tiles/` (~464 MB, 1,189 tiles, 6 levels, ~1.8 km finest cells,
matching ETOPO1's native resolution). Each tile is a 704x352-cell grid, so few levels are needed.

## Rebuild

1. Download ETOPO1 Bedrock (grid-registered GeoTIFF) into `data/`:
   https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/georeferenced_tiff/ETOPO1_Bed_g_geotiff.zip
   (unzip so `data/ETOPO1_Bed_g_geotiff.tif` exists). Natural Earth land shapefiles are already in `data/`.
2. ```
   docker build -t globe-builder .
   docker run --rm --tmpfs /scratch:size=14g -e RAW_DIR=/scratch -u $(id -u):$(id -g) \
     -v $PWD/data:/workspace/data:ro -v $PWD/output:/workspace/output globe-builder
   ```
   The `--tmpfs /scratch` keeps ~12.6 GB of uncompressed intermediate GLBs in RAM (needs ~16 GB free);
   without `RAW_DIR` they are written to `output/` first (needs ~13 GB of disk) and then compressed in place.
   Options (env): `MAX_LEVEL` (default 5), `COLS` (cells per tile, default 704; finest cell = 360/(COLS*2^MAX_LEVEL) deg),
   `COMPRESS=0` to skip Draco. Build takes ~25 min.

## View

`./serve.sh`, then open http://localhost:8080/ (CesiumJS from CDN).

## Notes

- Coastlines are cut *exactly* against the Natural Earth 10m land polygon: cells the coast crosses are clipped and
  re-triangulated, so waterways narrower than a grid cell (e.g. the Solent, Southampton Water) stay open.
  Heights are clamped to >= 0; the sea is a flat mesh 4 m below that (`--sea-offset`).
  `--bbox w s e n` builds only that area in detail (for quick tests).
- Draco-compressed normalised-uint8 `COLOR_0` renders red/garbled in CesiumJS, so `compress.mjs`
  converts colours to float first.
- Terrain heights come from ETOPO1 (~1.8 km); finer tiles need a finer DEM.
- Pure-ocean tiles use a coarse 64-cell grid, since a flat sea needs no detail.
