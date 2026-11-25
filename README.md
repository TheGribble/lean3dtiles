# lean3dtiles

Builds a global, land + sea terrain **3D Tiles 1.1** tileset (Draco-compressed GLB tiles,
green elevation-graded land, flat blue sea) for CesiumJS / Cesium for Unity.

The built tileset is in `output/terrain_tiles/` (~134 MB, 12,317 tiles, ~5 km finest cells).

## Rebuild

1. Download ETOPO1 Bedrock (grid-registered GeoTIFF) into `data/`:
   https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO1/data/bedrock/grid_registered/georeferenced_tiff/ETOPO1_Bed_g_geotiff.zip
   (unzip so `data/ETOPO1_Bed_g_geotiff.tif` exists). Natural Earth land shapefiles are already in `data/`.
2. ```
   docker build -t globe-builder .
   docker run --rm -u $(id -u):$(id -g) -v $PWD/data:/workspace/data:ro -v $PWD/output:/workspace/output globe-builder
   ```
   Options (env): `MAX_LEVEL` (default 7), `COMPRESS=0` to skip Draco.

## View

`./serve.sh`, then open http://localhost:8080/ (CesiumJS from CDN).

## Notes

- Draco-compressed normalised-uint8 `COLOR_0` renders red/garbled in CesiumJS, so `compress.mjs`
  converts colours to float first.
- Terrain heights come from ETOPO1 (~1.8 km); finer tiles need a finer DEM.
