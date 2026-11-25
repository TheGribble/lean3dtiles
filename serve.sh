#!/bin/bash
# Serve the tileset locally at http://localhost:8080/  (viewer: index.html)
cd "$(dirname "$0")/output/terrain_tiles" && python3 -m http.server 8080
