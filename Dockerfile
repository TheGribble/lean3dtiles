FROM python:3.11-slim

# libexpat1: needed by rasterio's bundled GDAL. nodejs/npm: Draco compression stage.
RUN apt-get update && apt-get install -y --no-install-recommends libexpat1 nodejs npm \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY package.json .
RUN npm install --omit=dev --no-audit --no-fund

COPY generate_tiles.py compress.mjs run.sh ./
CMD ["bash", "run.sh"]
