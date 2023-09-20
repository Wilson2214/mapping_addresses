#!/usr/bin/env sh

# Install converter
npm install -g geojson2ndjson

cd data

# Convert File
ndjson2geojson ms_hinds_parcels.ndgeojson > ms_hinds_parcels.geojson