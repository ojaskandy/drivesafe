#!/usr/bin/env bash
set -o errexit
pip install -r requirements.txt
mkdir -p drivesafe/static/uploads drivesafe/static/detections drivesafe/models/traffic_light
if [ ! -f "drivesafe/models/traffic_light/best_traffic_small_yolo.pt" ]; then
    cp best_traffic_small_yolo.pt drivesafe/models/traffic_light/
fi
if [ -d "templates" ]; then
    cp -r templates drivesafe/
fi
if [ -d "static" ]; then
    cp -r static/* drivesafe/static/
fi
