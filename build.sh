#!/usr/bin/env bash
set -o errexit
pip install -r requirements.txt
pip install ultralytics
mkdir -p drivesafe/static/uploads drivesafe/static/detections drivesafe/models/traffic_light drivesafe/templates
if [ ! -f "drivesafe/models/traffic_light/best_traffic_small_yolo.pt" ]; then
    cp best_traffic_small_yolo.pt drivesafe/models/traffic_light/
fi
echo "Copying templates..."
cp -r templates/* drivesafe/templates/
echo "Copying static files..."
if [ -d "static" ]; then
    cp -r static/* drivesafe/static/
fi
echo "Final directory structure:"
ls -R drivesafe/
