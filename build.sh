#!/usr/bin/env bash
set -o errexit

# Ensure proper installation order with specific versions first
pip install opencv-python-headless==4.5.3.56
pip install -r requirements.txt
pip install ultralytics

# Create necessary directories
mkdir -p drivesafe/static/uploads drivesafe/static/detections drivesafe/models/traffic_light drivesafe/templates

# Copy model file if needed
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
