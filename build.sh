#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
cd drivesafe
mkdir -p static/uploads
mkdir -p static/detections

# Make sure the model file is in the correct location
if [ ! -f "best_traffic_small_yolo.pt" ]; then
    echo "Copying model file to the correct location..."
    cp ../best_traffic_small_yolo.pt .
fi 