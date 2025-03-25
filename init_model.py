#!/usr/bin/env python3
"""
Script to preload and verify the YOLO model
"""
import os
import sys
import time
import traceback

def initialize_model():
    """Initialize the YOLO model and verify it works properly"""
    try:
        start_time = time.time()
        print("Initializing model verification...")
        
        # Try to import required libraries
        import torch
        from ultralytics import YOLO
        import cv2
        import numpy as np
        
        print(f"Libraries imported successfully")
        
        # Find model path
        model_path = os.path.join('drivesafe', 'models', 'traffic_light', 'best_traffic_small_yolo.pt')
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at {model_path}")
            return False
        
        # Initialize device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Load model
        print(f"Loading YOLO model from: {model_path}")
        model = YOLO(model_path)
        print("YOLO model loaded successfully")
        
        # Create test image (black image with white rectangle)
        print("Creating test image...")
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        
        # Run test inference
        print("Running test inference...")
        results = model(test_image)
        print("Test inference completed successfully")
        
        # Report success
        elapsed_time = time.time() - start_time
        print(f"Model verification completed successfully in {elapsed_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"Error during model verification: {str(e)}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = initialize_model()
    if success:
        print("YOLO model initialization successful!")
        sys.exit(0)
    else:
        print("YOLO model initialization failed!")
        sys.exit(1) 