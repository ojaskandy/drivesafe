import cv2
import numpy as np
from ultralytics import YOLO
import os
import torch
import torch.nn as nn
from flask import Flask, render_template, Response, request, redirect, url_for
import threading
import queue

app = Flask(__name__)

class DriveSafeDetector:
    def __init__(self):
        self.frameWidth = 640
        self.frameHeight = 480
        self.setup_models()
        self.is_running = False
        self.video = None
        
    def setup_models(self):
        traffic_model_path = os.path.join("models", "traffic_light", "best_traffic_small_yolo.pt")
        if not os.path.exists(traffic_model_path):
            raise FileNotFoundError(f"Traffic light model not found at {traffic_model_path}")
        print("Loading traffic light model...")
        try:
            self.traffic_light_model = YOLO(traffic_model_path)
            print("Traffic light model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
