from flask import Flask, render_template, Response, request, jsonify
import os
import sys
import logging
import traceback
import numpy as np
import threading
import cv2
import requests
from tqdm import tqdm
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Constants
HUGGINGFACE_MODEL_URL = "https://huggingface.co/ojaskandy/traffic-light-detection-yolo/resolve/main/best_traffic_small_yolo.pt"
MODEL_CACHE_DIR = os.path.expanduser("~/.cache/drivesafe/models")
MODEL_FILENAME = "best_traffic_small_yolo.pt"

# Global variables
yolo_model = None
model_lock = threading.Lock()

def download_model():
    """Download the model from Hugging Face if not already cached."""
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_CACHE_DIR, MODEL_FILENAME)
        
        if not os.path.exists(model_path):
            logger.info("Downloading model from Hugging Face...")
            response = requests.get(HUGGINGFACE_MODEL_URL, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 8192
            
            with open(model_path, 'wb') as f:
                with tqdm(total=total_size, unit='iB', unit_scale=True) as pbar:
                    for data in response.iter_content(block_size):
                        f.write(data)
                        pbar.update(len(data))
            
            logger.info(f"Model downloaded successfully to {model_path}")
        else:
            logger.info("Using cached model file")
        
        return model_path
    except Exception as e:
        logger.error(f"Error downloading model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def init_yolo_model():
    """Initialize the YOLO model."""
    global yolo_model
    try:
        model_path = download_model()
        if model_path:
            from ultralytics import YOLO
            yolo_model = YOLO(model_path)
            logger.info("YOLO model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}")
        logger.error(traceback.format_exc())
        yolo_model = None

def detect_lanes(frame):
    """Basic lane detection using OpenCV"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Define region of interest
        height, width = edges.shape
        mask = np.zeros_like(edges)
        polygon = np.array([[
            (0, height),
            (width, height),
            (width * 0.6, height * 0.6),
            (width * 0.4, height * 0.6)
        ]], np.int32)
        cv2.fillPoly(mask, polygon, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=40,
            maxLineGap=20
        )
        
        # Draw lines on the original frame
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            return frame, len(lines)
        return frame, 0
    except Exception as e:
        logger.error(f"Error in lane detection: {str(e)}")
        return frame, 0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_drive')
def start_drive():
    return render_template('start_drive.html')

@app.route('/process_frame', methods=['POST'])
def process_frame_endpoint():
    """Process a frame from the camera feed."""
    try:
        # Get the image data from the request
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'})
        
        # Convert base64 image to OpenCV format
        import base64
        img_data = data['image'].split(',')[1]
        img_bytes = base64.b64decode(img_data)
        img_np = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({'error': 'Invalid image data'})
        
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Process frame with lane detection
        processed_frame, num_lanes = detect_lanes(frame)
        
        # If YOLO model is available, try traffic light detection
        detection_status = []
        if yolo_model is not None:
            try:
                with model_lock:
                    results = yolo_model(processed_frame)
                processed_frame = results[0].plot()
                detection_status.append("Traffic light detection active")
            except Exception as e:
                logger.error(f"Error in YOLO detection: {str(e)}")
        
        # Add lane detection status
        if num_lanes > 0:
            detection_status.append(f"Lane detection active ({num_lanes} lanes)")
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        status_message = " | ".join(detection_status) if detection_status else "Processing active"
        
        return jsonify({
            'processed_image': f'data:image/jpeg;base64,{img_base64}',
            'detection_status': status_message
        })
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})

@app.route('/model_status')
def get_model_status():
    """Get the status of model loading."""
    return jsonify({
        'opencv_loaded': True,
        'yolo_loaded': yolo_model is not None,
        'opencv_error': None,
        'yolo_error': None if yolo_model is not None else "Model not loaded"
    })

if __name__ == '__main__':
    # Initialize YOLO model in a background thread
    threading.Thread(target=init_yolo_model, daemon=True).start()
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
