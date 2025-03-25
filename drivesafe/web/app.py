from flask import Flask, render_template, Response, request, jsonify, url_for, send_from_directory, stream_with_context
import os
import sys
import time
import logging
from pathlib import Path
import base64
import numpy as np
import threading
from queue import Queue
import traceback
import cv2
import requests
from tqdm import tqdm

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
yolo_model = None
cv2 = None
torch = None
YOLO = None
model_lock = threading.Lock()
frame_queue = Queue(maxsize=2)
detected_classes = {
    'red': 0,
    'green': 0,
    'yellow': 0,
    'none': 0
}

# Model status tracking
model_status = {
    'yolo_loaded': False,
    'yolo_error': None,
    'opencv_loaded': False,
    'opencv_error': None,
    'last_error': None
}

# Constants
HUGGINGFACE_MODEL_URL = "https://huggingface.co/ojaskandy/traffic-light-detection-yolo/resolve/main/best_traffic_small_yolo.pt"
MODEL_CACHE_DIR = os.path.expanduser("~/.cache/drivesafe/models")
MODEL_FILENAME = "best_traffic_small_yolo.pt"

def initialize_dependencies():
    """Initialize all required dependencies with detailed error tracking"""
    global cv2, torch, YOLO, model_status
    
    try:
        if cv2 is None:
            try:
                import cv2 as cv2_import
                cv2 = cv2_import
                model_status['opencv_loaded'] = True
                model_status['opencv_error'] = None
                logger.info("OpenCV initialized successfully")
            except Exception as e:
                error_msg = f"Failed to import OpenCV: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                model_status['opencv_error'] = error_msg
                model_status['opencv_loaded'] = False
        
        if torch is None:
            try:
                import torch as torch_import
                torch = torch_import
                logger.info("PyTorch initialized successfully")
            except Exception as e:
                error_msg = f"Failed to import PyTorch: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                model_status['yolo_error'] = error_msg
                return False
        
        if YOLO is None:
            try:
                from ultralytics import YOLO as YOLO_import
                YOLO = YOLO_import
                logger.info("YOLO imported successfully")
            except Exception as e:
                error_msg = f"Failed to import YOLO: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                model_status['yolo_error'] = error_msg
                return False
        
        return True
    except Exception as e:
        error_msg = f"Failed to initialize dependencies: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        model_status['last_error'] = error_msg
        return False

def detect_lanes(frame):
    """Basic lane detection using OpenCV"""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Define region of interest (bottom half of the frame)
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
        result = frame.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return result, len(lines) if lines is not None else 0
    except Exception as e:
        logger.error(f"Error in lane detection: {str(e)}\n{traceback.format_exc()}")
        return frame, 0

def download_model():
    """Download the model from Hugging Face if not already cached."""
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

def init_yolo_model():
    """Initialize the YOLO model."""
    try:
        model_path = download_model()
        import torch
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)
        model.conf = 0.5  # Confidence threshold
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def process_frame(frame, save_path=None):
    """Process a frame with YOLO model and/or lane detection"""
    global yolo_model, detected_classes, cv2, model_status
    
    try:
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        detection_status = []
        current_detected = {"red": 0, "green": 0, "yellow": 0, "none": 0}
        
        # Try YOLO detection if available
        if yolo_model is not None:
            try:
                # Convert frame to RGB (YOLO expects RGB)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Run inference with lower confidence threshold
                with model_lock:
                    results = yolo_model(frame_rgb, conf=0.3, iou=0.5)
                
                # Get the plotted frame with bounding boxes
                annotated_frame = results[0].plot()
                annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
                
                # Process detections
                boxes = results[0].boxes
                if len(boxes) > 0:
                    for box in boxes:
                        try:
                            cls = int(box.cls.item())
                            conf = float(box.conf.item())
                            class_name = results[0].names[cls].lower()
                            
                            if conf > 0.3 and class_name in current_detected:
                                current_detected[class_name] += 1
                                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                                cv2.putText(annotated_frame, f"{conf:.2f}", (x1, y1-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                        except Exception as e:
                            logger.error(f"Error processing detection box: {str(e)}")
                            continue
                
                frame = annotated_frame
                detection_status.append("Traffic light detection active")
                
            except Exception as e:
                error_msg = f"Error in YOLO detection: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                model_status['yolo_error'] = error_msg
        
        # Always try lane detection as fallback
        if model_status['opencv_loaded']:
            try:
                frame, num_lanes = detect_lanes(frame)
                if num_lanes > 0:
                    detection_status.append(f"Lane detection active ({num_lanes} lanes)")
            except Exception as e:
                error_msg = f"Error in lane detection: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                model_status['opencv_error'] = error_msg
        
        # Update global detection counters
        for key in detected_classes:
            if current_detected[key] > 0:
                detected_classes[key] += current_detected[key]
        
        # Determine overall detection status
        if not detection_status:
            status = "No detection methods available"
        else:
            status = " | ".join(detection_status)
            if current_detected["red"] > 0:
                status += f" | Red light detected! ({current_detected['red']} instances)"
            elif current_detected["yellow"] > 0:
                status += f" | Yellow light detected! ({current_detected['yellow']} instances)"
            elif current_detected["green"] > 0:
                status += f" | Green light detected! ({current_detected['green']} instances)"
        
        return frame, status, current_detected
    
    except Exception as e:
        error_msg = f"Error processing frame: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        model_status['last_error'] = error_msg
        return frame, f"Processing error: {str(e)}", None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_drive')
def start_drive():
    return render_template('start_drive.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
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
        
        # Process frame with lane detection
        processed_frame = process_frame_with_lanes(frame)
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_image': f'data:image/jpeg;base64,{img_base64}',
            'detection_status': 'Lane detection successful'
        })
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})

@app.route('/stats', methods=['GET'])
def get_stats():
    global detected_classes, yolo_model
    
    return jsonify({
        'detections': detected_classes,
        'model_loaded': yolo_model is not None
    })

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    global detected_classes
    
    detected_classes = {
        'red': 0,
        'green': 0,
        'yellow': 0,
        'none': 0
    }
    
    return jsonify({'status': 'success'})

@app.route('/model_status', methods=['GET'])
def get_model_status():
    """Get the status of model loading."""
    global yolo_model
    
    status = {
        'opencv_loaded': True,
        'yolo_loaded': yolo_model is not None,
        'opencv_error': None,
        'yolo_error': None if yolo_model is not None else "Model not loaded"
    }
    
    return jsonify(status)

if __name__ == '__main__':
    # Initialize dependencies at startup
    if not initialize_dependencies():
        logger.error("Failed to initialize dependencies. Continuing with limited functionality.")
    
    # Initialize model at startup in a separate thread
    model_thread = threading.Thread(target=init_yolo_model)
    model_thread.start()
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port) 