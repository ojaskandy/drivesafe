from flask import Flask, render_template, Response, request, jsonify, url_for, send_from_directory, stream_with_context
import os
import sys
import time
import logging
from pathlib import Path
import base64
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
yolo_model = None
detected_classes = {
    'red': 0,
    'green': 0,
    'yellow': 0,
    'none': 0
}

def lazy_import():
    """Lazy import of heavy dependencies"""
    try:
        import cv2
        import torch
        from ultralytics import YOLO
        return cv2, torch, YOLO
    except ImportError as e:
        logger.error(f"Error importing dependencies: {str(e)}")
        return None, None, None

def init_yolo_model():
    """Initialize the YOLO model efficiently"""
    global yolo_model
    
    if yolo_model is not None:
        return yolo_model
    
    try:
        # Lazy import dependencies
        cv2, torch, YOLO = lazy_import()
        if None in (cv2, torch, YOLO):
            logger.error("Failed to import required dependencies")
            return None
        
        # Initialize device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # Load model directly
        model_path = os.path.join('drivesafe', 'models', 'traffic_light', 'best_traffic_small_yolo.pt')
        if not os.path.exists(model_path):
            logger.error(f"Model file not found at {model_path}")
            return None
            
        yolo_model = YOLO(model_path)
        logger.info("YOLO model loaded successfully")
        return yolo_model
        
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model: {str(e)}")
        return None

def process_frame(frame, save_path=None):
    """Process a frame with the YOLO model and return the results"""
    global yolo_model, detected_classes
    
    try:
        # Import cv2 here to avoid circular import
        cv2, _, _ = lazy_import()
        if cv2 is None:
            return frame, "OpenCV import failed", None
            
        # If model is not loaded, try loading it
        if yolo_model is None:
            yolo_model = init_yolo_model()
            if yolo_model is None:
                return frame, "Model loading failed", None
        
        # Process the frame with the YOLO model
        results = yolo_model(frame)
        
        # Get the plotted frame with bounding boxes
        annotated_frame = results[0].plot()
        
        # Extract class information from results
        boxes = results[0].boxes
        current_detected = {"red": 0, "green": 0, "yellow": 0, "none": 0}
        
        if len(boxes) > 0:
            for box in boxes:
                # Get class index and convert to class name
                cls = int(box.cls.item())
                class_name = results[0].names[cls]
                
                # Update detection counters
                if class_name in current_detected:
                    current_detected[class_name] += 1
        
        # Update global detection counters
        for key in detected_classes:
            if current_detected[key] > 0:
                detected_classes[key] += current_detected[key]
        
        # Determine overall detection status
        if current_detected["red"] > 0:
            detection_status = "Red light detected! Please stop."
        elif current_detected["yellow"] > 0:
            detection_status = "Yellow light detected! Proceed with caution."
        elif current_detected["green"] > 0:
            detection_status = "Green light detected! Safe to proceed."
        else:
            detection_status = "No traffic lights detected"
        
        return annotated_frame, detection_status, current_detected
    
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return frame, f"Processing error: {str(e)}", None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start_drive')
def start_drive():
    return render_template('start_drive.html')

@app.route('/process_frame', methods=['POST'])
def process_image():
    global detected_classes, yolo_model
    
    try:
        # Lazy import cv2
        cv2, _, _ = lazy_import()
        if cv2 is None:
            return jsonify({'error': 'Failed to import OpenCV'}), 500
            
        # Get the image data from the request
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data received'}), 400
        
        # Decode the base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Process the frame
        processed_frame, status, current_detections = process_frame(frame)
        
        # Encode the processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        # Return the processed image and detection information
        return jsonify({
            'processed_image': f'data:image/jpeg;base64,{processed_image}',
            'detection_status': status,
            'model_loaded': yolo_model is not None,
            'detections': detected_classes,
            'current_detections': current_detections
        })
        
    except Exception as e:
        logger.error(f"Error in process_frame endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'model_loaded': yolo_model is not None
        }), 500

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
def model_status():
    global yolo_model
    
    return jsonify({
        'model_loaded': yolo_model is not None
    })

if __name__ == '__main__':
    # Initialize model at startup in development
    if os.environ.get('FLASK_ENV') != 'production':
        yolo_model = init_yolo_model()
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 