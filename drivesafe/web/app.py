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
    """Detect lanes using OpenCV"""
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
        
        # Create result image
        result = frame.copy()
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        return result, len(lines) if lines is not None else 0
    except Exception as e:
        logger.error(f"Error in lane detection: {str(e)}\n{traceback.format_exc()}")
        return frame, 0

def init_yolo_model():
    """Initialize the YOLO model with comprehensive error handling"""
    global yolo_model, model_status
    
    with model_lock:
        if yolo_model is not None:
            return yolo_model
        
        try:
            # Initialize dependencies first
            if not initialize_dependencies():
                error_msg = "Failed to initialize dependencies for YOLO model"
                logger.error(error_msg)
                model_status['yolo_error'] = error_msg
                return None
            
            # Initialize device and set PyTorch settings
            try:
                if torch.cuda.is_available():
                    device = torch.device('cuda')
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                else:
                    device = torch.device('cpu')
                    if hasattr(torch, 'set_num_threads'):
                        torch.set_num_threads(4)
                
                logger.info(f"Using device: {device}")
            except Exception as e:
                error_msg = f"Error setting up PyTorch device: {str(e)}\n{traceback.format_exc()}"
                logger.error(error_msg)
                model_status['yolo_error'] = error_msg
                return None
            
            # Look for model in multiple locations
            model_paths = [
                'best_traffic_small_yolo.pt',
                os.path.join('models', 'traffic_light', 'best_traffic_small_yolo.pt'),
                os.path.join(os.path.dirname(__file__), 'best_traffic_small_yolo.pt'),
                os.path.join(os.path.dirname(__file__), 'models', 'traffic_light', 'best_traffic_small_yolo.pt')
            ]
            
            # Log all attempted paths
            logger.info("Attempting to load model from the following paths:")
            for path in model_paths:
                logger.info(f"- {os.path.abspath(path)}")
            
            model_loaded = False
            last_error = None
            
            for model_path in model_paths:
                try:
                    if os.path.exists(model_path):
                        logger.info(f"Found model at: {os.path.abspath(model_path)}")
                        logger.info(f"File size: {os.path.getsize(model_path)} bytes")
                        
                        yolo_model = YOLO(model_path)
                        if device.type == 'cpu':
                            yolo_model.cpu()
                        else:
                            yolo_model.to(device)
                        
                        # Verify model loaded correctly
                        dummy_input = torch.zeros((1, 3, 640, 480), device=device)
                        _ = yolo_model(dummy_input)
                        
                        logger.info("Model loaded and verified successfully")
                        model_loaded = True
                        model_status['yolo_loaded'] = True
                        model_status['yolo_error'] = None
                        break
                    else:
                        logger.warning(f"Model not found at: {os.path.abspath(model_path)}")
                except Exception as e:
                    last_error = f"Error loading model from {model_path}: {str(e)}\n{traceback.format_exc()}"
                    logger.error(last_error)
                    continue
            
            if not model_loaded:
                error_msg = f"Could not load model from any path. Last error: {last_error}"
                logger.error(error_msg)
                model_status['yolo_error'] = error_msg
                return None
            
            return yolo_model
            
        except Exception as e:
            error_msg = f"Failed to initialize YOLO model: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            model_status['yolo_error'] = error_msg
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
def process_image():
    global detected_classes, yolo_model, cv2
    
    try:
        # Ensure dependencies are initialized
        if not initialize_dependencies():
            logger.error("Failed to initialize dependencies")
            return jsonify({'error': 'Failed to initialize dependencies'}), 500
        
        # Get the image data from the request
        data = request.json
        if not data or 'image' not in data:
            logger.error("No image data received")
            return jsonify({'error': 'No image data received'}), 400
        
        try:
            # Decode the base64 image
            image_data = data['image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            np_arr = np.frombuffer(image_bytes, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error("Failed to decode image")
                return jsonify({'error': 'Failed to decode image'}), 400
            
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return jsonify({'error': f'Error decoding image: {str(e)}'}), 400
        
        # Process the frame
        try:
            processed_frame, status, current_detections = process_frame(frame)
            
            if processed_frame is None:
                logger.error("Failed to process frame")
                return jsonify({'error': 'Failed to process frame'}), 500
            
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
            logger.error(f"Error processing frame: {str(e)}")
            return jsonify({'error': f'Error processing frame: {str(e)}'}), 500
        
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
def get_model_status():
    """Get detailed model status information"""
    global model_status
    
    return jsonify({
        'yolo_loaded': model_status['yolo_loaded'],
        'yolo_error': model_status['yolo_error'],
        'opencv_loaded': model_status['opencv_loaded'],
        'opencv_error': model_status['opencv_error'],
        'last_error': model_status['last_error']
    })

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