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
cv2 = None
torch = None
YOLO = None
model_lock = threading.Lock()
frame_queue = Queue(maxsize=2)  # Limit queue size to prevent memory issues
detected_classes = {
    'red': 0,
    'green': 0,
    'yellow': 0,
    'none': 0
}

def initialize_dependencies():
    """Initialize all required dependencies"""
    global cv2, torch, YOLO
    
    try:
        if cv2 is None:
            import cv2 as cv2_import
            cv2 = cv2_import
            logger.info("OpenCV initialized successfully")
            
        if torch is None:
            import torch as torch_import
            torch = torch_import
            logger.info("PyTorch initialized successfully")
            
        if YOLO is None:
            from ultralytics import YOLO as YOLO_import
            YOLO = YOLO_import
            logger.info("YOLO imported successfully")
            
        return True
    except Exception as e:
        logger.error(f"Failed to initialize dependencies: {str(e)}")
        return False

def init_yolo_model():
    """Initialize the YOLO model efficiently with caching"""
    global yolo_model
    
    with model_lock:
        if yolo_model is not None:
            return yolo_model
        
        try:
            # Initialize dependencies first
            if not initialize_dependencies():
                logger.error("Failed to initialize dependencies")
                return None
            
            # Initialize device and set PyTorch settings
            if torch.cuda.is_available():
                device = torch.device('cuda')
                # Enable TF32 for better performance on Ampere GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            else:
                device = torch.device('cpu')
                # Enable OpenMP for CPU
                if hasattr(torch, 'set_num_threads'):
                    torch.set_num_threads(4)  # Adjust based on CPU cores
            
            logger.info(f"Using device: {device}")
            
            # Look for model in multiple locations
            model_paths = [
                'best_traffic_small_yolo.pt',
                os.path.join('models', 'traffic_light', 'best_traffic_small_yolo.pt'),
                os.path.join(os.path.dirname(__file__), 'best_traffic_small_yolo.pt'),
                os.path.join(os.path.dirname(__file__), 'models', 'traffic_light', 'best_traffic_small_yolo.pt')
            ]
            
            model_loaded = False
            for model_path in model_paths:
                if os.path.exists(model_path):
                    logger.info(f"Loading model from: {model_path}")
                    try:
                        yolo_model = YOLO(model_path)
                        if device.type == 'cpu':
                            yolo_model.cpu()
                        else:
                            yolo_model.to(device)
                        
                        # Warmup the model
                        logger.info("Warming up model...")
                        dummy_input = torch.zeros((1, 3, 640, 480), device=device)
                        for _ in range(2):
                            if device.type == 'cuda':
                                torch.cuda.synchronize()
                            _ = yolo_model(dummy_input)
                        
                        logger.info("Model warmup complete")
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.error(f"Failed to load model from {model_path}: {str(e)}")
                        continue
            
            if not model_loaded:
                logger.error("Could not load model from any path")
                return None
            
            return yolo_model
            
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {str(e)}")
            return None

def process_frame(frame, save_path=None):
    """Process a frame with the YOLO model and return the results"""
    global yolo_model, detected_classes, cv2
    
    try:
        # Quick check if model is loaded
        if yolo_model is None:
            return frame, "Model not initialized", None
        
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        
        # Process the frame with the YOLO model
        try:
            # Convert frame to RGB (YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run inference with lower confidence threshold
            with model_lock:
                results = yolo_model(frame_rgb, conf=0.3, iou=0.5)
        except Exception as e:
            logger.error(f"Error during model inference: {str(e)}")
            return frame, f"Error during detection: {str(e)}", None
        
        # Get the plotted frame with bounding boxes
        try:
            annotated_frame = results[0].plot()
            # Convert back to BGR for OpenCV
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")
            return frame, "Error visualizing results", None
        
        # Extract class information from results
        boxes = results[0].boxes
        current_detected = {"red": 0, "green": 0, "yellow": 0, "none": 0}
        
        if len(boxes) > 0:
            for box in boxes:
                try:
                    # Get class index and confidence
                    cls = int(box.cls.item())
                    conf = float(box.conf.item())
                    class_name = results[0].names[cls].lower()
                    
                    # Only count detections with confidence > 0.3
                    if conf > 0.3 and class_name in current_detected:
                        current_detected[class_name] += 1
                        # Add confidence to the frame
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        cv2.putText(annotated_frame, f"{conf:.2f}", (x1, y1-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                except Exception as e:
                    logger.error(f"Error processing detection box: {str(e)}")
                    continue
        
        # Update global detection counters
        for key in detected_classes:
            if current_detected[key] > 0:
                detected_classes[key] += current_detected[key]
        
        # Determine overall detection status
        if current_detected["red"] > 0:
            detection_status = f"Red light detected! ({current_detected['red']} instances)"
        elif current_detected["yellow"] > 0:
            detection_status = f"Yellow light detected! ({current_detected['yellow']} instances)"
        elif current_detected["green"] > 0:
            detection_status = f"Green light detected! ({current_detected['green']} instances)"
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
def model_status():
    global yolo_model
    
    return jsonify({
        'model_loaded': yolo_model is not None
    })

if __name__ == '__main__':
    # Initialize dependencies at startup
    if not initialize_dependencies():
        logger.error("Failed to initialize dependencies. Exiting.")
        sys.exit(1)
    
    # Initialize model at startup in a separate thread
    model_thread = threading.Thread(target=init_yolo_model)
    model_thread.start()
    
    # Start the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port) 