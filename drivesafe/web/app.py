from flask import Flask, render_template, Response, request, jsonify, url_for, send_from_directory, stream_with_context
import os
import torch
import numpy as np
from datetime import datetime
import sys
import time
from werkzeug.utils import secure_filename
import traceback
import logging
from pathlib import Path
import base64
import threading
import queue
import uuid
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Class names
class_names = ['red', 'yellow', 'green', 'off', 'person', 'lane']

# Global variables
camera = None
is_detecting = False
model = None
yolo_model = None
model_loading = False
processing_queue = queue.Queue(maxsize=10)
processing_active = False
detected_classes = {
    'red': 0,
    'green': 0,
    'yellow': 0,
    'none': 0
}

def init_yolo_model():
    """Initialize the YOLO model in a separate thread"""
    global yolo_model, model_loading
    
    if yolo_model is not None or model_loading:
        return
    
    model_loading = True
    
    try:
        logger.info("Initializing YOLO model...")
        start_time = time.time()
        
        # Check if we're running in production
        if os.environ.get('FLASK_ENV') == 'production':
            # In production, the model should already be initialized
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from ultralytics import YOLO
            model_path = os.path.join('drivesafe', 'models', 'traffic_light', 'best_traffic_small_yolo.pt')
            yolo_model = YOLO(model_path)
        else:
            # In development, try to use the initialization script
            try:
                # Try running the initialization script
                script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'init_model.py')
                if os.path.exists(script_path):
                    logger.info(f"Running model initialization script: {script_path}")
                    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
                    if result.returncode != 0:
                        logger.error(f"Model initialization script failed: {result.stderr}")
                        raise Exception(f"Model initialization script failed: {result.stderr}")
                    else:
                        logger.info(f"Model initialization script succeeded: {result.stdout}")
                
                # Load the model directly
                from ultralytics import YOLO
                model_path = os.path.join('drivesafe', 'models', 'traffic_light', 'best_traffic_small_yolo.pt')
                yolo_model = YOLO(model_path)
            except Exception as e:
                logger.error(f"Error loading YOLO model: {str(e)}")
                traceback.print_exc()
                raise
        
        elapsed_time = time.time() - start_time
        logger.info(f"YOLO model initialized successfully in {elapsed_time:.2f} seconds")
    except Exception as e:
        logger.error(f"Failed to initialize YOLO model: {str(e)}")
        traceback.print_exc()
    finally:
        model_loading = False

def start_model_loading_thread():
    """Start a thread to load the YOLO model"""
    thread = threading.Thread(target=init_yolo_model)
    thread.daemon = True
    thread.start()
    return thread

# Start loading the model in the background when the app starts
model_thread = start_model_loading_thread()

def get_model():
    """Lazy load the model only when needed to avoid import issues"""
    global model
    if model is None:
        try:
            import cv2
            from ultralytics import YOLO
            print("Initializing YOLO model...")
            # Initialize model
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"Using device: {device}")
            
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'traffic_light', 'best_traffic_small_yolo.pt')
            print(f"Loading YOLO model from: {model_path}")
            if not os.path.exists(model_path):
                print(f"ERROR: Model file not found at {model_path}")
                return None
            
            model = YOLO(model_path)
            print("YOLO model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    return model

def get_camera():
    """Get or initialize the camera"""
    global camera
    try:
        import cv2
        if camera is None:
            print("Initializing camera...")
            camera = cv2.VideoCapture(0)
            # Set camera properties for better performance
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            camera.set(cv2.CAP_PROP_FPS, 30)
            camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
            if not camera.isOpened():
                print("Warning: Camera could not be opened")
                return None
            print("Camera initialized successfully")
        return camera
    except Exception as e:
        print(f"Error initializing camera: {str(e)}")
        return None

def release_camera():
    """Release the camera resources"""
    global camera, is_detecting
    if camera is not None:
        try:
            import cv2
            camera.release()
            print("Camera released")
        except Exception as e:
            print(f"Error releasing camera: {str(e)}")
        finally:
            camera = None
    is_detecting = False

def detect_objects(frame):
    """Detect objects in a frame using YOLO"""
    import cv2
    # Get model
    model = get_model()
    if model is None:
        # If model loading failed, just return the original frame with no detections
        return frame, []
    
    try:
        # Run inference with YOLO
        results = model(frame, conf=0.25)
        
        # Process detections
        processed_frame = frame.copy()
        detections = []
        
        # Process traffic light detections
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Get class name and color
                class_name = r.names[cls].lower() if cls < len(r.names) else "unknown"
                
                # Add detection to results
                detections.append({
                    'label': class_name,
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                
                # Draw detection box with appropriate color based on class
                color = (0, 255, 0)  # Default green
                if 'red' in class_name:
                    color = (0, 0, 255)  # Red
                elif 'yellow' in class_name:
                    color = (0, 255, 255)  # Yellow
                elif 'green' in class_name:
                    color = (0, 255, 0)  # Green
                elif 'person' in class_name:
                    color = (255, 0, 0)  # Blue
                
                # Draw rectangle and label
                cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(processed_frame, f'{class_name} {conf:.2f}',
                        (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Print detection summary for logging purposes
        if detections:
            print(f"Detected {len(detections)} objects: {[d['label'] for d in detections]}")
        
        return processed_frame, detections
    
    except Exception as e:
        print(f"Error during detection: {str(e)}")
        import traceback
        traceback.print_exc()
        return frame, []

def generate_frames():
    """Generate video frames for streaming"""
    global is_detecting
    import cv2
    
    # Ensure model is preloaded
    print("Preloading model before starting camera feed...")
    model = get_model()
    if model is None:
        print("WARNING: Failed to load YOLO model")
    
    is_detecting = True
    print("Starting detection...")
    frame_count = 0
    error_count = 0
    max_errors = 5
    
    while is_detecting:
        try:
            # Get camera
            camera = get_camera()
            if not camera or not camera.isOpened():
                print("Error: Camera not available")
                time.sleep(0.5)
                error_count += 1
                if error_count > max_errors:
                    print("Too many camera errors, stopping detection")
                    break
                continue
            
            # Read frame
            success, frame = camera.read()
            if not success:
                print("Error: Could not read frame")
                time.sleep(0.5)
                error_count += 1
                if error_count > max_errors:
                    print("Too many frame reading errors, stopping detection")
                    break
                continue
            
            error_count = 0  # Reset error count on successful frame read
            frame_count += 1
            
            # Mirror the frame for more natural webcam view
            frame = cv2.flip(frame, 1)
            
            # Process frame with detection if model is available
            if model is not None:
                frame, detections = detect_objects(frame)
            
            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (255, 255, 255), 2)
            
            # Add frame counter (for debugging)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1)
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print("Error: Could not encode frame")
                continue
                
            frame = buffer.tobytes()
            
            # Yield the frame in multipart response format
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
                
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Brief pause to prevent tight loop if errors occur
            time.sleep(0.5)
            error_count += 1
            if error_count > max_errors:
                print("Too many errors in generate_frames, stopping detection")
                break
    
    print("Stopping detection and releasing camera...")
    release_camera()

@app.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')

@app.route('/start_drive')
def start_drive():
    """Render the live detection page"""
    return render_template('start_drive.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route with increased robustness"""
    try:
        # Add CORS headers for better streaming compatibility
        response = Response(
            stream_with_context(generate_frames()),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
        response.headers.add('Cache-Control', 'no-cache, no-store, must-revalidate')
        response.headers.add('Pragma', 'no-cache')
        response.headers.add('Expires', '0')
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
    except Exception as e:
        print(f"Error in video_feed route: {str(e)}")
        import traceback
        traceback.print_exc()
        return Response('Video feed error: ' + str(e), status=500)

@app.route('/stop_video')
def stop_video():
    """Stop the video feed and release the camera"""
    global is_detecting
    print("Received stop_video request")
    is_detecting = False
    return jsonify({'status': 'success'})

@app.route('/upload_drive')
def upload_drive():
    """Render the upload page"""
    return render_template('upload_drive.html')

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
        
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
        
    if video:
        try:
            filename = secure_filename(video.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            video.save(filepath)
            
            # Process video
            cap = cv2.VideoCapture(filepath)
            if not cap.isOpened():
                return jsonify({'error': 'Could not open video file'}), 400
                
            processed_filename = 'processed_' + filename
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], processed_filename)
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(processed_filepath, fourcc, fps, (width, height))
            
            detections_summary = {
                'traffic_lights': {'red': 0, 'yellow': 0, 'green': 0},
                'pedestrians': 0,
                'lane_markings': 0
            }
            
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame, frame_detections = detect_objects(frame)
                
                # Update detection counts
                for det in frame_detections:
                    if det['label'] == 'red':
                        detections_summary['traffic_lights']['red'] += 1
                    elif det['label'] == 'yellow':
                        detections_summary['traffic_lights']['yellow'] += 1
                    elif det['label'] == 'green':
                        detections_summary['traffic_lights']['green'] += 1
                    elif det['label'] == 'person':
                        detections_summary['pedestrians'] += 1
                    elif det['label'] == 'lane':
                        detections_summary['lane_markings'] += 1
                
                out.write(processed_frame)
                frame_count += 1
            
            cap.release()
            out.release()
            
            # Clean up original video
            os.remove(filepath)
            
            # Calculate total traffic lights
            total_lights = sum(detections_summary['traffic_lights'].values())
            
            return jsonify({
                'status': 'success',
                'video_url': url_for('static', filename=f'uploads/{processed_filename}'),
                'detections': {
                    'traffic_lights': {
                        'total': total_lights,
                        'details': detections_summary['traffic_lights']
                    },
                    'pedestrians': detections_summary['pedestrians'],
                    'lane_markings': detections_summary['lane_markings'],
                    'frames_processed': frame_count
                }
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid video file'}), 400

@app.route('/process_frame', methods=['POST'])
def process_image():
    global detected_classes, yolo_model
    
    try:
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
            'model_loading': model_loading,
            'detections': detected_classes,
            'current_detections': current_detections
        })
        
    except Exception as e:
        logger.error(f"Error in process_frame endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'error': str(e),
            'model_loaded': yolo_model is not None,
            'model_loading': model_loading
        }), 500

@app.route('/stats', methods=['GET'])
def get_stats():
    global detected_classes, yolo_model
    
    return jsonify({
        'detections': detected_classes,
        'model_loaded': yolo_model is not None,
        'model_loading': model_loading
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
    global yolo_model, model_loading
    
    return jsonify({
        'model_loaded': yolo_model is not None,
        'model_loading': model_loading
    })

def process_frame(frame, save_path=None):
    """Process a frame with the YOLO model and return the results"""
    global yolo_model, detected_classes
    
    try:
        # If model is not loaded yet, wait for it
        if yolo_model is None:
            if not model_loading:
                start_model_loading_thread()
            return frame, "Model loading", None
        
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
        traceback.print_exc()
        return frame, f"Processing error: {str(e)}", None

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000))) 