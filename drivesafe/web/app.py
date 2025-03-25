from flask import Flask, render_template, Response, request, jsonify, url_for, send_from_directory
import cv2
import os
import torch
import numpy as np
from datetime import datetime
import sys
from werkzeug.utils import secure_filename
from ultralytics import YOLO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static', 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'traffic_light', 'best_traffic_small_yolo.pt')
model = YOLO(model_path)

# Class names
class_names = ['red', 'yellow', 'green', 'off', 'person', 'lane']

# Global variables
camera = None
is_detecting = False

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        # Set camera properties for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    return camera

def release_camera():
    global camera, is_detecting
    if camera is not None:
        camera.release()
        camera = None
    is_detecting = False

def detect_objects(frame):
    # Run inference
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
            class_name = r.names[cls].lower()
            
            # Add detection to results
            detections.append({
                'label': class_name,
                'confidence': conf,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
            
            # Draw detection box
            color = (0, 255, 0)  # Default green
            if 'red' in class_name:
                color = (0, 0, 255)  # Red
            elif 'yellow' in class_name:
                color = (0, 255, 255)  # Yellow
            elif 'person' in class_name:
                color = (255, 0, 0)  # Blue
            
            cv2.rectangle(processed_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(processed_frame, f'{class_name} {conf:.2f}',
                      (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return processed_frame, detections

def generate_frames():
    global is_detecting
    is_detecting = True
    
    while is_detecting:
        camera = get_camera()
        if not camera.isOpened():
            break
            
        success, frame = camera.read()
        if not success:
            break
            
        try:
            # Mirror the frame for more natural webcam view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            frame, detections = detect_objects(frame)
            
            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.8, (255, 255, 255), 2)
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
                
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        except Exception as e:
            print(f"Error in generate_frames: {str(e)}")
            break
    
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
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    """Stop the video feed and release the camera"""
    release_camera()
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

if __name__ == '__main__':
    app.run(debug=True) 