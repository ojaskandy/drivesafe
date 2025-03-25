from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
import logging
import traceback
import base64

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

def detect_lanes(frame):
    """Process a frame to detect lanes."""
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        logger.debug("Converted frame to grayscale")
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        logger.debug("Applied Gaussian blur")
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        logger.debug("Applied Canny edge detection")
        
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
        logger.debug("Applied region of interest mask")
        
        # Apply Hough transform
        lines = cv2.HoughLinesP(
            masked_edges,
            rho=1,
            theta=np.pi/180,
            threshold=20,
            minLineLength=40,
            maxLineGap=20
        )
        logger.debug(f"Detected {len(lines) if lines is not None else 0} lines")
        
        # Create a copy of the frame for drawing
        result = frame.copy()
        
        # Draw lines on the frame
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            return result, len(lines)
        
        return result, 0
        
    except Exception as e:
        logger.error(f"Error in lane detection: {str(e)}")
        logger.error(traceback.format_exc())
        return frame, 0

@app.route('/')
def home():
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
            logger.error("No image data received")
            return jsonify({'error': 'No image data received'})
        
        # Convert base64 image to OpenCV format
        try:
            img_data = data['image'].split(',')[1]
            img_bytes = base64.b64decode(img_data)
            img_np = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
            logger.debug("Successfully decoded image")
        except Exception as e:
            logger.error(f"Error decoding image: {str(e)}")
            return jsonify({'error': 'Invalid image data'})
        
        if frame is None:
            logger.error("Failed to decode image")
            return jsonify({'error': 'Invalid image data'})
        
        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        logger.debug("Resized frame to 640x480")
        
        # Process frame with lane detection
        processed_frame, num_lanes = detect_lanes(frame)
        logger.debug(f"Processed frame, detected {num_lanes} lanes")
        
        # Convert processed frame back to base64
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'processed_image': f'data:image/jpeg;base64,{img_base64}',
            'detection_status': f"Lane detection active ({num_lanes} lanes)" if num_lanes > 0 else "No lanes detected"
        })
        
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)})

@app.route('/model_status')
def get_model_status():
    """Get the status of OpenCV."""
    try:
        # Simple test to check if OpenCV is working
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.line(test_img, (0, 0), (99, 99), (255, 255, 255), 2)
        logger.debug("OpenCV test successful")
        
        return jsonify({
            'opencv_loaded': True,
            'opencv_error': None
        })
    except Exception as e:
        logger.error(f"OpenCV test failed: {str(e)}")
        return jsonify({
            'opencv_loaded': False,
            'opencv_error': str(e)
        })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
