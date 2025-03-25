# DriveSafe

An AI-powered application for traffic light and lane detection to improve driving safety.

## Features

- 🚦 **Traffic Light Detection**: Automatically detects and classifies traffic lights as red, yellow, or green
- 🛣️ **Lane Detection**: Identifies road lanes to help maintain proper positioning
- ⚡ **Real-time Processing**: Performs detections in real-time directly in your browser
- 🔒 **Privacy-focused**: All processing happens on your device - no video data is sent to any server

## Quick Start

The easiest way to run DriveSafe is using the provided run script:

```bash
# On macOS/Linux
python run_app.py

# On Windows
python run_app.py
```

This will:
1. Check if a converted model is available
2. Convert a YOLO model file (if found in the `models/` directory) to TensorFlow.js format
3. Start the application

Then open your browser to http://localhost:5000

## Requirements

### Basic Requirements

- Python 3.7 or higher
- Flask
- Access to a webcam (for live detection)

### Model Conversion Requirements (optional)
Only needed if you want to convert your own YOLO model:

- torch
- onnx
- onnx-tf
- tensorflow
- tensorflowjs
- ultralytics

You can install these with:

```bash
pip install torch onnx onnx-tf tensorflow tensorflowjs ultralytics
```

## Manual Setup

If you prefer to set up manually:

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. If you have a YOLO model file (`.pt`) and want to convert it:
   ```bash
   python convert_model.py --model models/your_model.pt --output converted_model
   ```

3. Start the application:
   ```bash
   cd drivesafe/web
   python app.py
   ```

4. Open your browser to http://localhost:5000

## How It Works

DriveSafe uses TensorFlow.js to run a YOLO object detection model directly in your browser. This means:

- The model loads and runs on your device
- After the first load, it's cached for faster startup
- No video data leaves your device
- The app continues working even when offline (after initial load)

## Browser Compatibility

DriveSafe works best with:
- Chrome (recommended)
- Firefox
- Edge
- Safari (12+)

For best performance, use a recent browser version with WebGL support.

## Troubleshooting

### Model Loading Issues
- If the model fails to load, the app will try to use a fallback from a CDN
- For best performance, place your YOLO model file (`.pt`) in the `models/` directory and run `run_app.py`

### Camera Issues
- Make sure your browser has permission to access the camera
- If the camera doesn't start, try reloading the page
- For some computers, using an external webcam may work better than the built-in one

### Performance Issues
- Lower-end devices may experience reduced framerate
- Close other tabs/applications to free up resources
- If detection is slow, the app automatically limits the frame rate to maintain stability

## License

This project is licensed under the MIT License - see the LICENSE file for details. 