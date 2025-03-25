from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import os
import logging
import base64
import numpy as np
import traceback
from pathlib import Path

# Configure logging for better debugging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)

# Path to static assets
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Serve model files for client-side loading
@app.route('/models/<path:filename>')
def serve_model(filename):
    """Serve model files from the models directory."""
    return send_from_directory(MODELS_DIR, filename)

@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')

@app.route('/start-drive')
def start_drive():
    """Render the start drive page."""
    return render_template('start_drive.html')

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/model_status')
def get_model_status():
    """Return model status - this is a placeholder since model status is now handled client-side."""
    return jsonify({
        "status": "client_side_model",
        "message": "Model loading is handled client-side with TensorFlow.js"
    })

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Get port from environment variable or use 5000 as default
    port = int(os.environ.get('PORT', 5000))
    # Use 0.0.0.0 to make the server publicly available
    app.run(host='0.0.0.0', port=port, debug=True) 