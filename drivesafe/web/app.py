from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import os
import logging
import base64
import numpy as np
import traceback
from pathlib import Path
import shutil

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
STATIC_MODELS_DIR = os.path.join(STATIC_DIR, 'models')
REPO_MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')

# Ensure models directories exist
os.makedirs(STATIC_MODELS_DIR, exist_ok=True)
os.makedirs(REPO_MODELS_DIR, exist_ok=True)

# If there are models in the repo models directory but not in static, copy them
def copy_models_to_static():
    """Copy models from the repository models directory to the static models directory."""
    try:
        logger.info("Checking for models to copy to static directory...")
        
        # Check if repository has tfjs_model
        repo_tfjs_dir = os.path.join(REPO_MODELS_DIR, 'tfjs_model')
        static_tfjs_dir = os.path.join(STATIC_MODELS_DIR, 'tfjs_model')
        
        if os.path.exists(repo_tfjs_dir) and not os.path.exists(static_tfjs_dir):
            logger.info(f"Copying models from {repo_tfjs_dir} to {static_tfjs_dir}")
            os.makedirs(static_tfjs_dir, exist_ok=True)
            
            # Copy all files
            for file in os.listdir(repo_tfjs_dir):
                source = os.path.join(repo_tfjs_dir, file)
                destination = os.path.join(static_tfjs_dir, file)
                if os.path.isfile(source):
                    shutil.copy2(source, destination)
            
            logger.info("Model files copied successfully")
    except Exception as e:
        logger.error(f"Error copying models: {str(e)}")

# Run the copy operation at startup
copy_models_to_static()

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
    # Check if model files exist in static directory
    model_json_path = os.path.join(STATIC_MODELS_DIR, 'tfjs_model', 'model.json')
    model_exists = os.path.exists(model_json_path)
    
    return jsonify({
        "status": "client_side_model",
        "model_exists": model_exists,
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