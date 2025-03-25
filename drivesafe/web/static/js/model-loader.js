/**
 * Model Loader for Traffic Light Detection
 * 
 * Handles downloading, caching, and using the YOLO model for traffic light detection.
 * Uses TensorFlow.js and IndexedDB for model storage.
 */

// Configuration
const LOCAL_MODEL_URL = "/static/models/tfjs_model/model.json"; // Local model path in Flask static directory
const FALLBACK_MODEL_URL = "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json"; // Using Google's hosted model as fallback
const MODEL_CACHE_KEY = "traffic-light-model-v1";
const MAX_RETRIES = 3;
const RETRY_DELAY = 2000; // 2 seconds

// Status constants
const STATUS = {
    INITIALIZING: "initializing",
    DOWNLOADING: "downloading",
    LOADING: "loading",
    READY: "ready",
    ERROR: "error"
};

class ModelLoader {
    constructor() {
        this.model = null;
        this.status = STATUS.INITIALIZING;
        this.progress = 0;
        this.error = null;
        this.retryCount = 0;
        this.statusListeners = [];
        
        // Try to initialize the model right away
        this.init();
    }
    
    /**
     * Initialize the model loading process
     */
    async init() {
        try {
            // First check if TensorFlow.js is available
            if (!window.tf) {
                console.error("TensorFlow.js is not loaded. Make sure to include the script in your HTML.");
                this.setStatus(STATUS.ERROR, "TensorFlow.js is not available");
                return;
            }
            
            // Check for WebGL support (required for model inference)
            const webglSupported = await this.checkWebGLSupport();
            if (!webglSupported) {
                this.setStatus(STATUS.ERROR, "WebGL is not supported in your browser");
                return;
            }
            
            // Check for IndexedDB support (for model caching)
            const indexedDBSupported = await this.checkIndexedDBSupport();
            if (!indexedDBSupported) {
                console.warn("IndexedDB is not supported. Model will not be cached.");
            }
            
            // Start loading the model
            await this.loadModel();
        } catch (err) {
            console.error("Error initializing model:", err);
            this.setStatus(STATUS.ERROR, err.message);
        }
    }
    
    /**
     * Check if WebGL is supported
     */
    async checkWebGLSupport() {
        try {
            const canvas = document.createElement('canvas');
            const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
            return !!gl;
        } catch (e) {
            return false;
        }
    }
    
    /**
     * Check if IndexedDB is supported
     */
    async checkIndexedDBSupport() {
        try {
            return !!window.indexedDB;
        } catch (e) {
            return false;
        }
    }
    
    /**
     * Load model from cache or download
     */
    async loadModel() {
        try {
            // Try to load from cache first
            if (await this.isModelCached()) {
                this.setStatus(STATUS.LOADING, "Loading model from cache...");
                await this.loadModelFromCache();
            } else {
                // If not cached, download the model
                this.setStatus(STATUS.DOWNLOADING, "Downloading model...");
                await this.downloadModel();
            }
        } catch (err) {
            console.error("Error loading model:", err);
            
            // Retry logic
            if (this.retryCount < MAX_RETRIES) {
                this.retryCount++;
                this.setStatus(STATUS.INITIALIZING, `Retrying (${this.retryCount}/${MAX_RETRIES})...`);
                
                // Wait before retrying
                await new Promise(resolve => setTimeout(resolve, RETRY_DELAY));
                await this.loadModel();
            } else {
                this.setStatus(STATUS.ERROR, `Failed to load model: ${err.message}`);
            }
        }
    }
    
    /**
     * Check if model is cached in IndexedDB
     */
    async isModelCached() {
        try {
            return await tf.io.listModels().then(models => {
                return Object.keys(models).includes(`indexeddb://${MODEL_CACHE_KEY}`);
            });
        } catch (err) {
            console.warn("Error checking model cache:", err);
            return false;
        }
    }
    
    /**
     * Load model from IndexedDB cache
     */
    async loadModelFromCache() {
        try {
            console.log("Loading model from cache...");
            this.model = await tf.loadGraphModel(`indexeddb://${MODEL_CACHE_KEY}`);
            
            // Warmup the model with a dummy tensor
            await this.warmupModel();
            
            this.setStatus(STATUS.READY);
            console.log("Model loaded from cache successfully");
        } catch (err) {
            console.error("Error loading model from cache:", err);
            
            // If cache is corrupted, try downloading again
            console.log("Falling back to downloading model...");
            await this.downloadModel();
        }
    }
    
    /**
     * Download model from server
     */
    async downloadModel() {
        try {
            console.log("Trying to download model...");
            this.setStatus(STATUS.DOWNLOADING);
            
            // First try to load from local server
            try {
                console.log("Trying local model first...");
                this.model = await tf.loadGraphModel(LOCAL_MODEL_URL, {
                    onProgress: (fraction) => {
                        this.progress = Math.round(fraction * 100);
                        this.notifyListeners();
                    }
                });
                console.log("Local model loaded successfully");
            } catch (localError) {
                // If local model fails, try the fallback CDN
                console.log("Local model failed, trying fallback CDN...", localError);
                this.model = await tf.loadGraphModel(FALLBACK_MODEL_URL, {
                    onProgress: (fraction) => {
                        this.progress = Math.round(fraction * 100);
                        this.notifyListeners();
                    }
                });
                console.log("Fallback model loaded successfully");
            }
            
            // Cache the model for future use
            try {
                await this.model.save(`indexeddb://${MODEL_CACHE_KEY}`);
                console.log("Model cached successfully");
            } catch (err) {
                console.warn("Failed to cache model:", err);
                // Continue even if caching fails
            }
            
            // Warmup the model
            await this.warmupModel();
            
            this.setStatus(STATUS.READY);
            console.log("Model downloaded and ready");
        } catch (err) {
            console.error("Error downloading model:", err);
            throw err;
        }
    }
    
    /**
     * Warmup the model with a dummy tensor
     */
    async warmupModel() {
        try {
            this.setStatus(STATUS.LOADING, "Warming up model...");
            
            // Create a dummy tensor for warmup
            const dummyTensor = tf.zeros([1, 640, 480, 3]);
            
            // Run inference on the dummy tensor
            const result = await this.model.executeAsync(dummyTensor);
            
            // Clean up memory
            tf.dispose([dummyTensor, ...result]);
            
            console.log("Model warmup complete");
        } catch (err) {
            console.error("Error during model warmup:", err);
            // Continue even if warmup fails
        }
    }
    
    /**
     * Process a video frame for traffic light detection
     * 
     * @param {HTMLVideoElement|HTMLCanvasElement|HTMLImageElement} videoElement The video or canvas element to process
     * @returns {Object} Detection results
     */
    async detectTrafficLights(videoElement) {
        if (this.status !== STATUS.READY) {
            throw new Error(`Model is not ready (status: ${this.status})`);
        }
        
        try {
            // Convert the video frame to a tensor
            const tensor = tf.browser.fromPixels(videoElement)
                .resizeBilinear([640, 480]) // Resize to model input size
                .expandDims(0) // Add batch dimension
                .div(255.0); // Normalize pixel values
            
            // Run model inference
            const result = await this.model.executeAsync(tensor);
            
            // Process results to get bounding boxes, classes, and scores
            const [boxes, scores, classes, valid_detections] = result;
            
            // Convert tensors to arrays for easier use
            const boxesArray = boxes.arraySync()[0];
            const scoresArray = scores.arraySync()[0];
            const classesArray = classes.arraySync()[0];
            const validDetections = valid_detections.arraySync()[0];
            
            // Clean up memory
            tf.dispose([tensor, ...result]);
            
            // Filter results to include only traffic lights with sufficient confidence
            const detections = [];
            for (let i = 0; i < validDetections; i++) {
                if (scoresArray[i] > 0.5) { // Confidence threshold
                    detections.push({
                        box: boxesArray[i], // [y1, x1, y2, x2] normalized coordinates
                        score: scoresArray[i],
                        class: classesArray[i]
                    });
                }
            }
            
            // Count traffic lights by color (assuming class 0=red, 1=yellow, 2=green)
            const counts = {
                red: 0,
                yellow: 0,
                green: 0,
                unknown: 0
            };
            
            detections.forEach(detection => {
                if (detection.class === 0) counts.red++;
                else if (detection.class === 1) counts.yellow++;
                else if (detection.class === 2) counts.green++;
                else counts.unknown++;
            });
            
            return {
                detections,
                counts
            };
        } catch (err) {
            console.error("Error during traffic light detection:", err);
            throw err;
        }
    }
    
    /**
     * Draw bounding boxes on a canvas
     * 
     * @param {HTMLCanvasElement} canvas The canvas to draw on
     * @param {Object} detectionResult The result from detectTrafficLights
     */
    drawDetections(canvas, detectionResult) {
        const ctx = canvas.getContext('2d');
        const detections = detectionResult.detections;
        
        // Clear the canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Draw each detection
        detections.forEach(detection => {
            const [y1, x1, y2, x2] = detection.box;
            const width = canvas.width;
            const height = canvas.height;
            
            // Convert normalized coordinates to pixel coordinates
            const x = x1 * width;
            const y = y1 * height;
            const w = (x2 - x1) * width;
            const h = (y2 - y1) * height;
            
            // Set color based on class
            let color;
            let label;
            if (detection.class === 0) {
                color = 'red';
                label = 'Red';
            } else if (detection.class === 1) {
                color = 'yellow';
                label = 'Yellow';
            } else if (detection.class === 2) {
                color = 'lime';
                label = 'Green';
            } else {
                color = 'white';
                label = 'Unknown';
            }
            
            // Draw bounding box
            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
            
            // Draw label background
            ctx.fillStyle = color;
            const textWidth = ctx.measureText(label).width;
            ctx.fillRect(x, y - 20, textWidth + 10, 20);
            
            // Draw label text
            ctx.fillStyle = 'black';
            ctx.font = '16px Arial';
            ctx.fillText(label, x + 5, y - 5);
        });
    }
    
    /**
     * Set the model status and notify listeners
     * 
     * @param {string} status The new status
     * @param {string} message Optional message
     */
    setStatus(status, message = null) {
        this.status = status;
        this.error = message && status === STATUS.ERROR ? message : null;
        this.notifyListeners();
    }
    
    /**
     * Add a status listener
     * 
     * @param {Function} listener Function to call when status changes
     */
    addStatusListener(listener) {
        this.statusListeners.push(listener);
    }
    
    /**
     * Remove a status listener
     * 
     * @param {Function} listener Function to remove
     */
    removeStatusListener(listener) {
        this.statusListeners = this.statusListeners.filter(l => l !== listener);
    }
    
    /**
     * Notify all listeners of status change
     */
    notifyListeners() {
        const status = {
            status: this.status,
            progress: this.progress,
            error: this.error
        };
        
        this.statusListeners.forEach(listener => {
            try {
                listener(status);
            } catch (err) {
                console.error("Error in status listener:", err);
            }
        });
    }
}

// Create a singleton instance
const modelLoader = new ModelLoader();

// Export the singleton
window.trafficLightModel = modelLoader; 