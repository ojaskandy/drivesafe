/**
 * Model Loader for Traffic Light Detection
 * 
 * Handles downloading, caching, and using the YOLO model for traffic light detection.
 * Uses TensorFlow.js and IndexedDB for model storage.
 */

// Configuration
const LOCAL_MODEL_URL = "/static/models/tfjs_model/model.json"; // Local model path in Flask static directory
const FALLBACK_MODEL_URL = "https://storage.googleapis.com/tfjs-models/savedmodel/ssd_mobilenet_v2/model.json"; // CORS-friendly Google Storage URL
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
        this.useSimulation = false;
        
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
                this.useSimulation = true;
                return;
            }
            
            // Check for WebGL support (required for model inference)
            const webglSupported = await this.checkWebGLSupport();
            if (!webglSupported) {
                console.warn("WebGL is not supported in your browser. Using simulation mode.");
                this.setStatus(STATUS.ERROR, "WebGL is not supported in your browser");
                this.useSimulation = true;
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
            this.useSimulation = true;
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
                console.warn("Failed after multiple retries. Switching to simulation mode.");
                this.setStatus(STATUS.ERROR, `Failed to load model: ${err.message}`);
                this.useSimulation = true;
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
                // If local model fails, try the fallback CDN with CORS-friendly URL
                console.log("Local model failed, trying fallback CDN...", localError);
                
                try {
                    this.model = await tf.loadGraphModel(FALLBACK_MODEL_URL, {
                        onProgress: (fraction) => {
                            this.progress = Math.round(fraction * 100);
                            this.notifyListeners();
                        }
                    });
                    console.log("Fallback model loaded successfully");
                } catch (fallbackError) {
                    console.error("Fallback model also failed:", fallbackError);
                    console.warn("Switching to simulation mode");
                    this.useSimulation = true;
                    throw new Error("All model loading attempts failed");
                }
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
        // If we're in simulation mode, just return simulated traffic lights
        if (this.useSimulation || this.status !== STATUS.READY) {
            return this.simulateTrafficLights();
        }
        
        try {
            // Convert the video frame to a tensor - fixing the type issue
            const tensor = tf.tidy(() => {
                // Start with pixel data as in the range 0-255
                const pixels = tf.browser.fromPixels(videoElement);
                
                // Resize to expected model input size
                const resized = tf.image.resizeBilinear(pixels, [640, 480]);
                
                // For MobileNet SSD models, no need to normalize to 0-1 range
                // Instead, keep as integers in the 0-255 range
                const expanded = resized.expandDims(0);
                
                // Cast to int32 as required by the error message
                return expanded.cast('int32');
            });
            
            // Run model inference
            let result;
            try {
                result = await this.model.executeAsync({'image_tensor': tensor});
            } catch (execError) {
                console.error("Model execution error:", execError);
                
                // Try alternate format for model input
                try {
                    console.log("Trying alternate model input format...");
                    // Clean up first tensor
                    tf.dispose(tensor);
                    
                    // Create new tensor with different format - try the normalized 0-1 range version
                    const altTensor = tf.tidy(() => {
                        const pixels = tf.browser.fromPixels(videoElement);
                        const resized = tf.image.resizeBilinear(pixels, [640, 480]);
                        const normalized = resized.div(255.0);
                        return normalized.expandDims(0);
                    });
                    
                    result = await this.model.executeAsync(altTensor);
                    tf.dispose(altTensor);
                } catch (altError) {
                    console.error("Alternate model format also failed:", altError);
                    return this.simulateTrafficLights();
                }
            }
            
            // Process results - handle both YOLO format and SSD MobileNet format
            let detections = [];
            let counts = {
                red: 0,
                yellow: 0,
                green: 0,
                unknown: 0
            };
            
            try {
                // First try SSD MobileNet format (our fallback model)
                const [boxes, scores, classes, valid_detections] = result;
                
                // Convert tensors to arrays for easier use
                const boxesArray = boxes.arraySync()[0];
                const scoresArray = scores.arraySync()[0];
                const classesArray = classes.arraySync()[0];
                const validDetections = valid_detections.arraySync()[0];
                
                // SSD MobileNet detects general objects, so we'll map some classes to traffic lights
                // Class 10 is traffic light in COCO dataset
                for (let i = 0; i < validDetections; i++) {
                    if (scoresArray[i] > 0.4) { // Confidence threshold
                        const classId = Math.floor(classesArray[i]);
                        let mappedClass = 0; // Default to red light
                        let confidence = scoresArray[i];
                        
                        // Class 10 in COCO is "traffic light"
                        if (classId === 10) {
                            // Simulate traffic light colors based on position
                            // Top third of the light is red, middle is yellow, bottom is green
                            const [y1, x1, y2, x2] = boxesArray[i];
                            const height = y2 - y1;
                            const centerY = y1 + (height / 2);
                            
                            if (centerY < 0.4) {
                                mappedClass = 0; // Red (top)
                                counts.red++;
                            } else if (centerY < 0.6) {
                                mappedClass = 1; // Yellow (middle)
                                counts.yellow++;
                            } else {
                                mappedClass = 2; // Green (bottom)
                                counts.green++;
                            }
                            
                            detections.push({
                                box: boxesArray[i], // [y1, x1, y2, x2] normalized coordinates
                                score: confidence,
                                class: mappedClass
                            });
                        } else if (classId === 1) { // Person (class 1 in COCO)
                            // Just for demonstration
                            counts.unknown++;
                            detections.push({
                                box: boxesArray[i],
                                score: confidence,
                                class: 3 // Unknown class
                            });
                        }
                    }
                }
            } catch (err) {
                console.warn("Error processing model results:", err);
                // If processing fails, fall back to simulation
                return this.simulateTrafficLights();
            } finally {
                // Clean up memory
                tf.dispose(result);
                tf.dispose(tensor);
            }
            
            // If no detections were found, return empty results instead of simulation
            return {
                detections,
                counts,
                simulated: false
            };
        } catch (err) {
            console.error("Error during traffic light detection:", err);
            // On any error, return simulated results
            return this.simulateTrafficLights();
        }
    }
    
    /**
     * Generate simulated traffic light detections
     * for when the model isn't available or errors occur
     */
    simulateTrafficLights() {
        // Instead of showing random positions for simulation,
        // let's make a more controlled demo with just occasional detections
        
        // Only show a light 20% of the time, to avoid constant false detections
        const shouldShowLight = Math.random() < 0.2;
        
        if (!shouldShowLight) {
            return {
                detections: [],
                counts: { red: 0, yellow: 0, green: 0, unknown: 0 },
                simulated: true
            };
        }
        
        // Show just one light for a more realistic demo
        const detections = [];
        const counts = { red: 0, yellow: 0, green: 0, unknown: 0 };
        
        // Fixed position in the upper right area (where traffic lights often are)
        const x1 = 0.7;
        const y1 = 0.2;
        const width = 0.06;
        const height = 0.15;
        
        // Random class (0=red, 1=yellow, 2=green) with preference for red
        const rand = Math.random();
        let classId;
        if (rand < 0.5) {
            classId = 0; // 50% red
            counts.red = 1;
        } else if (rand < 0.8) {
            classId = 1; // 30% yellow
            counts.yellow = 1;
        } else {
            classId = 2; // 20% green
            counts.green = 1;
        }
        
        detections.push({
            box: [y1, x1, y1 + height, x1 + width],
            score: 0.7 + (Math.random() * 0.2), // 0.7-0.9
            class: classId
        });
        
        return {
            detections,
            counts,
            simulated: true // Flag to indicate these are simulated results
        };
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
        
        // Clear the canvas (commenting this out since we're drawing on top of the video frame)
        // ctx.clearRect(0, 0, canvas.width, canvas.height);
        
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
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);
            
            // Draw label background
            ctx.fillStyle = color;
            ctx.font = '16px Arial';
            const textWidth = ctx.measureText(`${label} ${(detection.score * 100).toFixed(0)}%`).width;
            ctx.fillRect(x, y - 22, textWidth + 10, 22);
            
            // Draw label text
            ctx.fillStyle = 'black';
            ctx.fillText(`${label} ${(detection.score * 100).toFixed(0)}%`, x + 5, y - 5);
        });
        
        // Add a simulation indicator if these are simulated results
        if (detectionResult.simulated) {
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillRect(10, 10, 180, 30);
            ctx.fillStyle = 'white';
            ctx.font = '14px Arial';
            ctx.fillText('SIMULATION MODE (DEMO)', 20, 30);
        }
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
            error: this.error,
            simulation: this.useSimulation
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