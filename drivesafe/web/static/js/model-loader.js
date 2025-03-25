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
            
            // Create a dummy tensor with correct format (int32)
            const dummyTensor = tf.tidy(() => {
                // Create zeros tensor with correct shape
                const zeros = tf.zeros([1, 640, 480, 3]);
                // Convert to int32 which is what the model expects
                return tf.cast(zeros, 'int32');
            });
            
            // Run inference on the dummy tensor with correct input format
            const result = await this.model.executeAsync({'image_tensor': dummyTensor});
            
            // Clean up memory
            tf.dispose([dummyTensor, ...(Array.isArray(result) ? result : [result])]);
            
            console.log("Model warmup complete");
        } catch (err) {
            console.error("Error during model warmup:", err);
            // Continue even if warmup fails - we'll handle format in actual detection
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
            
            // Run model inference with extensive error handling
            let result;
            try {
                // First attempt: Use the image_tensor format commonly used by TF models
                result = await this.model.executeAsync({'image_tensor': tensor});
                console.log("Model execution successful with image_tensor format");
            } catch (execError) {
                console.error("Model execution error with image_tensor format:", execError);
                
                // Second attempt: Try passing the tensor directly
                try {
                    console.log("Trying direct tensor input...");
                    result = await this.model.executeAsync(tensor);
                    console.log("Model execution successful with direct tensor input");
                } catch (directError) {
                    console.error("Model execution error with direct tensor input:", directError);
                    
                    // Third attempt: Try with normalized tensor (0-1 range)
                    try {
                        console.log("Trying normalized tensor input...");
                        // Clean up first tensor
                        tf.dispose(tensor);
                        
                        const normalizedTensor = tf.tidy(() => {
                            const pixels = tf.browser.fromPixels(videoElement);
                            const resized = tf.image.resizeBilinear(pixels, [640, 480]);
                            const normalized = resized.div(255.0);
                            return normalized.expandDims(0);
                        });
                        
                        result = await this.model.executeAsync(normalizedTensor);
                        console.log("Model execution successful with normalized tensor");
                        tf.dispose(normalizedTensor);
                    } catch (normalizedError) {
                        console.error("All model execution attempts failed:", normalizedError);
                        return this.simulateTrafficLights();
                    }
                }
            }
            
            // Process results - handle different possible output formats
            let detections = [];
            let counts = {
                red: 0,
                yellow: 0,
                green: 0,
                unknown: 0
            };
            
            try {
                // Log what we received to help with debugging
                console.log("Model output format:", 
                    Array.isArray(result) ? `Array of ${result.length} tensors` : typeof result);
                
                // Handle different possible result formats
                if (Array.isArray(result)) {
                    // Case 1: Standard TF.js detection model output [boxes, scores, classes, numDetections]
                    if (result.length >= 4) {
                        const boxesTensor = result[0];
                        const scoresTensor = result[1];
                        const classesTensor = result[2];
                        const numDetectionsTensor = result[3];
                        
                        // Safely extract tensor data if available
                        if (boxesTensor && scoresTensor && classesTensor && numDetectionsTensor) {
                            // Check if these are actually tensors with arraySync method
                            if (typeof boxesTensor.arraySync === 'function' &&
                                typeof scoresTensor.arraySync === 'function' &&
                                typeof classesTensor.arraySync === 'function' &&
                                typeof numDetectionsTensor.arraySync === 'function') {
                                
                                try {
                                    // Extract data with extra validation
                                    const boxesData = boxesTensor.arraySync();
                                    const scoresData = scoresTensor.arraySync();
                                    const classesData = classesTensor.arraySync();
                                    const numDetections = numDetectionsTensor.arraySync()[0];  // Should be a scalar
                                    
                                    if (boxesData && boxesData[0] && scoresData && scoresData[0] && 
                                        classesData && classesData[0] && 
                                        typeof numDetections === 'number') {
                                        
                                        const boxesArray = boxesData[0];
                                        const scoresArray = scoresData[0];
                                        const classesArray = classesData[0];
                                        
                                        // Process detections if we have valid data
                                        this.processDetections(
                                            boxesArray, 
                                            scoresArray, 
                                            classesArray, 
                                            numDetections, 
                                            detections, 
                                            counts
                                        );
                                    } else {
                                        console.warn("Invalid array data in tensors");
                                    }
                                } catch (dataError) {
                                    console.error("Error extracting tensor data:", dataError);
                                }
                            } else {
                                console.warn("One or more tensors missing arraySync method");
                            }
                        } else {
                            console.warn("One or more output tensors is null/undefined");
                        }
                    } else {
                        console.warn("Unexpected result format: Array with fewer than 4 elements");
                    }
                } else if (result && typeof result === 'object') {
                    // Case 2: Object format (some models output named properties)
                    console.log("Object result format, properties:", Object.keys(result));
                    
                    // Here we could handle other known output formats
                    // This is a placeholder for future expansion
                }
            } catch (err) {
                console.warn("Error processing model results:", err);
                // If processing fails, fall back to simulation
                return this.simulateTrafficLights();
            } finally {
                // Clean up memory - safely dispose all tensors
                if (Array.isArray(result)) {
                    result.forEach(tensor => {
                        if (tensor && typeof tensor.dispose === 'function') {
                            tensor.dispose();
                        }
                    });
                } else if (result && typeof result.dispose === 'function') {
                    result.dispose();
                }
                
                if (tensor && typeof tensor.dispose === 'function') {
                    tensor.dispose();
                }
            }
            
            // Return results or simulation if nothing was detected
            if (detections.length === 0) {
                console.log("No detections found, returning empty results");
                // If nothing detected, return empty results
                return {
                    detections: [],
                    counts: { red: 0, yellow: 0, green: 0, unknown: 0 },
                    simulated: false
                };
            }
            
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
     * Process detection data from model output tensors
     */
    processDetections(boxes, scores, classes, numDetections, detections, counts) {
        // Process up to the number of valid detections
        const validDetections = Math.min(numDetections, boxes.length);
        
        // The Syazvinski model is specifically trained for traffic lights with color classification
        // It uses the following classes: 0=red_light, 1=yellow_light, 2=green_light
        // This is different from SSD MobileNet which uses COCO classes (10=traffic light)
        console.log("Processing detections, found:", validDetections);
        
        for (let i = 0; i < validDetections; i++) {
            if (scores[i] > 0.3) { // Lower threshold to detect more traffic lights
                const classId = Math.floor(classes[i]);
                let mappedClass = classId; // Use the actual class from model
                let confidence = scores[i];
                
                console.log("Detected object with class:", classId, "confidence:", confidence);
                
                // Check if this is a traffic light class
                // Either direct from Syazvinski model (0=red, 1=yellow, 2=green)
                // Or from COCO dataset (10=traffic light)
                if (classId === 0 || classId === 1 || classId === 2) {
                    // Direct traffic light classes from the specialized model
                    if (classId === 0) {
                        counts.red++;
                        console.log("Detected RED traffic light");
                    } else if (classId === 1) {
                        counts.yellow++;
                        console.log("Detected YELLOW traffic light");
                    } else if (classId === 2) {
                        counts.green++;
                        console.log("Detected GREEN traffic light");
                    }
                    
                    detections.push({
                        box: boxes[i], // [y1, x1, y2, x2] normalized coordinates
                        score: confidence,
                        class: classId // Use original class ID
                    });
                } else if (classId === 10) {
                    // Handle generic traffic light from COCO dataset
                    // Determine color based on position in the bounding box
                    const [y1, x1, y2, x2] = boxes[i];
                    const height = y2 - y1;
                    const centerY = y1 + (height / 2);
                    
                    if (centerY < 0.4) {
                        counts.red++;
                        mappedClass = 0; // Red
                    } else if (centerY < 0.6) {
                        counts.yellow++;
                        mappedClass = 1; // Yellow
                    } else {
                        counts.green++;
                        mappedClass = 2; // Green
                    }
                    
                    detections.push({
                        box: boxes[i],
                        score: confidence,
                        class: mappedClass
                    });
                } else if (classId === 1 && confidence > 0.5) {
                    // Person detection - uncomment if needed
                    // counts.unknown++;
                    // detections.push({
                    //     box: boxes[i],
                    //     score: confidence,
                    //     class: 3 // Unknown class
                    // });
                }
            }
        }
        
        // Log detection counts
        console.log("Detection counts:", counts);
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