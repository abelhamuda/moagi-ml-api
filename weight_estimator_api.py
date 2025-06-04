from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model variable
model = None

def load_model():
    """Load the TensorFlow model"""
    global model
    try:
        # Try to import TensorFlow
        import tensorflow as tf
        
        model_path = os.getenv('MODEL_PATH', 'laundry_weight_model.keras')
        
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            logger.info(f"Model loaded successfully from {model_path}")
            return True
        else:
            logger.error(f"Model file not found: {model_path}")
            return False
            
    except ImportError:
        logger.error("TensorFlow not available")
        return False
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False

# Try to load model at startup
model_loaded = load_model()

@app.route('/', methods=['GET'])
def root():
    """Root endpoint with API information"""
    return jsonify({
        'service': 'Laundry Weight Estimation API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model_loaded,
        'endpoints': {
            'health': '/health',
            'estimate_weight': '/estimate_weight (POST with image file)'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_status': 'loaded' if model_loaded else 'not_loaded'
    }), 200

@app.route('/estimate_weight', methods=['POST'])
def estimate_weight():
    """Estimate laundry weight from image"""
    
    # Check if model is loaded
    if not model_loaded or model is None:
        logger.error("Model not available for prediction")
        return jsonify({
            'error': 'Model not loaded', 
            'fallback': True
        }), 500

    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image provided. Please upload an image file.', 
            'fallback': True
        }), 400

    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'error': 'No image selected', 
            'fallback': True
        }), 400

    try:
        # Read and decode image
        logger.info(f"Processing image: {file.filename}")
        img_bytes = file.read()
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.error("Could not decode image")
            return jsonify({
                'error': 'Could not decode image. Please ensure it\'s a valid image file.', 
                'fallback': True
            }), 400

        # Log original image dimensions
        logger.info(f"Original image shape: {img.shape}")

        # Preprocess image for model
        img_resized = cv2.resize(img, (224, 224))
        img_normalized = img_resized / 255.0
        img_batch = np.expand_dims(img_normalized, axis=0)

        logger.info("Image preprocessed successfully")

        # Make prediction
        prediction = model.predict(img_batch, verbose=0)
        weight = float(prediction[0][0])
        
        # Ensure reasonable weight bounds
        weight = max(0.5, min(weight, 50.0))  # Between 0.5kg and 50kg

        logger.info(f"Weight prediction: {weight:.2f} kg")
        
        return jsonify({
            'weight': round(weight, 2),
            'unit': 'kg',
            'success': True,
            'message': f'Estimated laundry weight: {weight:.2f} kg'
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}', 
            'fallback': True
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'error': 'Image file too large. Maximum size is 16MB.',
        'fallback': True
    }), 413

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/', '/health', '/estimate_weight']
    }), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        'error': 'Internal server error',
        'fallback': True
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Model loaded: {model_loaded}")
    
    app.run(host='0.0.0.0', port=port, debug=debug)