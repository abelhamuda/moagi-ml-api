from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
from flask.logging import create_logger

app = Flask(__name__)
logger = create_logger(app)

# Load the model once at startup
try:
    model_path = os.getenv('MODEL_PATH', 'laundry_weight_model.keras')
    model = tf.keras.models.load_model(model_path)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    model = None

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'Laundry Weight Estimation API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None
    }), 200

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_status': 'loaded' if model is not None else 'error'
    }), 200

@app.route('/estimate_weight', methods=['POST'])
def estimate_weight():
    # Check if model is loaded
    if model is None:
        return jsonify({
            'error': 'Model not loaded', 
            'fallback': True
        }), 500

    # Check if image is provided
    if 'image' not in request.files:
        return jsonify({
            'error': 'No image provided', 
            'fallback': True
        }), 400

    file = request.files['image']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({
            'error': 'No image selected', 
            'fallback': True
        }), 400

    try:
        # Read and decode image
        img_array = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({
                'error': 'Could not decode image', 
                'fallback': True
            }), 400

        # Preprocess image
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make prediction
        prediction = model.predict(img, verbose=0)
        weight = float(prediction[0][0])
        weight = max(0.5, weight)  # Ensure minimum weight

        logger.info(f"Weight prediction: {weight}")
        
        return jsonify({
            'weight': weight,
            'unit': 'kg',
            'success': True
        }), 200

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}', 
            'fallback': True
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({
        'error': 'File too large',
        'fallback': True
    }), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({
        'error': 'Internal server error',
        'fallback': True
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    app.run(host='0.0.0.0', port=port, debug=debug)