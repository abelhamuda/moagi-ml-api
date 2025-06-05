from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf
import os
import logging
from flask_cors import CORS
import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

try:
    logger.info("Attempting to load model: laundry_weight_model.keras")
    model = tf.keras.models.load_model('laundry_weight_model.keras')
    logger.info("Model loaded successfully")
    logger.info(f"Initial memory usage: {psutil.virtual_memory().percent}%")
    logger.info("Application ready")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Model loading failed: {str(e)}")

@app.route('/health', methods=['GET'])
def health():
    logger.info("Health check requested")
    logger.info(f"Memory usage during health check: {psutil.virtual_memory().percent}%")
    return jsonify({'status': 'healthy'}), 200

@app.route('/estimate_weight', methods=['POST'])
def estimate_weight():
    logger.info("Received request at /estimate_weight")
    logger.info(f"Memory usage before processing: {psutil.virtual_memory().percent}%")
    if 'image' not in request.files:
        logger.warning("No image provided in request")
        return jsonify({'error': 'No image provided', 'fallback': True}), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        logger.error("Failed to decode image")
        return jsonify({'error': 'Could not load image', 'fallback': True}), 400

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    try:
        logger.info("Predicting weight...")
        weight = model.predict(img)[0][0]
        weight = max(0.5, float(weight))
        logger.info(f"Predicted weight: {weight}")
        logger.info(f"Memory usage after prediction: {psutil.virtual_memory().percent}%")
        return jsonify({'weight': weight}), 200
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': str(e), 'fallback': True}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    logger.info(f"Starting Flask app on port {port}")
    app.run(host='0.0.0.0', port=port)