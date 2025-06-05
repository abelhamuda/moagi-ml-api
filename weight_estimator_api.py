from flask import Flask, request, jsonify
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the model once at startup
model = tf.keras.models.load_model('laundry_weight_model.keras')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/estimate_weight', methods=['POST'])
def estimate_weight():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided', 'fallback': True}), 400

    file = request.files['image']
    img_array = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Could not load image', 'fallback': True}), 400

    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0) 

    try:
        weight = model.predict(img)[0][0]
        weight = max(0.5, float(weight))  # Ensure minimum weight
        return jsonify({'weight': weight}), 200
    except Exception as e:
        return jsonify({'error': str(e), 'fallback': True}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)