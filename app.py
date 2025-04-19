import os
from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from pathlib import Path
import json
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import io
from PIL import Image
import time

app = Flask(__name__)

# Constants
MODEL_PATH = Path("artifacts/models/serving")
LABEL_MAPPING_PATH = Path("artifacts/models/label_mapping.json")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Initialize thread pool for parallel processing
executor = ThreadPoolExecutor(max_workers=4)

# Load model and label mapping at startup
@lru_cache(maxsize=1)
def load_model():
    return tf.saved_model.load(str(MODEL_PATH))

@lru_cache(maxsize=1)
def load_label_mapping():
    with open(LABEL_MAPPING_PATH, 'r') as f:
        return json.load(f)

def preprocess_image(image_data):
    """Preprocess image for model inference"""
    try:
        # Convert bytes to image
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        
        # Convert to numpy array
        image = np.array(image)
        
        # Resize
        image = cv2.resize(image, IMG_SIZE)
        
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return image
    except Exception as e:
        app.logger.error(f"Error preprocessing image: {str(e)}")
        return None

def batch_predict(images):
    """Perform batch prediction"""
    try:
        model = load_model()
        predictions = model(tf.convert_to_tensor(images))
        return predictions.numpy()
    except Exception as e:
        app.logger.error(f"Error in batch prediction: {str(e)}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": time.time()})

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint for single image prediction"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read and preprocess image
        image_file = request.files['image']
        image_data = image_file.read()
        processed_image = preprocess_image(image_data)
        
        if processed_image is None:
            return jsonify({'error': 'Error processing image'}), 400
        
        # Add batch dimension
        image_batch = np.expand_dims(processed_image, axis=0)
        
        # Predict
        predictions = batch_predict(image_batch)
        if predictions is None:
            return jsonify({'error': 'Error making prediction'}), 500
            
        # Get label mapping
        label_mapping = load_label_mapping()
        
        # Process results
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        predicted_class = {v: k for k, v in label_mapping.items()}.get(class_idx)
        
        # Get top 3 predictions
        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {
                'class': {v: k for k, v in label_mapping.items()}.get(idx),
                'confidence': float(predictions[0][idx])
            }
            for idx in top_3_idx
        ]
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'top_3_predictions': top_3_predictions
        })
        
    except Exception as e:
        app.logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict_endpoint():
    """Endpoint for batch prediction"""
    if 'images' not in request.files:
        return jsonify({'error': 'No images provided'}), 400
    
    try:
        # Process multiple images
        images = request.files.getlist('images')
        if len(images) > BATCH_SIZE:
            return jsonify({'error': f'Maximum batch size is {BATCH_SIZE}'}), 400
        
        # Process images in parallel
        processed_images = []
        futures = []
        
        for image_file in images:
            image_data = image_file.read()
            future = executor.submit(preprocess_image, image_data)
            futures.append(future)
        
        # Collect results
        for future in futures:
            processed_image = future.result()
            if processed_image is not None:
                processed_images.append(processed_image)
        
        if not processed_images:
            return jsonify({'error': 'No valid images to process'}), 400
        
        # Convert to batch
        image_batch = np.stack(processed_images)
        
        # Predict
        predictions = batch_predict(image_batch)
        if predictions is None:
            return jsonify({'error': 'Error making predictions'}), 500
            
        # Get label mapping
        label_mapping = load_label_mapping()
        
        # Process results
        results = []
        for pred in predictions:
            class_idx = np.argmax(pred)
            confidence = float(pred[class_idx])
            predicted_class = {v: k for k, v in label_mapping.items()}.get(class_idx)
            
            # Get top 3 predictions
            top_3_idx = np.argsort(pred)[-3:][::-1]
            top_3_predictions = [
                {
                    'class': {v: k for k, v in label_mapping.items()}.get(idx),
                    'confidence': float(pred[idx])
                }
                for idx in top_3_idx
            ]
            
            results.append({
                'prediction': predicted_class,
                'confidence': confidence,
                'top_3_predictions': top_3_predictions
            })
        
        return jsonify({'results': results})
        
    except Exception as e:
        app.logger.error(f"Error processing batch request: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Load model at startup
    load_model()
    load_label_mapping()
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)