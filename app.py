import os
from flask import Flask, request, jsonify, send_file
from flask_restx import Api, Resource, fields
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
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
from src.solar_panel_detector.components.monitoring import ModelMonitor
from src.solar_panel_detector.utils.logger import logger

app = Flask(__name__)
api = Api(app, version='1.0', 
         title='Solar Panel Fault Detection API',
         description='API for detecting faults in solar panels using computer vision',
         doc='/docs')

# Rate limiting configuration
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Constants
MODEL_PATH = Path("artifacts/models/serving")
LABEL_MAPPING_PATH = Path("artifacts/models/label_mapping.json")
IMG_SIZE = (224, 224)
BATCH_SIZE = int(os.getenv('MODEL_BATCH_SIZE', '32'))

# Initialize thread pool and monitoring
executor = ThreadPoolExecutor(max_workers=int(os.getenv('MODEL_NUM_WORKERS', '4')))
monitor = ModelMonitor()

# API models
prediction_model = api.model('Prediction', {
    'class': fields.String(description='The predicted class'),
    'confidence': fields.Float(description='Confidence score')
})

prediction_result = api.model('PredictionResult', {
    'prediction': fields.String(description='Primary prediction'),
    'confidence': fields.Float(description='Confidence score'),
    'inference_time_ms': fields.Float(description='Inference time in milliseconds'),
    'top_3_predictions': fields.List(fields.Nested(prediction_model))
})

batch_result = api.model('BatchResult', {
    'results': fields.List(fields.Nested(prediction_result)),
    'inference_time_ms': fields.Float(description='Total inference time in milliseconds'),
    'batch_size': fields.Integer(description='Number of images processed')
})

# Model optimization
@lru_cache(maxsize=1)
def load_model():
    # Enable mixed precision for faster inference
    tf.config.optimizer.set_jit(True)
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
    
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
        
        # Normalize and convert to float16 for mixed precision
        image = image.astype(np.float32) / 255.0
        image = tf.cast(image, tf.float16)
        
        return image
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def batch_predict(images):
    """Perform batch prediction with timing"""
    try:
        start_time = time.time()
        model = load_model()
        
        # Convert to mixed precision
        images = tf.cast(images, tf.float16)
        predictions = model(images)
        predictions = tf.cast(predictions, tf.float32)  # Convert back for post-processing
        
        inference_time = time.time() - start_time
        return predictions.numpy(), inference_time
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return None, 0

@api.route('/health')
class HealthCheck(Resource):
    @limiter.exempt
    def get(self):
        """Check service health"""
        report = monitor.generate_performance_report()
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "metrics": report['performance_metrics']
        }

@api.route('/predict')
class Predict(Resource):
    @api.doc(description='Detect faults in a single solar panel image')
    @api.expect(api.parser().add_argument('image', 
                                        type='file', 
                                        required=True,
                                        help='Solar panel image to analyze'))
    @api.response(200, 'Success', prediction_result)
    @api.response(400, 'Invalid input')
    @api.response(500, 'Internal server error')
    @limiter.limit("30 per minute")
    def post(self):
        """Endpoint for single image prediction"""
        if 'image' not in request.files:
            api.abort(400, 'No image provided')
        
        try:
            # Read and preprocess image
            image_file = request.files['image']
            image_data = image_file.read()
            processed_image = preprocess_image(image_data)
            
            if processed_image is None:
                api.abort(400, 'Error processing image')
            
            # Add batch dimension
            image_batch = np.expand_dims(processed_image, axis=0)
            
            # Predict with timing
            predictions, inference_time = batch_predict(image_batch)
            if predictions is None:
                api.abort(500, 'Error making prediction')
                
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
            
            # Log prediction
            monitor.log_prediction(
                prediction=predicted_class,
                confidence=confidence,
                inference_time=inference_time
            )
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'inference_time_ms': inference_time * 1000,
                'top_3_predictions': top_3_predictions
            }
            
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            api.abort(500, f'Internal server error: {str(e)}')

@api.route('/batch_predict')
class BatchPredict(Resource):
    @api.doc(description='Detect faults in multiple solar panel images')
    @api.expect(api.parser().add_argument('images', 
                                        type='file',
                                        required=True,
                                        action='append',
                                        help='Solar panel images to analyze'))
    @api.response(200, 'Success', batch_result)
    @api.response(400, 'Invalid input')
    @api.response(500, 'Internal server error')
    @limiter.limit("10 per minute")
    def post(self):
        """Endpoint for batch prediction"""
        if 'images' not in request.files:
            api.abort(400, 'No images provided')
        
        try:
            # Process multiple images
            images = request.files.getlist('images')
            if len(images) > BATCH_SIZE:
                api.abort(400, f'Maximum batch size is {BATCH_SIZE}')
            
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
                api.abort(400, 'No valid images to process')
            
            # Convert to batch
            image_batch = np.stack(processed_images)
            
            # Predict
            predictions, inference_time = batch_predict(image_batch)
            if predictions is None:
                api.abort(500, 'Error making predictions')
                
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
                
                # Log each prediction
                monitor.log_prediction(
                    prediction=predicted_class,
                    confidence=confidence,
                    inference_time=inference_time / len(predictions),
                    batch_size=len(predictions)
                )
            
            return {
                'results': results,
                'inference_time_ms': inference_time * 1000,
                'batch_size': len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Error processing batch request: {str(e)}")
            api.abort(500, f'Internal server error: {str(e)}')

@api.route('/metrics')
class Metrics(Resource):
    @api.doc(description='Get model performance metrics')
    @limiter.exempt
    def get(self):
        """Get current performance metrics"""
        return monitor.generate_performance_report()

@api.route('/dashboard')
class Dashboard(Resource):
    @api.doc(description='Get performance monitoring dashboard')
    @limiter.exempt
    def get(self):
        """Get the latest performance dashboard"""
        try:
            dashboard_dir = Path("artifacts/monitoring")
            dashboard_files = list(dashboard_dir.glob("performance_dashboard_*.html"))
            if not dashboard_files:
                api.abort(404, 'No dashboard available')
                
            latest_dashboard = max(dashboard_files, key=lambda x: x.stat().st_mtime)
            return send_file(latest_dashboard)
        except Exception as e:
            logger.error(f"Error serving dashboard: {str(e)}")
            api.abort(500, 'Error serving dashboard')

if __name__ == '__main__':
    # Load model at startup
    load_model()
    load_label_mapping()
    
    # Start Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)