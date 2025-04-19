import os
import sys
from pathlib import Path
import random

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.config.configuration import Config
from src.solar_panel_detector.components.data_preparation import DataPreparation
from src.solar_panel_detector.utils.logger import logger
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import cv2

def main():
    try:
        # Initialize configuration
        config = Config()
        
        # Load label mapping
        with open("artifacts/models/label_mapping.json", 'r') as f:
            label_mapping = json.load(f)
            
        # Create reverse mapping (index to label)
        idx_to_label = {i: label for label, i in label_mapping.items()}
        
        # Load model
        logger.info("Loading model...")
        model_path = "artifacts/models/final_model"
        model = tf.keras.models.load_model(model_path)
        
        # Get test image paths
        test_dir = Path(config.data.test_data_path)
        test_images = []
        
        for class_dir in test_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                for img_path in class_dir.glob("*.jpg"):
                    test_images.append((str(img_path), class_name))
        
        # Randomly select images to visualize
        num_images = 10
        selected_images = random.sample(test_images, num_images)
        
        # Create data preparation instance for preprocessing
        data_preparation = DataPreparation(config)
        
        # Create figure for visualization
        plt.figure(figsize=(15, 20))
        
        for i, (img_path, true_class) in enumerate(selected_images):
            # Load and preprocess image
            img = data_preparation.load_and_preprocess_image(img_path)
            
            # Make prediction
            pred = model.predict(np.expand_dims(img, axis=0))[0]
            pred_class_idx = np.argmax(pred)
            pred_class = idx_to_label[pred_class_idx]
            confidence = pred[pred_class_idx]
            
            # Get top 3 predictions
            top3_idx = np.argsort(pred)[-3:][::-1]
            top3_classes = [idx_to_label[idx] for idx in top3_idx]
            top3_confidences = [pred[idx] for idx in top3_idx]
            
            # Load original image for display
            orig_img = cv2.imread(img_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            # Plot image with predictions
            plt.subplot(num_images, 2, 2*i+1)
            plt.imshow(orig_img)
            plt.title(f"True: {true_class}")
            plt.axis('off')
            
            # Plot prediction confidences
            plt.subplot(num_images, 2, 2*i+2)
            bars = plt.barh(top3_classes, top3_confidences, color=['green' if cls == true_class else 'red' for cls in top3_classes])
            plt.xlim(0, 1)
            plt.title(f"Prediction: {pred_class} ({confidence:.2f})")
            
            # Add confidence values to bars
            for bar, conf in zip(bars, top3_confidences):
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{conf:.2f}", 
                        va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig("artifacts/models/prediction_visualization.png")
        logger.info("Visualization saved to artifacts/models/prediction_visualization.png")
        
    except Exception as e:
        logger.error(f"Error in visualization: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
