import os
import sys
import shutil
from pathlib import Path
import json
import tensorflow as tf

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.config.configuration import Config
from src.solar_panel_detector.utils.logger import logger

def main():
    try:
        # Initialize configuration
        config = Config()
        
        # Define deployment directory
        deployment_dir = Path("deployment")
        os.makedirs(deployment_dir, exist_ok=True)
        
        # Copy model to deployment directory
        model_path = Path("artifacts/models/final_model")
        deployment_model_path = deployment_dir / "model"
        
        if model_path.exists():
            # If model directory exists, copy it
            if deployment_model_path.exists():
                shutil.rmtree(deployment_model_path)
            shutil.copytree(model_path, deployment_model_path)
            logger.info(f"Model copied to {deployment_model_path}")
        else:
            logger.error(f"Model not found at {model_path}")
            return
        
        # Copy label mapping to deployment directory
        label_mapping_path = Path("artifacts/models/label_mapping.json")
        deployment_label_mapping_path = deployment_dir / "label_mapping.json"
        
        if label_mapping_path.exists():
            shutil.copy(label_mapping_path, deployment_label_mapping_path)
            logger.info(f"Label mapping copied to {deployment_label_mapping_path}")
        else:
            logger.error(f"Label mapping not found at {label_mapping_path}")
            return
        
        # Copy inference script to deployment directory
        inference_script_path = Path("src/solar_panel_detector/inference.py")
        deployment_inference_script_path = deployment_dir / "inference.py"
        
        if inference_script_path.exists():
            shutil.copy(inference_script_path, deployment_inference_script_path)
            logger.info(f"Inference script copied to {deployment_inference_script_path}")
        else:
            logger.error(f"Inference script not found at {inference_script_path}")
            return
        
        # Create a simple example script
        example_script = """
import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import json
from pathlib import Path
from inference import SolarPanelFaultDetector

def main():
    # Path to model and label mapping
    model_path = "model"
    label_mapping_path = "label_mapping.json"
    
    # Initialize detector
    detector = SolarPanelFaultDetector(model_path, label_mapping_path)
    
    # Example: Predict on a single image
    # Replace with your image path
    image_path = "example.jpg"
    
    if os.path.exists(image_path):
        # Get predictions
        predictions = detector.predict(image_path)
        print(f"Predictions for {image_path}:")
        for class_name, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {prob:.4f}")
        
        # Get top 3 predictions
        top_3 = detector.get_top_k_predictions(image_path, k=3)
        print("\\nTop 3 predictions:")
        for class_name, prob in top_3:
            print(f"  {class_name}: {prob:.4f}")
        
        # Classify image
        predicted_class, confidence = detector.classify_image(image_path)
        print(f"\\nClassification: {predicted_class} (confidence: {confidence:.4f})")
    else:
        print(f"Test image not found: {image_path}")
        print("Please provide a valid test image path.")

if __name__ == "__main__":
    main()
"""
        
        example_script_path = deployment_dir / "example.py"
        with open(example_script_path, 'w') as f:
            f.write(example_script)
        logger.info(f"Example script created at {example_script_path}")
        
        # Create a README file
        readme = """# Solar Panel Fault Detector

This is a deployment package for the Solar Panel Fault Detector model.

## Contents

- `model/`: The trained TensorFlow model
- `label_mapping.json`: Mapping of class indices to class names
- `inference.py`: Inference script with the SolarPanelFaultDetector class
- `example.py`: Example script showing how to use the model

## Usage

1. Install the required dependencies:
   ```
   pip install tensorflow opencv-python numpy
   ```

2. Place your solar panel image in this directory.

3. Run the example script:
   ```
   python example.py
   ```

4. Modify the example script to use your own image path.

## Model Information

This model can detect the following types of solar panel conditions:
- Bird-drop
- Clean
- Dusty
- Electrical-damage
- Physical-damage
- Snow-covered

## API

The `SolarPanelFaultDetector` class provides the following methods:

- `predict(image)`: Returns a dictionary of class probabilities
- `predict_batch(images)`: Predicts on a batch of images
- `classify_image(image, threshold=0.5)`: Returns the most likely class and confidence
- `get_top_k_predictions(image, k=3)`: Returns the top k predictions
"""
        
        readme_path = deployment_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme)
        logger.info(f"README created at {readme_path}")
        
        # Create a requirements.txt file
        requirements = """tensorflow>=2.8.0
numpy>=1.19.5
opencv-python>=4.5.5
"""
        
        requirements_path = deployment_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(requirements)
        logger.info(f"Requirements file created at {requirements_path}")
        
        logger.info("Deployment package created successfully")
        
    except Exception as e:
        logger.error(f"Error in deployment: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
