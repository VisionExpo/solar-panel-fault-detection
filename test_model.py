import os
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path

# Add the deployment directory to the path
sys.path.append(str(Path(__file__).parent))
from deployment.inference import SolarPanelFaultDetector

def main():
    print("Testing model loading and inference...")
    
    # Create a detector instance
    model_path = "deployment/model"
    label_mapping_path = "deployment/label_mapping.json"
    
    print(f"Loading model from {model_path}...")
    detector = SolarPanelFaultDetector(model_path, label_mapping_path)
    print("Model loaded successfully!")
    
    # Create a random test image
    print("Creating test image...")
    test_image = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
    print(f"Test image shape: {test_image.shape}")
    
    # Run inference
    print("Running inference...")
    predictions = detector.predict(test_image)
    print("Inference completed successfully!")
    print(f"Predictions: {predictions}")
    
    # Get top prediction
    top_class, confidence = detector.classify_image(test_image)
    print(f"Top class: {top_class}, confidence: {confidence}")
    
    print("Test completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
