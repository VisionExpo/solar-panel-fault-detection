import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import json
from pathlib import Path
from typing import Dict, List, Tuple, Union

class SolarPanelFaultDetector:
    """
    Class for detecting faults in solar panels using the trained model.
    """
    
    def __init__(self, model_path: str, label_mapping_path: str = None):
        """
        Initialize the detector with a trained model.
        
        Args:
            model_path (str): Path to the trained model
            label_mapping_path (str, optional): Path to the label mapping JSON file
        """
        self.model_path = model_path
        self.model = self._load_model()
        
        # Load label mapping if provided
        if label_mapping_path:
            with open(label_mapping_path, 'r') as f:
                self.label_mapping = json.load(f)
        else:
            # Default label mapping
            self.label_mapping = {
                "Bird-drop": 0,
                "Clean": 1,
                "Dusty": 2,
                "Electrical-damage": 3,
                "Physical-damage": 4,
                "Snow-covered": 5
            }
            
        # Create reverse mapping (index to label)
        self.idx_to_label = {v: k for k, v in self.label_mapping.items()}
        
        # Get model input shape
        self.input_shape = self.model.input_shape[1:3]  # (height, width)
    
    def _load_model(self) -> tf.keras.Model:
        """
        Load the trained model.
        
        Returns:
            tf.keras.Model: Loaded model
        """
        return tf.keras.models.load_model(self.model_path)
    
    def preprocess_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess an image for model input.
        
        Args:
            image (str or np.ndarray): Image path or image array
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image)
            if img is None:
                raise ValueError(f"Failed to load image from {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image.copy()
            if len(img.shape) == 2:  # Grayscale
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
        # Resize image to model input size
        img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize pixel values
        img = img.astype(np.float32)
        img = img / 255.0
        
        # Apply ImageNet normalization
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img = (img - mean) / std
        
        return img
    
    def predict(self, image: Union[str, np.ndarray]) -> Dict[str, float]:
        """
        Predict the fault type for a single image.
        
        Args:
            image (str or np.ndarray): Image path or image array
            
        Returns:
            Dict[str, float]: Dictionary mapping fault types to probabilities
        """
        # Preprocess image
        img = self.preprocess_image(image)
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        # Make prediction
        predictions = self.model.predict(img)[0]
        
        # Create dictionary of class probabilities
        result = {self.idx_to_label[i]: float(prob) for i, prob in enumerate(predictions)}
        
        return result
    
    def predict_batch(self, images: List[Union[str, np.ndarray]]) -> List[Dict[str, float]]:
        """
        Predict fault types for a batch of images.
        
        Args:
            images (List[str or np.ndarray]): List of image paths or image arrays
            
        Returns:
            List[Dict[str, float]]: List of dictionaries mapping fault types to probabilities
        """
        # Preprocess images
        processed_images = []
        for img in images:
            processed = self.preprocess_image(img)
            processed_images.append(processed)
        
        # Stack images into a batch
        batch = np.stack(processed_images)
        
        # Make predictions
        predictions = self.model.predict(batch)
        
        # Create list of dictionaries with class probabilities
        results = []
        for pred in predictions:
            result = {self.idx_to_label[i]: float(prob) for i, prob in enumerate(pred)}
            results.append(result)
        
        return results
    
    def classify_image(self, image: Union[str, np.ndarray], threshold: float = 0.5) -> Tuple[str, float]:
        """
        Classify an image and return the most likely fault type.
        
        Args:
            image (str or np.ndarray): Image path or image array
            threshold (float, optional): Confidence threshold for classification
            
        Returns:
            Tuple[str, float]: Predicted class and confidence
        """
        # Get predictions
        predictions = self.predict(image)
        
        # Find class with highest probability
        predicted_class = max(predictions, key=predictions.get)
        confidence = predictions[predicted_class]
        
        # Apply threshold
        if confidence < threshold:
            return "Unknown", confidence
        
        return predicted_class, confidence
    
    def get_top_k_predictions(self, image: Union[str, np.ndarray], k: int = 3) -> List[Tuple[str, float]]:
        """
        Get the top k predictions for an image.
        
        Args:
            image (str or np.ndarray): Image path or image array
            k (int, optional): Number of top predictions to return
            
        Returns:
            List[Tuple[str, float]]: List of (class, probability) tuples
        """
        # Get predictions
        predictions = self.predict(image)
        
        # Sort predictions by probability (descending)
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k predictions
        return sorted_predictions[:k]


def main():
    """
    Example usage of the SolarPanelFaultDetector class.
    """
    # Path to model and label mapping
    model_path = "artifacts/models/final_model"
    label_mapping_path = "artifacts/models/label_mapping.json"
    
    # Initialize detector
    detector = SolarPanelFaultDetector(model_path, label_mapping_path)
    
    # Example: Predict on a single image
    test_image_path = "data/test/Clean/clean_panel_001.jpg"  # Replace with an actual test image path
    
    if os.path.exists(test_image_path):
        # Get predictions
        predictions = detector.predict(test_image_path)
        print(f"Predictions for {test_image_path}:")
        for class_name, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
            print(f"  {class_name}: {prob:.4f}")
        
        # Get top 3 predictions
        top_3 = detector.get_top_k_predictions(test_image_path, k=3)
        print("\nTop 3 predictions:")
        for class_name, prob in top_3:
            print(f"  {class_name}: {prob:.4f}")
        
        # Classify image
        predicted_class, confidence = detector.classify_image(test_image_path)
        print(f"\nClassification: {predicted_class} (confidence: {confidence:.4f})")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please provide a valid test image path.")

if __name__ == "__main__":
    main()
