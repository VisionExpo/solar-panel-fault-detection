import os
import numpy as np
import cv2
import json
import tensorflow as tf
from typing import Dict, List, Tuple, Union, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SolarPanelFaultDetector:
    """
    A class for detecting faults in solar panels using a trained TensorFlow model.
    """

    def __init__(self, model_path: str, label_mapping_path: str):
        """
        Initialize the SolarPanelFaultDetector.

        Args:
            model_path: Path to the saved TensorFlow model
            label_mapping_path: Path to the JSON file containing label mapping
        """
        self.model_path = model_path
        self.label_mapping_path = label_mapping_path

        # Load the model
        logger.info(f"Loading model from {model_path}")
        self.model = self._load_model()

        # Load label mapping
        logger.info(f"Loading label mapping from {label_mapping_path}")
        self.label_mapping = self._load_label_mapping()

        # Get input shape from model
        self.input_shape = self.model.input_shape[1:3]  # (height, width)
        logger.info(f"Model input shape: {self.input_shape}")

    def _load_model(self) -> tf.keras.Model:
        """
        Load the TensorFlow model.

        Returns:
            The loaded TensorFlow model
        """
        try:
            model = tf.keras.models.load_model(self.model_path)
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def _load_label_mapping(self) -> Dict[str, int]:
        """
        Load the label mapping from a JSON file.

        Returns:
            A dictionary mapping class names to class indices
        """
        try:
            with open(self.label_mapping_path, 'r') as f:
                label_mapping = json.load(f)

            # Convert keys to strings if they are integers
            if all(k.isdigit() for k in label_mapping.keys()):
                label_mapping = {v: int(k) for k, v in label_mapping.items()}

            return label_mapping
        except Exception as e:
            logger.error(f"Error loading label mapping: {e}")
            raise

    def preprocess_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        """
        Preprocess an image for inference.

        Args:
            image: Path to an image file or a numpy array containing the image

        Returns:
            A preprocessed image as a numpy array
        """
        try:
            # Load image if it's a path
            if isinstance(image, str):
                logger.info(f"Loading image from path: {image}")
                img = cv2.imread(image)
                if img is None:
                    logger.error(f"Could not load image from {image}")
                    raise ValueError(f"Could not load image from {image}")
                logger.info(f"Image loaded with shape: {img.shape}, dtype: {img.dtype}")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                logger.info("Converted image from BGR to RGB")
            else:
                logger.info(f"Using provided image array with shape: {image.shape}, dtype: {image.dtype}")
                img = image.copy()

                # Check for NaN or infinity values
                if np.isnan(image).any() or np.isinf(image).any():
                    logger.error("Input image contains NaN or infinity values")
                    raise ValueError("Input image contains NaN or infinity values")

                # Convert BGR to RGB if needed
                if len(img.shape) == 3 and img.shape[2] == 3:
                    if isinstance(image, np.ndarray) and image.dtype == np.uint8:
                        logger.info("Converting image from BGR to RGB")
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Resize image to model input size
            logger.info(f"Resizing image from {img.shape} to {(self.input_shape[0], self.input_shape[1])}")
            img = cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
            logger.info(f"Resized image shape: {img.shape}")

            # Normalize pixel values to [0, 1]
            logger.info(f"Normalizing pixel values. Before: min={img.min()}, max={img.max()}, dtype={img.dtype}")
            img = img.astype(np.float32) / 255.0
            logger.info(f"After normalization: min={img.min()}, max={img.max()}, dtype={img.dtype}")

            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            logger.info(f"Final preprocessed image shape: {img.shape}")

            return img

        except Exception as e:
            import traceback
            logger.error(f"Error in preprocess_image: {e}")
            logger.error(traceback.format_exc())
            raise

    def predict(self, image: Union[str, np.ndarray]) -> Dict[str, float]:
        """
        Predict the class probabilities for an image.

        Args:
            image: Path to an image file or a numpy array containing the image

        Returns:
            A dictionary mapping class names to probabilities
        """
        try:
            # Preprocess the image
            logger.info("Preprocessing image for prediction")
            preprocessed_img = self.preprocess_image(image)

            # Make prediction
            logger.info("Running model prediction")
            predictions = self.model.predict(preprocessed_img, verbose=0)[0]
            logger.info(f"Raw prediction output shape: {predictions.shape}")
            logger.info(f"Raw prediction values: {predictions}")

            # Check for NaN or infinity values in predictions
            if np.isnan(predictions).any() or np.isinf(predictions).any():
                logger.error("Prediction contains NaN or infinity values")
                raise ValueError("Model produced invalid prediction values (NaN or infinity)")

            # Map indices to class names
            logger.info("Mapping prediction indices to class names")
            result = {}
            for class_name, idx in self.label_mapping.items():
                if idx < len(predictions):
                    result[class_name] = float(predictions[idx])
                else:
                    logger.error(f"Index {idx} for class {class_name} is out of bounds for predictions array of length {len(predictions)}")
                    raise IndexError(f"Class index {idx} is out of bounds for predictions array")

            logger.info(f"Final prediction result: {result}")
            return result

        except Exception as e:
            import traceback
            logger.error(f"Error in predict: {e}")
            logger.error(traceback.format_exc())
            raise

    def predict_batch(self, images: List[Union[str, np.ndarray]]) -> List[Dict[str, float]]:
        """
        Predict the class probabilities for a batch of images.

        Args:
            images: List of image paths or numpy arrays

        Returns:
            A list of dictionaries mapping class names to probabilities
        """
        # Preprocess all images
        preprocessed_imgs = np.vstack([self.preprocess_image(img) for img in images])

        # Make predictions
        predictions = self.model.predict(preprocessed_imgs)

        # Map indices to class names for each prediction
        results = []
        for pred in predictions:
            result = {}
            for class_name, idx in self.label_mapping.items():
                result[class_name] = float(pred[idx])
            results.append(result)

        return results

    def classify_image(self, image: Union[str, np.ndarray], threshold: float = 0.5, predictions: Dict[str, float] = None) -> Tuple[str, float]:
        """
        Classify an image and return the most likely class and confidence.

        Args:
            image: Path to an image file or a numpy array containing the image
            threshold: Confidence threshold for classification
            predictions: Optional pre-computed predictions to avoid redundant model calls

        Returns:
            A tuple containing the predicted class name and confidence
        """
        # Get predictions if not provided
        if predictions is None:
            predictions = self.predict(image)

        # Find the class with the highest probability
        top_class = max(predictions.items(), key=lambda x: x[1])
        class_name, confidence = top_class

        # If confidence is below threshold, return "Unknown"
        if confidence < threshold:
            return "Unknown", confidence

        return class_name, confidence

    def get_top_k_predictions(self, image: Union[str, np.ndarray], k: int = 3, predictions: Dict[str, float] = None) -> List[Tuple[str, float]]:
        """
        Get the top k predictions for an image.

        Args:
            image: Path to an image file or a numpy array containing the image
            k: Number of top predictions to return
            predictions: Optional pre-computed predictions to avoid redundant model calls

        Returns:
            A list of tuples containing class names and confidences
        """
        # Get predictions if not provided
        if predictions is None:
            predictions = self.predict(image)

        # Sort predictions by confidence (descending)
        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)

        # Return top k predictions
        return sorted_predictions[:k]


def main():
    """
    Example usage of the SolarPanelFaultDetector class.
    """
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect faults in solar panels')
    parser.add_argument('--model', type=str, default='model', help='Path to the saved model')
    parser.add_argument('--labels', type=str, default='label_mapping.json', help='Path to the label mapping file')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold')
    args = parser.parse_args()

    # Initialize detector
    detector = SolarPanelFaultDetector(args.model, args.labels)

    # Classify image
    class_name, confidence = detector.classify_image(args.image, args.threshold)
    print(f"Prediction: {class_name} (confidence: {confidence:.4f})")

    # Get top 3 predictions
    top_3 = detector.get_top_k_predictions(args.image, k=3)
    print("Top 3 predictions:")
    for class_name, confidence in top_3:
        print(f"  {class_name}: {confidence:.4f}")


if __name__ == "__main__":
    main()
