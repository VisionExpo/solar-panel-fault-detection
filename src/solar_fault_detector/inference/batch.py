from pathlib import Path
from typing import List, Dict

import numpy as np
import tensorflow as tf

from solar_fault_detector.data.preprocessing import ImagePreprocessor
from solar_fault_detector.config.config import ModelConfig


class BatchInferenceEngine:
    """
    Runs batch inference on a directory of images.
    Intended for offline evaluation and bulk predictions.
    """

    def __init__(self, model_path: Path, config: ModelConfig):
        self.model = tf.keras.models.load_model(model_path)
        self.config = config
        self.preprocessor = ImagePreprocessor(config)

    def predict_images(self, image_paths: List[Path]) -> List[Dict]:
        """
        Run inference on a list of image paths.
        """
        batch = self.preprocessor.preprocess_batch(image_paths)
        predictions = self.model.predict(batch)

        results = []
        for path, probs in zip(image_paths, predictions):
            results.append({
                "image": path.name,
                "predicted_class": int(np.argmax(probs)),
                "confidence": float(np.max(probs)),
                "probabilities": probs.tolist(),
            })

        return results

    def predict_directory(self, image_dir: Path) -> List[Dict]:
        """
        Run inference on all images inside a directory.
        """
        image_paths = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        if not image_paths:
            raise ValueError(f"No images found in directory: {image_dir}")

        return self.predict_images(image_paths)
