from pathlib import Path
from typing import Dict

import numpy as np
import tensorflow as tf

from solar_fault_detector.data.preprocessing import ImagePreprocessor
from solar_fault_detector.config.config import ModelConfig


class Predictor:
    """
    Single-image inference engine.
    Used by APIs, Streamlit apps, and real-time inference.
    """

    def __init__(self, model_path: Path, config: ModelConfig):
        self.model = tf.keras.models.load_model(model_path)
        self.config = config
        self.preprocessor = ImagePreprocessor(config)

    def predict(self, image_path: Path) -> Dict:
        """
        Run inference on a single image.
        """
        image = self.preprocessor.load_and_preprocess(image_path)
        image = tf.expand_dims(image, axis=0)

        probs = self.model.predict(image)[0]

        return {
            "image": image_path.name,
            "predicted_class": int(np.argmax(probs)),
            "confidence": float(np.max(probs)),
            "probabilities": probs.tolist(),
        }
