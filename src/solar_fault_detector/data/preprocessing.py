import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image


class ImagePreprocessor:
    def __init__(self, config=None):
        self.config = config
        self.target_size = (224, 224)
        if config and hasattr(config, 'img_size'):
            self.target_size = config.img_size

    def load_and_preprocess(self, image_path: Path) -> tf.Tensor:
        with Image.open(image_path) as img:
            image = img.convert("RGB")
            image = image.resize(self.target_size)
        image_array = np.array(image) / 255.0
        return tf.convert_to_tensor(image_array, dtype=tf.float32)

    def preprocess_batch(self, image_paths: list[Path]) -> tf.Tensor:
        arrays = []
        for p in image_paths:
            # Add batch dimension to allow concatenation
            arr = np.expand_dims(self.load_and_preprocess(p).numpy(), axis=0)
            arrays.append(arr)

        concatenated = np.concatenate(arrays, axis=0)
        del arrays
        return tf.convert_to_tensor(concatenated)
