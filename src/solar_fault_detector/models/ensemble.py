from typing import List
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from solar_fault_detector.models.base import BaseModel
from solar_fault_detector.config.config import ModelConfig
from solar_fault_detector.models.cnn import CNNModel


class EnsembleModel(BaseModel):
    """
    Ensemble of multiple CNN models with probability averaging.
    """

    def __init__(self, config: ModelConfig, num_models: int = 3):
        super().__init__(config)
        self.num_models = num_models
        self.sub_models: List[tf.keras.Model] = []

    def build(self) -> tf.keras.Model:
        """
        Build ensemble by instantiating multiple CNN models.
        """
        inputs = layers.Input(
            shape=(
                self.config.img_size[0],
                self.config.img_size[1],
                self.config.num_channels,
            )
        )

        outputs = []
        for i in range(self.num_models):
            cnn = CNNModel(self.config)
            sub_model = cnn.build()
            self.sub_models.append(sub_model)
            outputs.append(sub_model(inputs))

        avg_output = layers.Average()(outputs)

        self.model = models.Model(inputs=inputs, outputs=avg_output)
        return self.model

    def save(self, path: Path) -> None:
        """
        Save ensemble model.
        """
        if self.model is None:
            raise ValueError("No ensemble model available to save.")

        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
