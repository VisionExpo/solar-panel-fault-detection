from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import tensorflow as tf

from solar_fault_detector.config.config import ModelConfig


class BaseModel(ABC):
    """
    Abstract base class for all models.
    Enforces a consistent interface across architectures.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.model: tf.keras.Model | None = None

    @abstractmethod
    def build(self) -> tf.keras.Model:
        """
        Build and return the Keras model architecture.
        """
        raise NotImplementedError

    def compile(self) -> None:
        """
        Compile the model with optimizer, loss, and metrics.
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation.")

        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.config.learning_rate
        )

        self.model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )

    def train(self, train_data, val_data, callbacks=None) -> Any:
        """
        Train the model.
        """
        if self.model is None:
            raise ValueError("Model must be built before training.")

        return self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=self.config.epochs,
            callbacks=callbacks
        )

    def save(self, path: Path) -> None:
        """
        Save the trained model.
        """
        if self.model is None:
            raise ValueError("No model available to save.")

        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
