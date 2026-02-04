import tensorflow as tf
from tensorflow.keras import layers, models

from solar_fault_detector.models.base import BaseModel
from solar_fault_detector.config.config import ModelConfig


class CNNModel(BaseModel):
    """
    Convolutional Neural Network for solar panel fault classification.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)

    def build(self) -> tf.keras.Model:
        """
        Build CNN architecture.
        """
        inputs = layers.Input(
            shape=(
                self.config.img_size[0],
                self.config.img_size[1],
                self.config.num_channels,
            )
        )

        x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Conv2D(128, (3, 3), activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)

        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(
            self.config.num_classes, activation="softmax"
        )(x)

        self.model = models.Model(inputs=inputs, outputs=outputs)
        return self.model
