import tensorflow as tf

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
        inputs = tf.keras.layers.Input(
            shape=(
                self.config.img_size[0],
                self.config.img_size[1],
                self.config.num_channels,
            )
        )

        x = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Conv2D(128, (3, 3), activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.MaxPooling2D()(x)

        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation="relu")(x)
        x = tf.keras.layers.Dropout(0.5)(x)

        outputs = tf.keras.layers.Dense(self.config.num_classes, activation="softmax")(
            x
        )

        self.model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
        return self.model
