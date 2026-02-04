# tests/test_model.py

import tensorflow as tf

from solar_fault_detector.config.config import ModelConfig
from solar_fault_detector.models.cnn import CNNModel
from solar_fault_detector.models.ensemble import EnsembleModel


def test_cnn_model_build():
    config = ModelConfig(
        img_size=(64, 64),
        num_channels=3,
        num_classes=4,
        batch_size=2,
    )

    model_wrapper = CNNModel(config)
    model = model_wrapper.build()

    assert isinstance(model, tf.keras.Model)
    assert model.output_shape[-1] == config.num_classes

    # Forward pass sanity check
    dummy_input = tf.random.uniform(
        shape=(1, 64, 64, 3)
    )
    output = model(dummy_input)
    assert output.shape == (1, config.num_classes)


def test_ensemble_model_build():
    config = ModelConfig(
        img_size=(64, 64),
        num_channels=3,
        num_classes=3,
        batch_size=2,
    )

    ensemble = EnsembleModel(config, num_models=2)
    model = ensemble.build()

    assert isinstance(model, tf.keras.Model)
    assert model.output_shape[-1] == config.num_classes

    # Forward pass sanity check
    dummy_input = tf.random.uniform(
        shape=(1, 64, 64, 3)
    )
    output = model(dummy_input)
    assert output.shape == (1, config.num_classes)
