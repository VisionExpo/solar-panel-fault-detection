import pytest
import tensorflow as tf
import numpy as np
from pathlib import Path
from src.solar_panel_detector.components.model import SolarPanelModel
from src.solar_panel_detector.config.configuration import Config
import shutil
import os

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def test_data():
    # Create synthetic test data
    images = np.random.rand(10, 224, 224, 3).astype(np.float32)
    labels = np.random.randint(0, 6, size=10)
    dataset = tf.data.Dataset.from_tensor_slices((images, labels)).batch(2)
    return dataset

@pytest.fixture
def label_mapping():
    return {
        'Bird-drop': 0,
        'Clean': 1,
        'Dusty': 2,
        'Electrical-damage': 3,
        'Physical-Damage': 4,
        'Snow-Covered': 5
    }

def test_model_creation(config):
    model = SolarPanelModel(config)
    assert model.model is not None
    assert len(model.model.layers) > 0

def test_model_training(config, test_data, label_mapping):
    model = SolarPanelModel(config)
    model.train(test_data, test_data, label_mapping)
    assert model.history is not None
    assert 'loss' in model.history.history
    assert 'accuracy' in model.history.history

def test_model_evaluation(config, test_data, label_mapping):
    model = SolarPanelModel(config)
    report = model.evaluate(test_data, label_mapping)
    assert isinstance(report, dict)
    assert 'accuracy' in report

def test_model_save_load(config, tmp_path):
    original_model = SolarPanelModel(config)
    serving_path = original_model.save_model_for_serving()
    assert serving_path.exists()

    # Load saved model
    loaded_model = tf.saved_model.load(str(serving_path))
    assert loaded_model is not None

def test_model_prediction(config):
    model = SolarPanelModel(config)
    test_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
    prediction = model.model.predict(test_input)
    assert prediction.shape == (1, 6)  # 6 classes
    assert np.all(prediction >= 0) and np.all(prediction <= 1)  # Valid probabilities