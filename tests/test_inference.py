from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
from PIL import Image

from solar_fault_detector.config.config import ModelConfig
from solar_fault_detector.inference.predictor import Predictor
from solar_fault_detector.inference.batch import BatchInferenceEngine


def _create_dummy_image(path: Path):
    img = Image.fromarray(
        np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    )
    img.save(path)


@patch("tensorflow.keras.models.load_model")
def test_single_image_predictor(mock_load_model, tmp_path):
    # Mock model
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([[0.1, 0.7, 0.2]])
    mock_load_model.return_value = mock_model

    img_path = tmp_path / "test.jpg"
    _create_dummy_image(img_path)

    config = ModelConfig(num_classes=3, img_size=(64, 64))
    predictor = Predictor(
        model_path=Path("dummy_model"),
        config=config,
    )

    result = predictor.predict(img_path)

    assert "predicted_class" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert result["predicted_class"] == 1


@patch("tensorflow.keras.models.load_model")
def test_batch_inference_engine(mock_load_model, tmp_path):
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array(
        [
            [0.2, 0.5, 0.3],
            [0.6, 0.2, 0.2],
        ]
    )
    mock_load_model.return_value = mock_model

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    for i in range(2):
        _create_dummy_image(image_dir / f"img_{i}.jpg")

    config = ModelConfig(num_classes=3, img_size=(64, 64))
    engine = BatchInferenceEngine(
        model_path=Path("dummy_model"),
        config=config,
    )

    results = engine.predict_directory(image_dir)

    assert len(results) == 2
    assert all("predicted_class" in r for r in results)
    assert all("confidence" in r for r in results)
