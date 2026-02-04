import shutil
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

from solar_fault_detector.config.config import DataConfig, ModelConfig
from solar_fault_detector.data.ingestion import DataIngestor
from solar_fault_detector.data.preprocessing import ImagePreprocessor


def _create_dummy_dataset(root: Path):
    """
    Create a small dummy image dataset with class folders.
    """
    classes = ["class_a", "class_b"]
    for cls in classes:
        cls_dir = root / cls
        cls_dir.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(cls_dir / f"{cls}_{i}.jpg")


def test_data_ingestion_creates_splits(tmp_path):
    source_dir = tmp_path / "dataset"
    _create_dummy_dataset(source_dir)

    config = DataConfig(
        root_dir=tmp_path / "artifacts",
        data_dir=source_dir,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=42,
    )

    ingestor = DataIngestor(config)
    splits = ingestor.ingest()

    assert (splits["train"]).exists()
    assert (splits["val"]).exists()
    assert (splits["test"]).exists()

    # Ensure class folders exist in each split
    for split in splits.values():
        assert any(split.iterdir())


def test_image_preprocessing(tmp_path):
    img_path = tmp_path / "test.jpg"
    img = Image.fromarray(
        np.random.randint(0, 255, (80, 80, 3), dtype=np.uint8)
    )
    img.save(img_path)

    model_config = ModelConfig(img_size=(64, 64))
    preprocessor = ImagePreprocessor(model_config)

    tensor = preprocessor.load_and_preprocess(img_path)

    assert isinstance(tensor, tf.Tensor)
    assert tensor.shape == (64, 64, 3)
    assert tf.reduce_max(tensor).numpy() <= 1.0
    assert tf.reduce_min(tensor).numpy() >= 0.0