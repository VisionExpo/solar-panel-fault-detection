from pathlib import Path
from typing import List, Dict, Union

from solar_fault_detector.config.config import Config
from solar_fault_detector.inference.predictor import Predictor
from solar_fault_detector.inference.batch import BatchInferenceEngine


def run_single_inference(
    image_path: Path,
    model_path: Path | None = None,
) -> Dict:
    """
    Run inference on a single image.
    """
    config = Config()
    model_path = model_path or config.model.best_model_path

    predictor = Predictor(
        model_path=model_path,
        config=config.model,
    )

    return predictor.predict(image_path)


def run_batch_inference(
    image_dir: Path,
    model_path: Path | None = None,
) -> List[Dict]:
    """
    Run batch inference on a directory of images.
    """
    config = Config()
    model_path = model_path or config.model.best_model_path

    engine = BatchInferenceEngine(
        model_path=model_path,
        config=config.model,
    )

    return engine.predict_directory(image_dir)


if __name__ == "__main__":
    raise RuntimeError(
        "This module is intended to be imported and used by "
        "scripts, CI pipelines, or deployment jobs."
    )
