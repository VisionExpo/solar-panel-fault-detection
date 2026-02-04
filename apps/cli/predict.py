import argparse
from pathlib import Path
import json

from solar_fault_detector.config.config import Config
from solar_fault_detector.inference.predictor import Predictor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local inference on a solar panel image"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to input image (jpg/png)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Path to trained model (defaults to config best model)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    config = Config()
    model_path = args.model_path or config.model.best_model_path

    predictor = Predictor(
        model_path=model_path,
        config=config.model,
    )

    prediction = predictor.predict(args.image)

    print(json.dumps(prediction, indent=2))


if __name__ == "__main__":
    main()
