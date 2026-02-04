from pathlib import Path
import json
import csv

from solar_fault_detector.config.config import Config
from solar_fault_detector.inference.batch import BatchInferenceEngine


def run_batch_inference(
    image_dir: Path,
    output_dir: Path,
):
    """
    Run batch inference on a directory of images and save results.
    """
    config = Config()
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = BatchInferenceEngine(
        model_path=config.model.best_model_path,
        config=config.model,
    )

    results = engine.predict_directory(image_dir)

    # Save JSON
    json_path = output_dir / "predictions.json"
    with json_path.open("w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image", "predicted_class", "confidence"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "image": r["image"],
                    "predicted_class": r["predicted_class"],
                    "confidence": r["confidence"],
                }
            )


if __name__ == "__main__":
    run_batch_inference(
        image_dir=Path("data/batch_images"),
        output_dir=Path("artifacts/batch_output"),
    )
