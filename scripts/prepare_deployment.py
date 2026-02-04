from pathlib import Path
import shutil

from solar_fault_detector.config.config import Config


def prepare_deployment(
    output_dir: Path = Path("deployment/artifacts"),
):
    """
    Prepare model artifacts for deployment.
    """
    config = Config()

    model_path = config.model.best_model_path
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run training before deployment."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy model
    model_dest = output_dir / "model"
    if model_dest.exists():
        shutil.rmtree(model_dest)

    shutil.copytree(model_path.parent, model_dest)

    # Copy label mapping if present
    label_mapping = Path("src/solar_fault_detector/models/label_mapping.json")
    if label_mapping.exists():
        shutil.copy(label_mapping, output_dir / "label_mapping.json")

    print("Deployment artifacts prepared:")
    print(f"- Model directory: {model_dest.resolve()}")
    print(f"- Output path: {output_dir.resolve()}")


if __name__ == "__main__":
    prepare_deployment()
