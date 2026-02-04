from pathlib import Path

import tensorflow as tf

from solar_fault_detector.config.config import Config
from solar_fault_detector.training.evaluator import Evaluator
from solar_fault_detector.monitoring.wandb_tracker import WandbTracker


def run_evaluation(
    model_path: Path | None = None,
    dataset=None,
):
    """
    Run evaluation on a trained model.
    """
    config = Config()

    model_path = model_path or config.model.best_model_path
    model = tf.keras.models.load_model(model_path)

    tracker = WandbTracker(
        training_config=config.training,
        model_config=config.model,
        run_name="evaluation",
    )

    tracker.start()

    evaluator = Evaluator(tracker=tracker)
    metrics = evaluator.evaluate(
        model=model,
        dataset=dataset,
        log_to_wandb=config.training.use_wandb,
    )

    tracker.finish()
    return metrics


if __name__ == "__main__":
    # Dataset should be constructed in pipeline or imported from data layer
    raise RuntimeError(
        "Dataset not provided. "
        "Use run_evaluation(model_path, dataset) from pipeline."
    )
