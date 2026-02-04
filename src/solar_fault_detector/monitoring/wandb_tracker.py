from typing import Dict, Any, Optional, List
from pathlib import Path

import wandb
import numpy as np

from solar_fault_detector.config.config import TrainingConfig, ModelConfig


class WandbTracker:
    """
    Centralized Weights & Biases tracker.
    Handles experiment lifecycle and logging.
    """

    def __init__(
        self,
        training_config: TrainingConfig,
        model_config: ModelConfig,
        run_name: Optional[str] = None,
    ):
        self.training_config = training_config
        self.model_config = model_config
        self.run_name = run_name
        self._run = None

    def start(self) -> None:
        """
        Initialize W&B run.
        """
        if not self.training_config.use_wandb:
            return

        self._run = wandb.init(
            project=self.training_config.wandb_project,
            entity=self.training_config.wandb_entity,
            group=self.training_config.wandb_run_group,
            name=self.run_name,
            mode=self.training_config.wandb_mode,
            config={
                **vars(self.model_config),
            },
        )

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        """
        Log scalar metrics.
        """
        if not self._run:
            return

        wandb.log(metrics, step=step)

    def log_images(
        self,
        images: List[np.ndarray],
        captions: Optional[List[str]] = None,
        key: str = "predictions",
    ) -> None:
        """
        Log images to W&B.
        """
        if not self._run:
            return

        wandb_images = [
            wandb.Image(img, caption=cap if captions else None)
            for img, cap in zip(images, captions or [])
        ]
        wandb.log({key: wandb_images})

    def log_artifact(self, artifact_path: Path, name: str, artifact_type: str) -> None:
        """
        Log a file or directory as a W&B artifact.
        """
        if not self._run:
            return

        artifact = wandb.Artifact(name=name, type=artifact_type)
        artifact.add_dir(str(artifact_path))
        wandb.log_artifact(artifact)

    def finish(self) -> None:
        """
        Finalize W&B run.
        """
        if self._run:
            self._run.finish()
            self._run = None
