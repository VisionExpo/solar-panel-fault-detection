from pathlib import Path
from typing import Optional

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from solar_fault_detector.config.config import Config
from solar_fault_detector.models.factory import ModelFactory
from solar_fault_detector.training.evaluator import Evaluator
from solar_fault_detector.monitoring.wandb_tracker import WandbTracker


class Trainer:
    """
    Unified training pipeline.
    Handles model creation, training, evaluation, and tracking.
    """

    def __init__(
        self,
        config: Config,
        model_type: str = "cnn",
        num_ensemble_models: int = 3,
    ):
        self.config = config

        # Model
        self.model_wrapper = ModelFactory.create(
            model_type=model_type,
            config=config.model,
            num_models=num_ensemble_models,
        )

        # Monitoring
        self.tracker = WandbTracker(
            training_config=config.training,
            model_config=config.model,
            run_name=model_type,
        )

        # Evaluation
        self.evaluator = Evaluator(tracker=self.tracker)

    def train(
        self,
        train_dataset,
        val_dataset,
    ) -> tf.keras.Model:
        """
        Execute full training workflow.
        """
        # Start experiment tracking
        self.tracker.start()

        # Build & compile model
        model = self.model_wrapper.build()
        self.model_wrapper.compile()

        callbacks = self._get_callbacks()

        # Train
        model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=self.config.model.epochs,
            callbacks=callbacks,
        )

        # Evaluate
        self.evaluator.evaluate(
            model,
            val_dataset,
            log_to_wandb=self.config.training.use_wandb,
        )

        # Save best model
        self.model_wrapper.save(self.config.model.best_model_path)

        # Finish tracking
        self.tracker.finish()

        return model

    def _get_callbacks(self):
        """
        Build training callbacks.
        """
        callbacks = []

        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=self.config.model.early_stopping_patience,
                restore_best_weights=True,
            )
        )

        callbacks.append(
            ModelCheckpoint(
                filepath=str(self.config.model.best_model_path),
                monitor="val_loss",
                save_best_only=True,
            )
        )

        return callbacks
