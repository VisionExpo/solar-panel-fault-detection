from typing import Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from solar_fault_detector.monitoring.wandb_tracker import WandbTracker


class Evaluator:
    """
    Handles model evaluation on validation or test datasets.
    """

    def __init__(self, tracker: WandbTracker | None = None):
        self.tracker = tracker

    def evaluate(
        self,
        model: tf.keras.Model,
        dataset,
        step: int | None = None,
        log_to_wandb: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate model and return metrics.
        """
        actuals_list = []
        preds_list = []

        for batch_x, batch_y in dataset:
            preds_batch = model(batch_x, training=False).numpy()
            actuals_list.append(np.argmax(batch_y, axis=1))
            preds_list.append(np.argmax(preds_batch, axis=1))

        y_true = np.concatenate(actuals_list)
        y_pred = np.concatenate(preds_list)
        del actuals_list
        del preds_list

        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        }

        if self.tracker and log_to_wandb:
            self.tracker.log_metrics(metrics, step=step)

        return metrics

    def confusion_matrix(self, model: tf.keras.Model, dataset) -> np.ndarray:
        """
        Compute confusion matrix.
        """
        actuals_list = []
        preds_list = []

        for batch_x, batch_y in dataset:
            preds_batch = model(batch_x, training=False).numpy()
            actuals_list.append(np.argmax(batch_y, axis=1))
            preds_list.append(np.argmax(preds_batch, axis=1))

        y_true = np.concatenate(actuals_list)
        y_pred = np.concatenate(preds_list)
        del actuals_list
        del preds_list

        return confusion_matrix(y_true, y_pred)
