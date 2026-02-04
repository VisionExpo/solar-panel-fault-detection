from typing import Literal

from solar_fault_detector.config.config import ModelConfig
from solar_fault_detector.models.base import BaseModel
from solar_fault_detector.models.cnn import CNNModel
from solar_fault_detector.models.ensemble import EnsembleModel


class ModelFactory:
    """
    Factory class to create model instances dynamically.
    """

    @staticmethod
    def create(
        model_type: Literal["cnn", "ensemble"],
        config: ModelConfig,
        **kwargs
    ) -> BaseModel:
        """
        Create and return a model instance based on model_type.
        """
        if model_type == "cnn":
            return CNNModel(config)

        if model_type == "ensemble":
            num_models = kwargs.get("num_models", 3)
            return EnsembleModel(config, num_models=num_models)

        raise ValueError(
            f"Unsupported model type: {model_type}. "
            "Supported types are ['cnn', 'ensemble']."
        )
