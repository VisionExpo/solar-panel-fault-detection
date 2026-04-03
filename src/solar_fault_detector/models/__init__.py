"""
Model architectures and factory for Solar Panel Fault Detection.

Available models:
- CNNModel: Convolutional Neural Network
- EnsembleModel: Ensemble of multiple models
"""

from solar_fault_detector.models.base import BaseModel
from solar_fault_detector.models.cnn import CNNModel
from solar_fault_detector.models.ensemble import EnsembleModel
from solar_fault_detector.models.factory import ModelFactory

__all__ = [
    "BaseModel",
    "CNNModel",
    "EnsembleModel",
    "ModelFactory",
]
