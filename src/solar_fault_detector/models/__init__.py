"""
Model architectures and factory for Solar Panel Fault Detection.

Available models:
- CNNModel: Convolutional Neural Network
- EnsembleModel: Ensemble of multiple models
- ModelOptimizer: Model optimization and quantization
- ModelExplainer: Model explainability with LIME and SHAP
"""

from solar_fault_detector.models.base import BaseModel
from solar_fault_detector.models.cnn import CNNModel
from solar_fault_detector.models.ensemble import EnsembleModel
from solar_fault_detector.models.factory import ModelFactory
from solar_fault_detector.models.optimizer import ModelOptimizer
from solar_fault_detector.models.explainer import ModelExplainer

__all__ = [
    "BaseModel",
    "CNNModel",
    "EnsembleModel",
    "ModelFactory",
    "ModelOptimizer",
    "ModelExplainer",
]
