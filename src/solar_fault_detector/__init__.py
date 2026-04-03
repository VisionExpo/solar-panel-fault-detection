"""
Solar Panel Fault Detection System

A comprehensive deep learning system for detecting and classifying faults
in solar panels using computer vision and machine learning.

Core modules:
- config: Configuration management
- models: Model architectures and factory
- training: Training pipelines and evaluation
- inference: Inference engines for predictions
- monitoring: Experiment tracking and monitoring
- utils: Utility functions and helpers
- data: Data loading and preprocessing
"""

from typing import TYPE_CHECKING, Any

__version__ = "1.0.0"
__author__ = "Vishal Gorule"
__description__ = "Solar Panel Fault Detection using Deep Learning"

if TYPE_CHECKING:
    from solar_fault_detector.config.config import Config
    from solar_fault_detector.models.factory import ModelFactory
    from solar_fault_detector.inference.predictor import Predictor
    from solar_fault_detector.inference.batch import BatchInferenceEngine


_LAZY_EXPORTS = {
    "Config": ("solar_fault_detector.config.config", "Config"),
    "ModelFactory": ("solar_fault_detector.models.factory", "ModelFactory"),
    "Predictor": ("solar_fault_detector.inference.predictor", "Predictor"),
    "BatchInferenceEngine": ("solar_fault_detector.inference.batch", "BatchInferenceEngine"),
}


def __getattr__(name: str) -> Any:
    """
    Lazily load top-level exports so package import has no heavy side effects.
    """
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_EXPORTS[name]
    module = __import__(module_path, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value

__all__ = [
    "Config",
    "ModelFactory", 
    "Predictor",
    "BatchInferenceEngine",
]
