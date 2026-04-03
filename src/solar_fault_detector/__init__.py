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

__version__ = "1.0.0"
__author__ = "Vishal Gorule"
__description__ = "Solar Panel Fault Detection using Deep Learning"

from solar_fault_detector.config.config import Config
from solar_fault_detector.models.factory import ModelFactory
from solar_fault_detector.inference.predictor import Predictor
from solar_fault_detector.inference.batch import BatchInferenceEngine

__all__ = [
    "Config",
    "ModelFactory", 
    "Predictor",
    "BatchInferenceEngine",
]
