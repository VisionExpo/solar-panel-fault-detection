"""
Inference engines for real-time and batch predictions.

Available engines:
- Predictor: Single image prediction
- BatchInferenceEngine: Multiple images processing
"""

from solar_fault_detector.inference.predictor import Predictor
from solar_fault_detector.inference.batch import BatchInferenceEngine

__all__ = [
    "Predictor",
    "BatchInferenceEngine",
]
