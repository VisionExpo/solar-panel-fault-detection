"""
Utility functions and helpers.

Available utilities:
- logger: Logging configuration and logger factory
- download_model: Model downloading from Hugging Face
- losses: Custom loss functions
- cache: Caching utilities for performance optimization
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from solar_fault_detector.utils.logger import get_logger
    from solar_fault_detector.utils.cache import (
        Cache,
        InMemoryCache,
        RedisCache,
        PredictionCache,
        ModelCache,
    )


_LAZY_EXPORTS = {
    "get_logger": ("solar_fault_detector.utils.logger", "get_logger"),
    "Cache": ("solar_fault_detector.utils.cache", "Cache"),
    "InMemoryCache": ("solar_fault_detector.utils.cache", "InMemoryCache"),
    "RedisCache": ("solar_fault_detector.utils.cache", "RedisCache"),
    "PredictionCache": ("solar_fault_detector.utils.cache", "PredictionCache"),
    "ModelCache": ("solar_fault_detector.utils.cache", "ModelCache"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_EXPORTS[name]
    module = __import__(module_path, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    "get_logger",
    "Cache",
    "InMemoryCache",
    "RedisCache",
    "PredictionCache",
    "ModelCache",
]
