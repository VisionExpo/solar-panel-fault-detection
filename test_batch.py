import sys
from unittest.mock import MagicMock
sys.modules['numpy'] = MagicMock()
sys.modules['tensorflow'] = MagicMock()
sys.modules['PIL'] = MagicMock()
sys.modules['redis'] = MagicMock()

from solar_fault_detector.inference.batch import BatchInferenceEngine
from solar_fault_detector.config.config import ModelConfig
from pathlib import Path

config = ModelConfig()
engine = BatchInferenceEngine(
    model_path=Path("dummy"),
    config=config,
    use_cache=True,
    cache_backend="redis"
)
from solar_fault_detector.utils.cache import InMemoryCache
assert isinstance(engine.model_cache.cache, InMemoryCache), "ModelCache should use InMemoryCache"
print("Success!")
