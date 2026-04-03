"""
Caching utilities for performance optimization.

Provides in-memory and Redis-based caching for predictions,
model loading, and preprocessing results.
"""

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class Cache:
    """
    Abstract base class for caching implementations.
    """

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL."""
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Delete value from cache."""
        raise NotImplementedError

    def clear(self) -> None:
        """Clear all cache entries."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        raise NotImplementedError


class InMemoryCache(Cache):
    """
    Simple in-memory cache with LRU eviction.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: list = []

    def get(self, key: str) -> Optional[Any]:
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if key in self.cache:
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # Evict least recently used
            lru_key = self.access_order.pop(0)
            del self.cache[lru_key]

        self.cache[key] = value
        self.access_order.append(key)

    def delete(self, key: str) -> None:
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)

    def clear(self) -> None:
        self.cache.clear()
        self.access_order.clear()

    def exists(self, key: str) -> bool:
        return key in self.cache


class RedisCache(Cache):
    """
    Redis-based distributed cache.
    """

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        try:
            import redis
            self.redis = redis.Redis(host=host, port=port, db=db)
            self.redis.ping()  # Test connection
        except ImportError:
            raise ImportError("redis package required for RedisCache")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Falling back to InMemoryCache")
            self.redis = None
            self.fallback_cache = InMemoryCache()

    def get(self, key: str) -> Optional[Any]:
        if self.redis:
            try:
                data = self.redis.get(key)
                return pickle.loads(data) if data else None
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
                return self.fallback_cache.get(key) if hasattr(self, 'fallback_cache') else None
        else:
            return self.fallback_cache.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        if self.redis:
            try:
                data = pickle.dumps(value)
                self.redis.set(key, data, ex=ttl)
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
                if hasattr(self, 'fallback_cache'):
                    self.fallback_cache.set(key, value, ttl)
        else:
            self.fallback_cache.set(key, value, ttl)

    def delete(self, key: str) -> None:
        if self.redis:
            try:
                self.redis.delete(key)
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
                if hasattr(self, 'fallback_cache'):
                    self.fallback_cache.delete(key)
        else:
            self.fallback_cache.delete(key)

    def clear(self) -> None:
        if self.redis:
            try:
                self.redis.flushdb()
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
                if hasattr(self, 'fallback_cache'):
                    self.fallback_cache.clear()
        else:
            self.fallback_cache.clear()

    def exists(self, key: str) -> bool:
        if self.redis:
            try:
                return self.redis.exists(key) > 0
            except Exception as e:
                logger.warning(f"Redis exists failed: {e}")
                return self.fallback_cache.exists(key) if hasattr(self, 'fallback_cache') else False
        else:
            return self.fallback_cache.exists(key)


class PredictionCache:
    """
    Specialized cache for model predictions.

    Caches predictions based on image hash to avoid redundant computations.
    """

    def __init__(self, cache_backend: Optional[Cache] = None, ttl: int = 3600):
        """
        Initialize prediction cache.

        Args:
            cache_backend: Cache implementation (defaults to InMemoryCache)
            ttl: Time to live in seconds
        """
        self.cache = cache_backend or InMemoryCache()
        self.ttl = ttl

    def _compute_image_hash(self, image_array: np.ndarray) -> str:
        """Compute hash of image array for cache key."""
        # Convert to bytes and hash
        image_bytes = image_array.tobytes()
        return hashlib.md5(image_bytes).hexdigest()

    def get_prediction(self, image_array: np.ndarray) -> Optional[Dict]:
        """
        Get cached prediction for image.

        Args:
            image_array: Preprocessed image array

        Returns:
            Cached prediction dict or None
        """
        cache_key = f"pred_{self._compute_image_hash(image_array)}"
        return self.cache.get(cache_key)

    def set_prediction(self, image_array: np.ndarray, prediction: Dict) -> None:
        """
        Cache prediction for image.

        Args:
            image_array: Preprocessed image array
            prediction: Prediction result dict
        """
        cache_key = f"pred_{self._compute_image_hash(image_array)}"
        self.cache.set(cache_key, prediction, self.ttl)

    def clear_expired(self) -> None:
        """Clear expired cache entries (if supported by backend)."""
        # InMemoryCache doesn't support TTL, Redis does
        pass


class ModelCache:
    """
    Cache for loaded models to avoid reloading.
    """

    def __init__(self, cache_backend: Optional[Cache] = None):
        self.cache = cache_backend or InMemoryCache()

    def get_model(self, model_path: Union[str, Path]) -> Optional[Any]:
        """Get cached model."""
        cache_key = f"model_{str(model_path)}"
        return self.cache.get(cache_key)

    def set_model(self, model_path: Union[str, Path], model: Any) -> None:
        """Cache loaded model."""
        cache_key = f"model_{str(model_path)}"
        self.cache.set(cache_key, model)

    def invalidate_model(self, model_path: Union[str, Path]) -> None:
        """Remove model from cache."""
        cache_key = f"model_{str(model_path)}"
        self.cache.delete(cache_key)