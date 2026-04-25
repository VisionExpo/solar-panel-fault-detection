from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
import tensorflow as tf

from solar_fault_detector.data.preprocessing import ImagePreprocessor
from solar_fault_detector.config.config import ModelConfig
from solar_fault_detector.utils.cache import PredictionCache, ModelCache


class BatchInferenceEngine:
    """
    Optimized batch inference engine with caching and performance enhancements.

    Features:
    - Memory-efficient batch processing
    - Prediction caching
    - Model caching
    - Configurable batch sizes
    - Progress tracking
    """

    def __init__(
        self,
        model_path: Path,
        config: ModelConfig,
        use_cache: bool = True,
        cache_backend: Optional[str] = None,
        batch_size: int = 32,
    ):
        """
        Initialize batch inference engine.

        Args:
            model_path: Path to saved model
            config: Model configuration
            use_cache: Whether to use prediction caching
            cache_backend: Cache backend ('memory' or 'redis')
            batch_size: Batch size for processing
        """
        self.config = config
        self.batch_size = batch_size
        self.preprocessor = ImagePreprocessor(config)

        # Initialize caches
        self.use_cache = use_cache
        if use_cache:
            if cache_backend == "redis":
                from solar_fault_detector.utils.cache import RedisCache

                cache = RedisCache()  # type: ignore
            else:
                from solar_fault_detector.utils.cache import InMemoryCache

                cache = InMemoryCache()  # type: ignore

            self.prediction_cache = PredictionCache(cache)  # type: ignore
            self.model_cache = ModelCache(cache)  # type: ignore
        else:
            self.prediction_cache = None  # type: ignore
            self.model_cache = None  # type: ignore

        # Load model with caching
        self.model = self._load_model(model_path)

    def _load_model(self, model_path: Path) -> tf.keras.Model:
        """Load model with caching support."""
        if self.model_cache:
            cached_model = self.model_cache.get_model(model_path)
            if cached_model:
                return cached_model

        model = tf.keras.models.load_model(model_path)

        if self.model_cache:
            self.model_cache.set_model(model_path, model)

        return model

    def predict_images(self, image_paths: List[Path]) -> List[Dict]:
        """
        Run optimized batch inference on image paths.

        Uses caching and memory-efficient processing.
        """
        results = []

        # Process in optimized batches
        for i in range(0, len(image_paths), self.batch_size):
            batch_paths = image_paths[i : i + self.batch_size]
            batch_results = self._predict_batch(batch_paths)
            results.extend(batch_results)

        return results

    def _predict_batch(self, image_paths: List[Path]) -> List[Dict]:
        """Predict on a single batch with caching."""
        results = []
        # Check cache first
        if self.prediction_cache:
            cached_results = []
            uncached_paths = []
            uncached_indices = []

            for idx, path in enumerate(image_paths):
                try:
                    # Load and check cache
                    image_array = self.preprocessor.load_and_preprocess(path)
                    cached_result = self.prediction_cache.get_prediction(image_array)
                    if cached_result:
                        cached_result["image"] = path.name  # Update path
                        cached_results.append((idx, cached_result))
                    else:
                        uncached_paths.append(path)
                        uncached_indices.append(idx)
                except Exception:
                    # On error, process without caching
                    uncached_paths.append(path)
                    uncached_indices.append(idx)

            # Process uncached images
            if uncached_paths:
                batch = self.preprocessor.preprocess_batch(uncached_paths)
                predictions = self.model(batch, training=False).numpy()

                for path, probs, orig_idx in zip(
                    uncached_paths, predictions, uncached_indices
                ):
                    result = {
                        "image": path.name,
                        "predicted_class": int(np.argmax(probs)),
                        "confidence": float(np.max(probs)),
                        "probabilities": probs.tolist(),
                    }
                    results.append((orig_idx, result))

                    # Cache the result
                    if self.prediction_cache:
                        try:
                            image_array = self.preprocessor.load_and_preprocess(path)
                            self.prediction_cache.set_prediction(image_array, result)
                        except Exception:
                            pass  # Skip caching on error
            else:
                results = cached_results
        else:
            # No caching - process all at once
            batch = self.preprocessor.preprocess_batch(image_paths)
            predictions = self.model(batch, training=False).numpy()

            for path, probs in zip(image_paths, predictions):
                result = {
                    "image": path.name,
                    "predicted_class": int(np.argmax(probs)),
                    "confidence": float(np.max(probs)),
                    "probabilities": probs.tolist(),
                }
                results.append((-1, result))

        # Sort by original order if needed
        if results and results[0][0] is not None:
            results.sort(key=lambda x: x[0])  # type: ignore
            final_results = [r for _, r in results]
        else:
            final_results = [r for _, r in results]

        return final_results

    def predict_directory(
        self, image_dir: Path, recursive: bool = False, file_pattern: str = "*.jpg"
    ) -> List[Dict]:
        """
        Run inference on all images in a directory.

        Args:
            image_dir: Directory containing images
            recursive: Whether to search subdirectories
            file_pattern: Glob pattern for image files
        """
        if recursive:
            image_paths = list(image_dir.rglob(file_pattern))
        else:
            image_paths = list(image_dir.glob(file_pattern))

        # Filter for supported formats
        supported_extensions = {".jpg", ".jpeg", ".png"}
        image_paths = [
            p for p in image_paths if p.suffix.lower() in supported_extensions
        ]

        if not image_paths:
            raise ValueError(f"No supported images found in directory: {image_dir}")

        return self.predict_images(image_paths)

    def benchmark_batch_size(
        self, test_image_path: Path, batch_sizes: List[int]
    ) -> Dict:
        """
        Benchmark different batch sizes for optimal performance.

        Args:
            test_image_path: Path to a test image
            batch_sizes: List of batch sizes to test

        Returns:
            Dictionary with timing results for each batch size
        """
        import time

        results = {}

        # Load test image
        test_image = self.preprocessor.load_and_preprocess(test_image_path)

        for batch_size in batch_sizes:
            # Create batch
            batch = np.tile(test_image, (batch_size, 1, 1, 1))

            # Warm up
            _ = self.model(batch[:1], training=False).numpy()

            # Time batch prediction
            start_time = time.time()
            _ = self.model(batch, training=False).numpy()
            end_time = time.time()

            batch_time = end_time - start_time
            per_image_time = batch_time / batch_size

            results[batch_size] = {
                "batch_time": batch_time,
                "per_image_time": per_image_time,
                "throughput": batch_size / batch_time,
            }

        return results
