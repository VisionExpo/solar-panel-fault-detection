1. **Fix Default Cache Bounds in `cache.py`**
   - In `src/solar_fault_detector/utils/cache.py`, update `PredictionCache.__init__` to default to `InMemoryCache(max_size=1000)`.
   - Update `ModelCache.__init__` to default to `InMemoryCache(max_size=2)`.
   - This prevents Out-Of-Memory crashes caused by unbounded caching of models and predictions, adhering to the memory constraints outlined in the memory.

2. **Fix `BatchInferenceEngine` ModelCache Override**
   - In `src/solar_fault_detector/inference/batch.py`, remove the lines `from solar_fault_detector.utils.cache import InMemoryCache` and `self.model_cache = ModelCache(InMemoryCache())` inside the `if use_cache:` block which overwrites the properly bounded model cache with an unbounded one.

3. **Complete pre commit steps**
   - Complete pre commit steps to make sure proper testing, verifications, reviews and reflections are done.

4. **Submit the fix**
   - Run tests (`PYTHONPATH=. make test`) and commit/push the solution to fix the OOM crash on Render.
