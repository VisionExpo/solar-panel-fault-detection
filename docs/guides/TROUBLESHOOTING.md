# ⚠️ Troubleshooting Guide

Common issues and their solutions.

## Installation Issues

### Issue: `ModuleNotFoundError: No module named 'solar_fault_detector'`

**Cause**: Package not properly installed

**Solution**:
```bash
# Make sure you're in the project directory
cd solar-panel-fault-detection

# Reinstall the package
pip install -e .

# Verify installation
python -c "import solar_fault_detector; print('✅ OK')"
```

### Issue: `tensorflow not installed` or `ImportError: No module named 'tensorflow'`

**Cause**: TensorFlow dependencies not installed

**Solution**:
```bash
# Install full requirements
pip install -r requirements.txt

# Or install TensorFlow directly
pip install tensorflow>=2.13

# Verify
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## Model Loading Issues

### Issue: `FileNotFoundError: artifacts/models/best_model.h5`

**Cause**: Model file doesn't exist

**Solution**:
```bash
# Download model from Hugging Face
python -m solar_fault_detector.utils.download_model

# Or manually download
python -c "
from solar_fault_detector.utils.download_model import ensure_model_exists
from pathlib import Path

model_path = ensure_model_exists(
    Path('artifacts/models/best_model.h5')
)
print(f'Model ready at: {model_path}')
"
```

### Issue: `ValueError: Unknown layer type` or `Model loading failed`

**Cause**: Model incompatibility with TensorFlow version

**Solution**:
```bash
# Check TensorFlow version
python -c "import tensorflow as tf; print(f'TensorFlow: {tf.__version__}')"

# Update TensorFlow if necessary
pip install --upgrade tensorflow

# Try loading model
python -c "import tensorflow; m = tensorflow.keras.models.load_model('artifacts/models/best_model.h5')"
```

---

## Image Processing Issues

### Issue: `PIL.UnidentifiedImageError: cannot identify image file`

**Cause**: Image file is corrupted or invalid format

**Solution**:
```python
from PIL import Image

# Test if image is valid
try:
    img = Image.open("test_image.jpg")
    print(f"✅ Valid image: {img.size}")
except Exception as e:
    print(f"❌ Invalid image: {e}")
    # Try converting the image
    # Re-save with PIL
```

### Issue: `ValueError: Image too small` or `Image too large`

**Cause**: Image dimensions outside acceptable range

**Solution**:
```python
from PIL import Image

# Check image dimensions
img = Image.open("test_image.jpg")
print(f"Size: {img.size}")

# Resize if needed
MIN_SIZE = 50
MAX_SIZE = 10000

if img.size[0] < MIN_SIZE or img.size[1] < MIN_SIZE:
    print("❌ Image too small")

if img.size[0] > MAX_SIZE or img.size[1] > MAX_SIZE:
    print("❌ Image too large")
```

---

## API Issues

### Issue: `Connection refused: Cannot connect to http://localhost:8000`

**Cause**: API server not running

**Solution**:
```bash
# Start API server
python -m uvicorn apps.api.fastapi_app:app --reload

# Or use a specific port if 8000 is busy
python -m uvicorn apps.api.fastapi_app:app --port 8001

# Check if port is in use
# On Windows
netstat -ano | findstr :8000

# On Linux/Mac
lsof -i :8000
```

### Issue: `503 Service Unavailable: Model not loaded`

**Cause**: Model failed to load during startup

**Solution**:
```bash
# Check logs for errors
python -m uvicorn apps.api.fastapi_app:app

# Ensure model file exists
python -m solar_fault_detector.utils.download_model

# Try loading model directly
python -c "
from pathlib import Path
import tensorflow as tf

model = tf.keras.models.load_model('artifacts/models/best_model.h5')
print('✅ Model loaded successfully')
"
```

### Issue: `400 Bad Request: Unsupported file type`

**Cause**: File format not supported (must be JPG or PNG)

**Solution**:
```bash
# Convert image to supported format
python -c "
from PIL import Image

# Open any image format
img = Image.open('image.bmp')

# Convert to JPG
img.convert('RGB').save('image.jpg')
print('✅ Converted to JPG')
"

# Or use ImageMagick
convert image.bmp image.jpg
```

---

## Prediction Issues

### Issue: `ValueError: input_0 expected dtype float32 got dtype uint8`

**Cause**: Image not properly normalized

**Solution**: Image preprocessing should handle this automatically. If error persists:

```python
import numpy as np

# Ensure image is float32 and normalized
image = np.array(image, dtype=np.float32)
image = image / 255.0  # Normalize to [0, 1]
print(f"Image dtype: {image.dtype}, range: [{image.min()}, {image.max()}]")
```

### Issue: `Prediction confidence very low (< 0.1)` for all classes

**Cause**: Model not properly trained or image not preprocessed correctly

**Solution**:
```python
# Verify image preprocessing
from solar_fault_detector.data import ImagePreprocessor
from solar_fault_detector.config import Config

config = Config()
preprocessor = ImagePreprocessor(config.model)

image = preprocessor.load_and_preprocess("test_image.jpg")

# Check range and shape
print(f"Shape: {image.shape}")
print(f"Range: [{image.min()}, {image.max()}]")
print(f"Mean: {image.mean()}, Std: {image.std()}")

# Verification: Should be close to 0.5 mean
if image.mean() < 0.1 or image.mean() > 0.9:
    print("⚠️ Warning: Image may be too dark or bright")
```

---

## GPU Issues

### Issue: TensorFlow not using GPU

**Cause**: GPU drivers not installed or TensorFlow not compiled with GPU support

**Solution**:
```python
import tensorflow as tf

# Check available devices
print(tf.config.list_physical_devices())

# List GPUs
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs: {len(gpus)}")

# If 0 GPUs detected:
# 1. Update NVIDIA drivers
# 2. Reinstall TensorFlow: pip install tensorflow[and-cuda]
# 3. Check CUDA compatibility with TensorFlow version
```

### Issue: `CUDA out of memory`

**Cause**: GPU memory insufficient for batch

**Solution**:
```python
# Reduce batch size in config
from solar_fault_detector.config import Config

config = Config()
config.model.batch_size = 4  # Reduce from default 16

# Or use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU
```

---

## Deployment Issues

### Render Deployment Fails

**Cause**: Build timeout, missing model, or dependency issues

**Diagnosis**:
```bash
# Check build logs in Render dashboard
# Common issues:
# 1. Model download timeout
# 2. TensorFlow installation slow
# 3. Memory limit exceeded

# Solutions:
# - Increase build timeout
# - Use production requirements only
# - Increase Render plan (Standard not Free)
```

### Docker Build Fails

**Cause**: Dependencies won't install or TensorFlow too large

**Solution**:
```bash
# Use production requirements
docker build -t detector -f Dockerfile.prod .

# Or build manually
docker run -it python:3.10-slim bash
pip install -r requirements-prod.txt
python -m solar_fault_detector.utils.download_model
```

---

## Performance Issues

### Issue: Prediction takes > 5 seconds

**Cause**: Model inference slow, likely CPU-based

**Solution**:
```python
# Check device being used
import tensorflow as tf

with tf.device('/GPU:0'):  # Force GPU
    prediction = model.predict(image)

# Or optimize model
# - Use quantization
# - Reduce model size
# - Use GPU acceleration
```

### Issue: Memory usage grows over time (memory leak)

**Cause**: Models or images not being garbage collected

**Solution**:
```python
# Ensure cleanup after prediction
import gc

# Make prediction
result = predictor.predict(image_path)

# Force garbage collection
gc.collect()

# For batch processing, process in chunks
for batch in batch_generator(image_paths, batch_size=32):
    results = engine.predict_batch(batch)
    gc.collect()  # Clean up after each batch
```

---

## Debugging

### Enable Verbose Logging

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.debug("This is a debug message")
logger.info("This is an info message")
logger.error("This is an error message")
```

### Print Variable States

```python
from solar_fault_detector.data import ImagePreprocessor
from solar_fault_detector.config import Config

config = Config()
preprocessor = ImagePreprocessor(config.model)

# Add debugging to preprocessing
image = preprocessor.load_and_preprocess("test.jpg")

print(f"✓ Image shape: {image.shape}")
print(f"✓ Min value: {image.min():.4f}")
print(f"✓ Max value: {image.max():.4f}")
print(f"✓ Mean value: {image.mean():.4f}")
print(f"✓ Std dev: {image.std():.4f}")
```

### Full Stack Trace

```python
import traceback

try:
    result = predictor.predict("image.jpg")
except Exception as e:
    print("❌ Error occurred:")
    traceback.print_exc()
```

---

## Getting Help

1. **Check logs**: Look at error messages and logs
2. **Search issues**: Check GitHub Issues for similar problems
3. **Enable debug**: Set `LOG_LEVEL=DEBUG` for detailed output
4. **Test components**: Isolate which component is failing
5. **Minimal reproduction**: Create smallest possible example that fails

