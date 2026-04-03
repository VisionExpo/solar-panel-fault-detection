# 🐍 Python API Reference

Direct Python API for the Solar Panel Fault Detection system.

## Installation

```python
# Installation
pip install -r requirements-prod.txt
pip install -e .
```

## Modules

### Configuration

```python
from solar_fault_detector.config.config import Config, ModelConfig, DataConfig

# Load default configuration
config = Config()

# Access individual configs
print(config.model.img_size)      # (384, 384)
print(config.model.num_classes)   # 6
print(config.model.best_model_path)
```

### Model Creation

```python
from solar_fault_detector.models.factory import ModelFactory

# Create CNN model
config = Config()
cnn_model = ModelFactory.create("cnn", config.model)
cnn_model.build()
cnn_model.compile()

# Create Ensemble model (3 models)
ensemble_model = ModelFactory.create(
    "ensemble",
    config.model,
    num_models=3
)
ensemble_model.build()
ensemble_model.compile()
```

### Single Image Inference

```python
from pathlib import Path
from solar_fault_detector.inference.predictor import Predictor
from solar_fault_detector.config.config import Config

# Initialize
config = Config()
predictor = Predictor(
    model_path=Path("artifacts/models/best_model.h5"),
    config=config.model
)

# Make prediction
image_path = Path("test_image.jpg")
result = predictor.predict(image_path)

print(result)
# Output:
# {
#     'image': 'test_image.jpg',
#     'predicted_class': 2,
#     'confidence': 0.9847,
#     'probabilities': [0.0023, 0.0089, 0.9847, ...]
# }
```

### Batch Inference

```python
from pathlib import Path
from solar_fault_detector.inference.batch import BatchInferenceEngine
from solar_fault_detector.config.config import Config

# Initialize
config = Config()
batch_engine = BatchInferenceEngine(
    model_path=Path("artifacts/models/best_model.h5"),
    config=config.model
)

# Predict on directory
image_dir = Path("test_images")
results = batch_engine.predict_directory(image_dir)

for result in results:
    print(f"{result['image']}: Class {result['predicted_class']}")
```

### Data Processing

```python
from solar_fault_detector.data.preprocessing import ImagePreprocessor
from solar_fault_detector.config.config import Config
from pathlib import Path

# Initialize
config = Config()
preprocessor = ImagePreprocessor(config)

# Load and preprocess single image
image_path = Path("solar_panel.jpg")
processed_image = preprocessor.load_and_preprocess(image_path)
print(processed_image.shape)  # (384, 384, 3)

# Preprocess batch
image_paths = [Path("img1.jpg"), Path("img2.jpg")]
batch = preprocessor.preprocess_batch(image_paths)
print(batch.shape)  # (2, 384, 384, 3)
```

### Logging

```python
from solar_fault_detector.utils.logger import get_logger
from pathlib import Path

# Create logger
logger = get_logger(
    name=__name__,
    log_file=Path("logs/my_app.log")
)

logger.info("Application started")
logger.warning("This is a warning")
logger.error("An error occurred")
```

### Model Download Utility

```python
from solar_fault_detector.utils.download_model import ensure_model_exists
from pathlib import Path

# Ensure model exists (downloads if needed)
model_path = ensure_model_exists(
    model_path=Path("artifacts/models/best_model.h5"),
    repo_id="VishalGorule09/SolarPanelModel"
)

print(f"Model ready at: {model_path}")
```

---

## Complete Example: Inference Pipeline

```python
from pathlib import Path
from solar_fault_detector.config.config import Config
from solar_fault_detector.inference.predictor import Predictor
from solar_fault_detector.utils.logger import get_logger
from solar_fault_detector.utils.download_model import ensure_model_exists

# Setup
logger = get_logger(__name__)
config = Config()

# Ensure model exists
model_path = ensure_model_exists(config.model.best_model_path)
logger.info(f"Model loaded from: {model_path}")

# Create predictor
predictor = Predictor(model_path, config.model)

# Run inference
image_path = Path("test_image.jpg")
result = predictor.predict(image_path)

# Process result
class_names = [
    "Normal",
    "Dust",
    "High Temperature",
    "Cracks",
    "Water Damage",
    "Delamination"
]

predicted_class = result["predicted_class"]
confidence = result["confidence"]

logger.info(
    f"Prediction: {class_names[predicted_class]} "
    f"({confidence:.2%})"
)

# Return result
print(result)
```

---

## Complete Example: Batch Processing

```python
from pathlib import Path
from solar_fault_detector.config.config import Config
from solar_fault_detector.inference.batch import BatchInferenceEngine
from solar_fault_detector.utils.logger import get_logger

# Setup
logger = get_logger(__name__)
config = Config()

# Create batch engine
batch_engine = BatchInferenceEngine(
    model_path=Path("artifacts/models/best_model.h5"),
    config=config.model
)

# Process directory
image_dir = Path("test_images")
results = batch_engine.predict_directory(image_dir)

# Analyze results
logger.info(f"Processed {len(results)} images")

high_confidence = [r for r in results if r["confidence"] > 0.9]
logger.info(f"High confidence predictions: {len(high_confidence)}")

# Export results
for result in results:
    print(f"{result['image']}: "
          f"Class {result['predicted_class']} "
          f"({result['confidence']:.2%})")
```

---

## API Classes Reference

### Predictor

```python
class Predictor:
    def __init__(self, model_path: Path, config: ModelConfig)
    def predict(self, image_path: Path) -> Dict
```

**Methods**:
- `predict(image_path)`: Single image prediction

### BatchInferenceEngine

```python
class BatchInferenceEngine:
    def __init__(self, model_path: Path, config: ModelConfig)
    def predict_images(self, image_paths: List[Path]) -> List[Dict]
    def predict_directory(self, image_dir: Path) -> List[Dict]
```

**Methods**:
- `predict_images(paths)`: Batch prediction from paths list
- `predict_directory(dir)`: Predict all images in directory

### ImagePreprocessor

```python
class ImagePreprocessor:
    def __init__(self, config: Config)
    def load_and_preprocess(self, image_path: Path) -> np.ndarray
    def preprocess_batch(self, image_paths: List[Path]) -> np.ndarray
```

**Methods**:
- `load_and_preprocess()`: Load and process single image
- `preprocess_batch()`: Process multiple images

### ModelFactory

```python
class ModelFactory:
    @staticmethod
    def create(
        model_type: Literal["cnn", "ensemble"],
        config: ModelConfig,
        **kwargs
    ) -> BaseModel
```

**Methods**:
- `create()`: Factory method to create model instances

---

## Type Hints Reference

```python
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import tensorflow as tf
from solar_fault_detector.config.config import Config, ModelConfig

# Common types
PredictionResult = Dict[str, any]
ImageArray = np.ndarray
ModelType = tf.keras.Model
ConfigType = Config
```

