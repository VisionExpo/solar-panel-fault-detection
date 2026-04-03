# 📊 Data Pipeline Documentation

Complete guide to data handling and preprocessing.

## Overview

The data pipeline handles the complete lifecycle of solar panel images:
1. **Loading** - Read images from disk
2. **Validation** - Check format and dimensions
3. **Preprocessing** - Resize, normalize, augment
4. **Batching** - Group for efficient processing
5. **Augmentation** - Apply transformations

---

## Data Preprocessing

### Image Preprocessing

```
Raw Image (Any size, any format)
    │
    ├─ Load with Pillow
    ├─ Convert to RGB
    ├─ Resize to 384x384
    ├─ Normalize to [0, 1]
    ├─ Convert to array
    └─ Return preprocessed image
```

### Specifications

| Parameter | Value |
|-----------|-------|
| Input Size | Variable |
| Output Size | 384 x 384 |
| Channels | 3 (RGB) |
| Data Type | float32 |
| Normalization | Pixel values / 255.0 |

---

## Data Augmentation Strategy

Applied during training to increase model robustness:

### Geometric Transformations
- **Rotation**: ±20 degrees
- **Width Shift**: ±20% of width
- **Height Shift**: ±20% of height
- **Shear**: ±20 degrees

### Appearance Transformations
- **Zoom**: ±20%
- **Horizontal Flip**: 50% probability
- **Fill Mode**: Nearest neighbor

### Configuration

```python
from solar_fault_detector.config.config import AugmentationConfig

augmentation = AugmentationConfig(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)
```

---

## Dataset Structure

### Required Format

```
your_dataset/
├── class_0/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── class_1/
│   ├── image_001.jpg
│   └── ...
├── class_2/
│   └── ...
├── class_3/
│   └── ...
├── class_4/
│   └── ...
└── class_5/
    └── ...
```

### Class Labels

| ID | Label | Description |
|----|-------|-------------|
| 0 | Normal | No faults detected |
| 1 | Dust | Dust accumulation |
| 2 | High Temperature | Thermal issues |
| 3 | Cracks | Panel cracks |
| 4 | Water Damage | Water exposure damage |
| 5 | Delamination | Material separation |

---

## Train/Val/Test Split

Default configuration:
- **Training**: 80%
- **Validation**: 10%
- **Testing**: 10%

```python
from solar_fault_detector.config.config import DataConfig

data_config = DataConfig(
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    random_state=42
)
```

---

## Data Loading in Code

### Single Image

```python
from solar_fault_detector.data.preprocessing import ImagePreprocessor
from solar_fault_detector.config.config import Config
from pathlib import Path

config = Config()
preprocessor = ImagePreprocessor(config)

# Load and preprocess
image_path = Path("solar_panel.jpg")
preprocessed = preprocessor.load_and_preprocess(image_path)

print(preprocessed.shape)  # (384, 384, 3)
print(preprocessed.min(), preprocessed.max())  # 0.0, 1.0
```

### Batch Processing

```python
from pathlib import Path
from solar_fault_detector.data.preprocessing import ImagePreprocessor
from solar_fault_detector.config.config import Config

config = Config()
preprocessor = ImagePreprocessor(config)

# Load multiple images
image_paths = [
    Path("img1.jpg"),
    Path("img2.jpg"),
    Path("img3.jpg")
]

batch = preprocessor.preprocess_batch(image_paths)
print(batch.shape)  # (3, 384, 384, 3)
```

---

## Memory Considerations

### Image Memory Usage

```
Per Image: 384 × 384 × 3 × 4 bytes = 1.76 MB (float32)

Batch Sizes:
- Single: ~2 MB
- Batch of 8: ~14 MB
- Batch of 16: ~28 MB
- Batch of 32: ~56 MB
```

### Recommended Batch Sizes

| Hardware | Recommended | Max |
|----------|-------------|-----|
| CPU | 8 | 16 |
| GPU (2GB VRAM) | 16 | 32 |
| GPU (4GB VRAM) | 32 | 64 |
| GPU (8GB VRAM) | 64 | 128 |

---

## Data Quality Checks

### Validation Steps

1. **File Existence**: File must exist
2. **Format Check**: Must be JPG/PNG
3. **Readability**: Image must be readable
4. **Dimension**: Must be at least 50x50 pixels
5. **Channels**: Must be convertible to RGB

### Error Handling

```python
try:
    image = preprocessor.load_and_preprocess(image_path)
except FileNotFoundError:
    print(f"Image not found: {image_path}")
except ValueError as e:
    print(f"Invalid image: {e}")
except Exception as e:
    print(f"Processing error: {e}")
```

---

## Performance Optimization

### Tips for Faster Processing

1. **Use Batch Processing**: Process multiple images together
   ```python
   # ❌ Slow
   for img in images:
       result = predictor.predict(img)
   
   # ✅ Fast
   results = batch_engine.predict_images(images)
   ```

2. **Cache Preprocessed Images**: Reuse preprocessed data
   ```python
   cache = {}
   for img_path in images:
       if img_path not in cache:
           cache[img_path] = preprocess(img_path)
   ```

3. **Use GPU**: Enable GPU acceleration
   ```python
   import tensorflow as tf
   print(tf.config.list_physical_devices('GPU'))
   ```

4. **Optimize Image Loading**: Use memory-mapped files for large datasets

---

## Data Statistics

### Recommended Dataset Size

| Use Case | Minimum | Recommended |
|----------|---------|-------------|
| Testing | 100 images | 500 images |
| Validation | 100 images | 500 images |
| Training | 500 images | 5000+ images |

### Class Balance

Recommended distribution:
- Balanced across all 6 classes
- Equal samples per class
- Stratified train/val/test split

