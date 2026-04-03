# 🔧 Configuration Guide

Customize settings and hyperparameters for your needs.

## Configuration File

Configuration is managed in `src/solar_fault_detector/config/config.py`

### Model Configuration

```python
from solar_fault_detector.config.config import ModelConfig

model_config = ModelConfig(
    img_size=(384, 384),        # Input image dimensions
    num_channels=3,              # RGB channels
    num_classes=6,              # Number of fault classes
    batch_size=16,              # Training batch size
    epochs=50,                  # Number of epochs
    learning_rate=5e-4,         # Adam optimizer learning rate
    early_stopping_patience=8,  # Early stopping patience
    model_dir="artifacts/models",        # Model directory
    best_model_path="artifacts/models/best_model.h5"  # Best model path
)
```

### Data Configuration

```python
from solar_fault_detector.config.config import DataConfig

data_config = DataConfig(
    root_dir="artifacts",           # Root data directory
    data_dir="Faulty_solar_panel",  # Dataset directory
    train_ratio=0.8,               # Training split
    val_ratio=0.1,                 # Validation split
    test_ratio=0.1,                # Test split
    random_state=42                # Random seed for reproducibility
)
```

### Training Configuration

```python
from solar_fault_detector.config.config import TrainingConfig

training_config = TrainingConfig(
    use_wandb=True,                                    # Enable W&B tracking
    wandb_project="solar_panel_fault_detection",       # Project name
    wandb_entity="your_username",                      # W&B entity
    wandb_run_group="baseline",                        # Run group name
    wandb_mode="online",                               # online | offline | disabled
    log_images=True,                                   # Log sample images
    log_confusion_matrix=True                          # Log confusion matrix
)
```

### Augmentation Configuration

```python
from solar_fault_detector.config.config import AugmentationConfig

augmentation_config = AugmentationConfig(
    rotation_range=20,              # Rotation range in degrees
    width_shift_range=0.2,          # Width shift range (fraction)
    height_shift_range=0.2,         # Height shift range (fraction)
    shear_range=0.2,                # Shear range
    zoom_range=0.2,                 # Zoom range
    horizontal_flip=True,           # Enable horizontal flip
    fill_mode="nearest"             # Pixel fill mode
)
```

---

## Using Custom Configuration

### Load Default Config

```python
from solar_fault_detector.config import Config

config = Config()

# Access sub-configs
print(config.model.img_size)
print(config.data.train_ratio)
print(config.training.use_wandb)
```

### Override Individual Settings

```python
from solar_fault_detector.config import Config

config = Config()

# Modify settings
config.model.batch_size = 32
config.model.epochs = 100
config.training.wandb_mode = "offline"

print(config.model.batch_size)  # 32
```

### Create Custom Config

```python
from solar_fault_detector.config import Config, ModelConfig

config = Config()

# Create custom model config
custom_model = ModelConfig(
    img_size=(512, 512),
    batch_size=8,
    epochs=30,
    learning_rate=1e-4
)

# Or modify dataclass
from dataclasses import replace

config = replace(
    config,
    model=custom_model
)
```

---

## Environment Variables

### W&B Configuration

```bash
# Set W&B API key
export WANDB_API_KEY=your_api_key_here

# Disable W&B for offline training
export WANDB_MODE=offline
```

### GPU Configuration

```bash
# Use specific GPU
export CUDA_VISIBLE_DEVICES=0

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### Logging

```bash
# Set log level
export LOG_LEVEL=DEBUG   # DEBUG, INFO, WARNING, ERROR
```

---

## Optimization Tips

### For Faster Training

```python
config.model.batch_size = 32  # Increase batch size
config.model.learning_rate = 1e-3  # Increase learning rate
config.model.epochs = 30  # Reduce epochs with early stopping
```

### For Better Accuracy

```python
config.model.batch_size = 8  # Smaller batches
config.model.learning_rate = 5e-4  # Fine-tuned learning rate
config.model.epochs = 100  # More epochs
config.augmentation.zoom_range = 0.3  # More augmentation
```

### For Memory Constraints

```python
config.model.batch_size = 4  # Reduce batch size
config.model.img_size = (256, 256)  # Smaller images
```

### For Production

```python
config.training.use_wandb = False  # Disable experiment tracking
config.training.log_images = False  # Don't log sample images
config.model.early_stopping_patience = 5  # Earlier stopping
```

---

## Configuration Validation

```python
from solar_fault_detector.config import Config

config = Config()

# Validate configuration
assert config.model.num_classes > 0, "Invalid num_classes"
assert config.data.train_ratio + config.data.val_ratio + config.data.test_ratio == 1.0, \
    "Split ratios must sum to 1.0"
assert config.model.learning_rate > 0, "Learning rate must be positive"

print("✅ Configuration is valid")
```

---

## Configuration Export

```python
from solar_fault_detector.config import Config
import json

config = Config()

# Convert to dict
config_dict = {
    'model': config.model.__dict__,
    'data': config.data.__dict__,
    'training': config.training.__dict__,
    'augmentation': config.augmentation.__dict__,
}

# Save to JSON
with open("config.json", "w") as f:
    json.dump(config_dict, f, indent=2, default=str)

# Load from JSON
with open("config.json", "r") as f:
    loaded_dict = json.load(f)
    print(json.dumps(loaded_dict, indent=2))
```

