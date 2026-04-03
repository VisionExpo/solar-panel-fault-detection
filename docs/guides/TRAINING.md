# 🎓 Training Guide

Complete guide to training models locally.

## Prerequisites

- Python 3.9+
- CUDA/GPU (optional but recommended)
- ~10 GB disk space for dataset + model
- Ray or access to labeled solar panel dataset

## Dataset Preparation

### Directory Structure

Organize your dataset:
```
solar_panel_images/
├── 0_normal/
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── 1_dust/
│   ├── img_001.jpg
│   └── ...
├── 2_high_temp/
│   ├── img_001.jpg
│   └── ...
├── 3_cracks/
│   ├── img_001.jpg
│   └── ...
├── 4_water_damage/
│   ├── img_001.jpg
│   └── ...
└── 5_delamination/
    ├── img_001.jpg
    └── ...
```

### Dataset Statistics

Recommended sizes:
```
Class             Min Images   Recommended
──────────────────────────────────────────
Normal            100          500+
Dust              100          500+
High Temperature  100          500+
Cracks            100          500+
Water Damage      100          500+
Delamination      100          500+
──────────────────────────────────────────
Total             600          3000+
```

## Setup for Training

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

### Setup W&B (Experiment Tracking)

Optional but recommended:

```bash
# Install W&B
pip install wandb

# Login
wandb login

# Provide your API key when prompted
```

### Configure W&B (Optional)

Edit `.env`:
```env
WANDB_PROJECT=solar_panel_fault_detection
WANDB_ENTITY=your_username
WANDB_MODE=online  # or offline for offline mode
```

## Training a Model

### Basic Training

```python
from pathlib import Path
from solar_fault_detector.config import Config
from solar_fault_detector.training import Trainer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load configuration
config = Config()

# Create data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    **config.augmentation.__dict__
)

# Load dataset
train_data = train_datagen.flow_from_directory(
    str(config.data.data_dir),
    target_size=config.model.img_size,
    batch_size=config.model.batch_size,
    class_mode='categorical'
)

val_datagen = ImageDataGenerator(rescale=1./255)
val_data = val_datagen.flow_from_directory(
    str(config.data.data_dir),
    target_size=config.model.img_size,
    batch_size=config.model.batch_size,
    class_mode='categorical'
)

# Create trainer
trainer = Trainer(
    config=config,
    model_type="cnn",
    num_ensemble_models=1
)

# Train
model = trainer.train(
    train_dataset=train_data,
    val_dataset=val_data
)

print("✅ Training complete!")
```

### Training with Ensemble

```python
from solar_fault_detector.training import Trainer

trainer = Trainer(
    config=config,
    model_type="ensemble",
    num_ensemble_models=3  # Use 3 models
)

model = trainer.train(
    train_dataset=train_data,
    val_dataset=val_data
)
```

### Custom Training Loop

```python
from solar_fault_detector.config import Config
from solar_fault_detector.models import ModelFactory
from solar_fault_detector.training import Evaluator
from solar_fault_detector.monitoring import WandbTracker

config = Config()

# Build model
model_wrapper = ModelFactory.create("cnn", config.model)
model = model_wrapper.build()
model_wrapper.compile()

# Setup monitoring
tracker = WandbTracker(
    training_config=config.training,
    model_config=config.model
)
tracker.start()

# Custom training
for epoch in range(config.model.epochs):
    # Train
    history = model.fit(
        train_data,
        epochs=1,
        validation_data=val_data,
        verbose=1
    )
    
    # Log metrics
    tracker.log_metrics({
        'loss': history.history['loss'][0],
        'val_loss': history.history['val_loss'][0],
        'accuracy': history.history['accuracy'][0],
        'val_accuracy': history.history['val_accuracy'][0],
    }, step=epoch)

# Finish
model_wrapper.save(config.model.best_model_path)
tracker.finish()

print(f"✅ Model saved to {config.model.best_model_path}")
```

---

## Hyperparameter Tuning

### Grid Search

```python
from itertools import product

# Define parameter grid
learning_rates = [1e-4, 5e-4, 1e-3]
batch_sizes = [8, 16, 32]

results = []

for lr, batch_size in product(learning_rates, batch_sizes):
    config = Config()
    config.model.learning_rate = lr
    config.model.batch_size = batch_size
    
    # Train model
    trainer = Trainer(config)
    model = trainer.train(train_data, val_data)
    
    # Evaluate
    val_loss, val_acc = model.evaluate(val_data)
    
    results.append({
        'lr': lr,
        'batch_size': batch_size,
        'val_loss': val_loss,
        'val_acc': val_acc
    })

# Find best
best = max(results, key=lambda x: x['val_acc'])
print(f"Best: LR={best['lr']}, Batch={best['batch_size']}, Acc={best['val_acc']:.3f}")
```

### Random Search

```python
import numpy as np

# Define parameter distributions
params = {
    'learning_rate': [10**x for x in np.linspace(-4, -2, 10)],
    'dropout': np.linspace(0.2, 0.8, 10),
    'l2_regularization': [10**x for x in np.linspace(-5, -2, 10)],
}

# Random sample and train
n_trials = 20
best_acc = 0

for trial in range(n_trials):
    # Sample random parameters
    config = Config()
    config.model.learning_rate = np.random.choice(params['learning_rate'])
    
    # Train
    trainer = Trainer(config)
    model = trainer.train(train_data, val_data)
    
    # Evaluate
    val_acc = model.evaluate(val_data)[1]
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_config = config
        print(f"✅ Trial {trial}: Better model found (Acc: {val_acc:.3f})")
```

---

## Monitoring Training

### Using W&B Dashboard

```python
# Start training
python -m your_training_script.py

# Open W&B link (printed to console)
# View live training metrics
# Compare runs
# Save best model
```

### Local Tensorboard

```bash
# Train with TensorBoard callback
tensorboard --logdir=./logs

# Open browser to http://localhost:6006
```

---

## Troubleshooting Training

### Issue: Training Loss Not Decreasing

**Possible causes:**
- Learning rate too high/low
- Data not properly normalized
- Model architecture issue
- Bad initialization

**Solution:**
```python
# Try lower learning rate
config.model.learning_rate = 1e-5

# Or use learning rate scheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau

callback = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,        # Reduce by 50%
    patience=3,
    min_lr=1e-6
)
```

### Issue: Model Overfitting

**Symptoms:**
- Training accuracy high, validation accuracy low
- Widening gap between train and val loss

**Solution:**
```python
# Increase dropout
config.model.dropout = 0.7

# Add more data augmentation
config.augmentation.zoom_range = 0.4
config.augmentation.rotation_range = 45

# Use L2 regularization
# Or early stopping (already implemented)
```

### Issue: GPU Out of Memory

**Solution:**
```python
# Reduce batch size
config.model.batch_size = 4  # From 16

# Or use CPU
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
```

---

## Evaluation

### Evaluate on Test Set

```python
from solar_fault_detector.training import Evaluator
from solar_fault_detector.monitoring import WandbTracker

# Load model
model = tf.keras.models.load_model(config.model.best_model_path)

# Create evaluator
tracker = WandbTracker(config.training, config.model)
evaluator = Evaluator(tracker=tracker)

# Evaluate
metrics = evaluator.evaluate(
    model,
    test_data,
    log_to_wandb=True
)

print("Test Results:")
print(f"Loss: {metrics['loss']:.4f}")
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Generate Reports

```python
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Get predictions
y_true = []
y_pred = []

for images, labels in test_data:
    preds = model.predict(images)
    y_pred.extend(np.argmax(preds, axis=1))
    y_true.extend(np.argmax(labels, axis=1))

# Classification report
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(cm)
```

---

## Model Export

### Save Model

```python
# Keras format (recommended)
model.save('artifacts/models/my_model.h5')

# SavedModel format
model.save('artifacts/models/my_model')

# TFLite (mobile)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Upload to Hugging Face

```bash
huggingface-cli login
huggingface-cli upload YourUsername/model-name model.h5
```

---

## Best Practices

1. ✅ **Always use validation set** to monitor overfitting
2. ✅ **Use early stopping** to prevent wasted computation
3. ✅ **Track experiments** with W&B for reproducibility
4. ✅ **Save best model** during training
5. ✅ **Test on held-out set** after training
6. ✅ **Check data augmentation** is applied correctly
7. ✅ **Use learning rate scheduler** for better convergence
8. ✅ **Log hyperparameters** for reproducibility

