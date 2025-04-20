# Solar Panel Fault Detection Model Architecture

## Base Model: EfficientNetB0

### Model Overview
- Architecture: EfficientNetB0 with custom top layers
- Input Shape: (224, 224, 3)
- Output Classes: 6

### Layer Structure

```
1. Base Model (EfficientNetB0)
   - Pre-trained on ImageNet
   - Weights frozen up to block 6

2. Custom Top Layers
   - Global Average Pooling 2D
   - Dropout (0.3)
   - Dense Layer (512 units, ReLU activation)
   - Dropout (0.4)
   - Dense Layer (6 units, Softmax activation)

3. Training Configuration
   - Optimizer: Adam (lr=0.001)
   - Loss: Categorical Crossentropy
   - Metrics: Accuracy, Precision, Recall, F1-Score

4. Data Augmentation
   - Random rotation (±15°)
   - Random zoom (0.9-1.1)
   - Random horizontal flip
   - Random brightness adjustment (±10%)
```

## Performance Metrics

### Training Parameters
- Batch Size: 32
- Epochs: 50
- Early Stopping: Patience 10
- Learning Rate Schedule: ReduceLROnPlateau

### Hardware Requirements
- Minimum RAM: 8GB
- Recommended GPU: 4GB VRAM
- CPU Mode: Supported but slower inference

## Model Optimization

### Quantization
- Post-training quantization applied
- Int8 precision for inference
- 75% size reduction from original

### Performance Benchmarks
- Average Inference Time: ~150ms
- Batch Processing: Up to 32 images
- Memory Footprint: 45MB (quantized)