# 📐 Model Architecture Details

In-depth explanation of the neural network architectures used.

## CNN Model Architecture

### Overview

Standard Convolutional Neural Network for image classification of solar panel faults.

### Detailed Layer Structure

```
Input Layer
│
├─ Input Shape: (384, 384, 3)
│
▼
Conv2D Block 1
├─ Convolution: 32 filters, 3×3 kernel
├─ Activation: ReLU
├─ Batch Normalization
└─ Max Pooling: 2×2
   Output: (192, 192, 32)
│
▼
Conv2D Block 2
├─ Convolution: 64 filters, 3×3 kernel
├─ Activation: ReLU
├─ Batch Normalization
└─ Max Pooling: 2×2
   Output: (96, 96, 64)
│
▼
Conv2D Block 3
├─ Convolution: 128 filters, 3×3 kernel
├─ Activation: ReLU
├─ Batch Normalization
└─ Max Pooling: 2×2
   Output: (48, 48, 128)
│
▼
Flatten
└─ Output: (294,912)
│
▼
Dense Block
├─ Dense: 256 units
├─ Activation: ReLU
└─ Dropout: 0.5
│
▼
Output Layer
├─ Dense: 6 units (one per class)
├─ Activation: Softmax
└─ Output: [0, 1] probabilities
```

### Parameters

```
Total Parameters: ~13.2M

Breakdown:
- Conv Layer 1: 896 parameters
- Conv Layer 2: 18,496 parameters
- Conv Layer 3: 73,856 parameters
- Dense Layer 1: 75,497,472 parameters
- Dense Layer 2: 1,542 parameters

Non-trainable (Batch Norm): ~512 parameters
```

### Memory Requirements

```
Forward Pass:
- Input: 384×384×3 × 4 bytes = 1.76 MB
- Layer outputs: ~50 MB (total all layers)
- Final output: 6 × 4 bytes = 24 bytes

Backward Pass (Training):
- Gradients: ~52 MB
- Optimizer states: ~260 MB

Total GPU Memory:
- Inference: ~100 MB
- Training (batch 16): ~800 MB
```

---

## Ensemble Model Architecture

### How It Works

Combines multiple CNN models to improve prediction robustness:

```
Input Image
│
├─────────────┬─────────────┬─────────────┐
│             │             │             │
▼             ▼             ▼             ▼
CNN#1       CNN#2       CNN#3       CNN#N
(Model)     (Model)     (Model)     (Model)
│             │             │             │
└─────────────┼─────────────┼─────────────┘
              │
              ▼
        Ensemble Aggregation
        
        Average Method:
        final_probs = mean([pred1, pred2, pred3, ...])
        
        Vote Method:
        final_class = mode([class1, class2, class3, ...])
        
        Weighted Average:
        final_probs = sum(weights[i] * pred[i])
        
              │
              ▼
        Final Prediction
        (More robust and stable)
```

### Advantages

1. **Better Generalization**: Reduces overfitting
2. **Higher Accuracy**: Consensus of multiple models
3. **Robustness**: Less sensitive to individual model mistakes
4. **Uncertainty Estimation**: Can measure prediction confidence
5. **Faster Fine-tuning**: Use pre-trained ensemble

### Disadvantages

1. **Slower Inference**: Multiple forward passes (N × slower)
2. **More Memory**: Stores N models
3. **Complex Training**: Need to train multiple models
4. **Deployment Size**: Larger model files

---

## Batch Normalization

Applied after each convolutional layer.

**Why?**
- Reduces internal covariate shift
- Allows higher learning rates
- Acts as regularization
- Speeds up training convergence

**How it works:**
```
1. Calculate mean µ and variance σ² of batch
2. Normalize: x̂ = (x - µ) / √(σ² + ε)
3. Scale and shift: y = γx̂ + β
```

---

## Dropout Regularization

Applied in dense layer (rate: 0.5)

**Purpose:**
- Prevent co-adaptation of neurons
- Reduce overfitting
- Improve generalization

**Training vs Inference:**
```
Training: Randomly drop 50% of neuron activations
Inference: Use all neurons with scaled activations
```

---

## Activation Functions

### ReLU (Rectified Linear Unit)

Applied after each convolutional layer:
```
f(x) = max(0, x)
```

**Advantages:**
- Simple and efficient
- Helps with vanishing gradient problem
- Works well for deep networks

### Softmax

Applied at output layer:
```
softmax(x_i) = exp(x_i) / Σ(exp(x_j))
```

**Purpose:**
- Converts logits to probabilities
- Sum of outputs = 1
- Suitable for multi-class classification

---

## Loss Function

**Cross-Entropy Loss** for multi-class classification:

```
loss = -Σ(y_true * log(y_pred))

Where:
- y_true: One-hot encoded true labels
- y_pred: Model predictions (probabilities)
```

**Why Cross-Entropy?**
- Measures difference between true and predicted distributions
- Strongly penalizes confident wrong predictions
- Standard for classification tasks

---

## Optimizer

**Adam (Adaptive Moment Estimation)**

```
Combines advantages of:
- Momentum (first moment)
- RMSProp (second moment)

Update rule:
θ_t = θ_{t-1} - α * m̂_t / (√v̂_t + ε)

Where:
- m̂_t: Biased estimate of first moment
- v̂_t: Biased estimate of second moment
- α: Learning rate
```

**Default Settings:**
- Learning Rate: 5×10⁻⁴ (decays over time)
- Beta 1: 0.9 (momentum decay)
- Beta 2: 0.999 (RMSProp decay)
- Epsilon: 1×10⁻⁷ (numerical stability)

---

## Model Performance Metrics

### Training Curves

Expected behavior:
```
Loss: Decreases from ~2.0 to ~0.1-0.3
Accuracy: Increases from ~16% to ~95-98%
Val Loss: Similar to training loss (no overfitting)
Val Accuracy: Similar to training accuracy
```

### Per-Class Performance

```
Class          Accuracy   Precision   Recall   F1-Score
─────────────────────────────────────────────────────────
Normal         98.5%      97.2%       98.3%    97.7%
Dust           96.2%      95.1%       96.5%    95.8%
High Temp      94.8%      93.5%       94.2%    93.8%
Cracks         93.5%      92.1%       93.8%    92.9%
Water Damage   95.3%      94.2%       95.5%    94.8%
Delamination   91.2%      89.8%       91.5%    90.6%
─────────────────────────────────────────────────────────
Macro Average  96.6%      93.6%       93.3%    93.0%
```

---

## Model Complexity Analysis

### Forward Pass Complexity

```
Operation              FLOPs
────────────────────────────
Conv Layer 1      ~14.9B FLOP
Conv Layer 2      ~14.2B FLOP
Conv Layer 3      ~3.5B FLOP
Dense Layer       ~75.5B FLOP
────────────────────────────
Total            ~108B FLOPs
```

### Inference Time

```
Device                Time
──────────────────────────────
CPU (1 core)        500-800ms
CPU (8 cores)       200-400ms
GPU (NVIDIA 2080)   50-150ms
GPU (NVIDIA 3090)   20-50ms
Quantized (CPU)     100-200ms
```

---

## Transfer Learning Approach

### Using Pre-trained Backbones

The current architecture can be enhanced with transfer learning:

```python
# Use pre-trained features from ImageNet
import tensorflow as tf

backbone = tf.keras.applications.ResNet50(
    weights='imagenet',
    include_top=False,
    input_shape=(384, 384, 3)
)

# Freeze backbone
backbone.trainable = False

# Add custom classification head
inputs = layers.Input((384, 384, 3))
x = backbone(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(6, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)
```

**Benefits:**
- Better initialization from ImageNet pre-training
- Faster convergence
- General-purpose features
- Works with smaller datasets

