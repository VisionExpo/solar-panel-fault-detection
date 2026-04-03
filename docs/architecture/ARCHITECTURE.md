# 🏗️ System Architecture

High-level overview of the Solar Panel Fault Detection system.

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Input Sources                             │
│        (Images from cameras, sensors, files)                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            Data Preprocessing Layer                          │
│  (Validation, Resizing, Normalization, Augmentation)        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│           Model Inference Engine                            │
│  (CNN / Ensemble Models running predictions)                │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│         Post-Processing & Outputs                           │
│  (Classification, Confidence scoring, Logging)              │
└─────────────────────┬───────────────────────────────────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌────────┐   ┌────────┐   ┌────────┐
    │  API   │   │Database│   │ Logs   │
    │Response│   │ Storage│   │Tracking│
    └────────┘   └────────┘   └────────┘
```

## Core Components

### 1. Configuration Management
- **File**: `src/solar_fault_detector/config/config.py`
- **Responsibility**: Centralized settings and hyperparameters
- **Components**:
  - `ModelConfig`: Model architecture parameters
  - `DataConfig`: Dataset paths and split ratios
  - `TrainingConfig`: Training hyperparameters
  - `AugmentationConfig`: Data augmentation settings

### 2. Models
- **Location**: `src/solar_fault_detector/models/`
- **Architecture**: Abstract factory pattern
  - `BaseModel`: Abstract base class
  - `CNNModel`: Convolutional neural network
  - `EnsembleModel`: Multiple model ensemble
  - `ModelFactory`: Dynamic model creation

### 3. Training Pipeline
- **Location**: `src/solar_fault_detector/training/`
- **Components**:
  - `Trainer`: Orchestrates training workflow
  - `Evaluator`: Performance metrics computation
  - `Tuning`: Hyperparameter optimization

### 4. Inference Engine
- **Location**: `src/solar_fault_detector/inference/`
- **Types**:
  - `Predictor`: Single image inference
  - `BatchInferenceEngine`: Multiple images
  - `Model Download`: Hugging Face integration

### 5. Data Processing
- **Location**: `src/solar_fault_detector/data/`
- **Features**:
  - Image loading and validation
  - Preprocessing (resizing, normalization)
  - Data augmentation
  - Batch generation

### 6. Monitoring & Tracking
- **Location**: `src/solar_fault_detector/monitoring/`
- **Integration**:
  - Weights & Biases (W&B) experiment tracking
  - Metrics logging
  - Artifact management

### 7. Applications
- **API**: FastAPI-based REST interface
- **CLI**: Command-line tools
- **Deployment**: Docker, Render configuration

---

## Data Flow

### Training Flow

```
Raw Dataset
    │
    ▼
Data Loading
    │
    ▼
Preprocessing (Resize, Normalize)
    │
    ▼
Data Augmentation (Rotation, Flip, etc.)
    │
    ▼
Train/Val/Test Split
    │
    ▼
Model Creation (CNN or Ensemble)
    │
    ▼
Training Loop
    ├─ Compute Predictions
    ├─ Calculate Loss
    ├─ Backward Pass
    ├─ Weight Update
    └─ Validation
    │
    ▼
Model Evaluation
    │
    ▼
Best Model Save
    │
    ▼
Experiment Log (W&B)
```

### Inference Flow

```
Input Image
    │
    ▼
Validation
    │
    ├─ File format check
    ├─ Dimension check
    └─ Size validation
    │
    ▼
Preprocessing
    │
    ├─ Load image
    ├─ Resize to 384x384
    ├─ Normalize pixel values
    └─ Add batch dimension
    │
    ▼
Model Inference
    │
    ├─ Forward pass
    └─ Get predictions
    │
    ▼
Post-Processing
    │
    ├─ Get class predictions
    ├─ Calculate confidence
    └─ Map to labels
    │
    ▼
Return Results
```

---

## Model Architecture

### CNN Model

```
Input (384 x 384 x 3)
    │
    ▼
Conv2D(32, 3x3) + ReLU + BatchNorm
    │
    ▼
MaxPool(2x2)
    │
    ▼
Conv2D(64, 3x3) + ReLU + BatchNorm
    │
    ▼
MaxPool(2x2)
    │
    ▼
Conv2D(128, 3x3) + ReLU + BatchNorm
    │
    ▼
MaxPool(2x2)
    │
    ▼
Flatten
    │
    ▼
Dense(256) + ReLU + Dropout(0.5)
    │
    ▼
Dense(6, softmax)
    │
    ▼
Output [0, 1] (Class probabilities)
```

### Ensemble Model

```
Image
    │
    ├─────────────────┬─────────────────┬──────────────────┐
    │                 │                 │                  │
    ▼                 ▼                 ▼                  ▼
  CNN 1             CNN 2             CNN 3            CNN N
    │                 │                 │                  │
    └─────────────────┼─────────────────┼──────────────────┘
                      │
                      ▼
            Average Predictions
                      │
                      ▼
            Output (Ensemble Result)
```

---

## Deployment Architecture

### Development
```
Local Machine
    │
    ├─ Python Environment
    ├─ Model (downloaded from Hugging Face)
    ├─ FastAPI Server
    └─ Optional: GPU acceleration
```

### Production
```
Render Cloud Platform
    │
    ├─ Docker Container
    ├─ Python 3.10 Runtime
    ├─ FastAPI with Uvicorn
    ├─ Model Auto-Download
    └─ Auto-scaling
```

---

## API Integration Points

### REST Endpoints
- `GET /health` - Service health check
- `POST /predict` - Single image prediction
- `GET /docs` - Swagger documentation

### Integration with External Services
- **Hugging Face Hub**: Model repository
- **Weights & Biases**: Experiment tracking (optional)
- **Docker Hub**: Container registry

---

## Performance Considerations

| Component | Optimization | Impact |
|-----------|--------------|--------|
| Model | Reduced from 3.5GB to 1.2GB | 66% faster deployment |
| Inference | Single-pass CNN | ~50-100ms per image |
| Batch Processing | Vectorized operations | 10-20x speedup |
| Caching | In-memory model | <5ms inference |

---

## Security & Error Handling

### Input Validation
- File type validation (JPG, PNG only)
- File size limits
- Dimension checks

### Error Recovery
- Graceful model loading
- HTTP 503 if model unavailable
- Detailed error messages in logs

### Production Safety
- Health check endpoint
- Structured logging
- Error monitoring integration

