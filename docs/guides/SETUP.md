# 🔧 Setup & Installation Guide

Complete instructions for setting up the Solar Panel Fault Detection system.

## Prerequisites

- **Python**: 3.9 or higher
- **Git**: For version control
- **Virtual Environment**: `venv`, `conda`, or `virtualenv`
- **Storage**: ~2GB for models and datasets
- **GPU (Optional)**: NVIDIA GPU with CUDA support for faster training

## Installation Steps

### 1. Clone Repository

```bash
git clone https://github.com/VisionExpo/solar-panel-fault-detection.git
cd solar-panel-fault-detection
```

### 2. Create Virtual Environment

**Using venv (recommended):**
```bash
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

**Using conda:**
```bash
conda create -n solar-detector python=3.10
conda activate solar-detector
```

### 3. Install Dependencies

**For production (inference only):**
```bash
pip install -r requirements-prod.txt
pip install -e .
```

**For development (with training):**
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. Download Model

The model is automatically downloaded from Hugging Face during deployment. For local use:

```bash
python -m solar_fault_detector.utils.download_model
```

### 5. Verify Installation

```bash
# Check that all packages are installed
python -c "import solar_fault_detector; print('✅ Installation successful!')"

# Run health check
python -m pytest tests/ -v
```

---

## Environment Configuration

### Create `.env` file

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
# Experiment Tracking
WANDB_API_KEY=your_wandb_api_key
WANDB_ENTITY=your_username
WANDB_PROJECT=solar_panel_fault_detection

# GPU/Device
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO
```

---

## Verify Installation

```bash
# Test the import
python -c "
from solar_fault_detector.config.config import Config
from solar_fault_detector.models.factory import ModelFactory
print('✅ All imports successful!')
"

# Test FastAPI app
python -m uvicorn apps.api.fastapi_app:app --reload

# Test in browser: http://localhost:8000/docs
```

---

## Docker Setup

### Build Docker Image

```bash
docker build -t solar-panel-detector .
```

### Run Container

```bash
docker run -p 5000:5000 solar-panel-detector
```

### Verify Container

```bash
curl http://localhost:5000/health
```

---

## GPU Setup (Optional)

### NVIDIA CUDA

```bash
# Install TensorFlow with GPU support
pip install tensorflow[and-cuda]

# Verify GPU is detected
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Apple Metal Acceleration (macOS)

```bash
# TensorFlow will automatically use Metal GPU acceleration if available
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

---

## Troubleshooting

### Issue: ImportError for solar_fault_detector

**Solution**: Make sure you're in the correct directory and ran `pip install -e .`

```bash
# Verify installation
pip show solar-fault-detector
```

### Issue: CUDA/GPU not detected

**Solution**: Reinstall TensorFlow with GPU support:

```bash
pip uninstall -y tensorflow
pip install tensorflow[and-cuda]
```

### Issue: Model download fails

**Solution**: Manually download from Hugging Face:

```bash
python -m solar_fault_detector.utils.download_model
```

### Issue: Port 5000 already in use

**Solution**: Use a different port:

```bash
python -m uvicorn apps.api.fastapi_app:app --port 8001
```

---

## Next Steps

- 📖 Read [Quick Start Guide](QUICKSTART.md)
- 🏗️ Learn about [System Architecture](../architecture/ARCHITECTURE.md)
- 🚀 Check [Deployment Guide](DEPLOYMENT.md) for production setup

