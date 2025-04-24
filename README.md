# 🔍 Solar Panel Fault Detection System

<div align="center">

![Version](https://img.shields.io/badge/version-1.0-blue?style=flat)
![Status](https://img.shields.io/badge/status-active-success?style=flat)
![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-009688?style=flat&logo=fastapi&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-3.x-F9D371?style=flat&logo=gradio&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployment-46E3B7?style=flat&logo=render&logoColor=white)
![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat)
![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-8A2BE2?style=flat&logo=git-lfs&logoColor=white)

</div>

A deep learning system for detecting and classifying faults in solar panels using computer vision and machine learning.

## ✨ Features

- 🔍 **Real-time Fault Detection**: Identify solar panel faults from images with high accuracy
- 📊 **Interactive Web Interface**: User-friendly Gradio interface for easy interaction
- 🚀 **REST API**: FastAPI-powered API for integration with other systems
- 📈 **Visualization Tools**: Comprehensive visualizations for EDA and model performance
- 🎯 **Multi-class Classification**: Support for 6 different fault categories
- ⚡ **Optimized Inference**: Fast and efficient GPU-accelerated model inference
- 📱 **Responsive Design**: Mobile-friendly interface for on-the-go inspections
- 🔄 **Batch Processing**: Process multiple images in batch mode

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/solar-panel-fault-detection.git
cd solar-panel-fault-detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start both the API and web interface:
```bash
python start_apps.py
```

This will open:
- Web Interface: http://localhost:8501
- API Documentation: http://localhost:5000/docs

## 🧩 Supported Fault Categories

![Bird Droppings](https://img.shields.io/badge/Category-Bird%20Droppings-yellow?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiNGRkQxMDAiLz48L3N2Zz4=)
![Clean](https://img.shields.io/badge/Category-Clean-brightgreen?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiM0Q0FGNTAiLz48L3N2Zz4=)
![Dusty](https://img.shields.io/badge/Category-Dusty-lightgrey?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiM5RTlFOUUiLz48L3N2Zz4=)
![Electrical Damage](https://img.shields.io/badge/Category-Electrical%20Damage-red?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiNGNDQzMzYiLz48L3N2Zz4=)
![Physical Damage](https://img.shields.io/badge/Category-Physical%20Damage-orange?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiNGRjk4MDAiLz48L3N2Zz4=)
![Snow Coverage](https://img.shields.io/badge/Category-Snow%20Coverage-blue?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiMyMTk2RjMiLz48L3N2Zz4=)

1. 🦅 **Bird droppings**: Solar panel with bird droppings on the surface
2. ✨ **Clean panels**: Solar panel with no visible faults or issues
3. 🌫️ **Dusty panels**: Solar panel covered with dust or dirt
4. ⚡ **Electrical damage**: Solar panel with electrical damage
5. 💢 **Physical damage**: Solar panel with physical damage
6. ❄️ **Snow coverage**: Solar panel covered with snow

## 🚀 Tech Stack

<div align="center">

| Technology | Purpose |
|------------|---------|
| ![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat&logo=python&logoColor=white) | Core language |
| ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow&logoColor=white) | Deep learning framework |
| ![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-009688?style=flat&logo=fastapi&logoColor=white) | API framework |
| ![Gradio](https://img.shields.io/badge/Gradio-3.x-F9D371?style=flat&logo=gradio&logoColor=white) | Web interface |
| ![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white) | Image processing |
| ![NumPy](https://img.shields.io/badge/NumPy-1.24.x-013243?style=flat&logo=numpy&logoColor=white) | Numerical computing |
| ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.x-11557c?style=flat) | Data visualization |
| ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white) | Containerization |
| ![Render](https://img.shields.io/badge/Render-Deployment-46E3B7?style=flat&logo=render&logoColor=white) | Cloud deployment |

</div>

## 🏗️ System Architecture

```ascii
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Web Interface  │────▶│  Backend API    │────▶│  ML Model       │
│  (Gradio)       │     │  (FastAPI)      │     │  (TensorFlow)   │
│                 │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └─────────────────┘
                               │
                               ▼
                 ┌─────────────────────────────┐
                 │                             │
                 │  Image Processing Pipeline  │
                 │  (OpenCV)                   │
                 │                             │
                 └─────────────────────────────┘
```

- **Frontend**: Streamlit web interface
- **Backend**: Flask RESTful API
- **Model**: EfficientNetB0 with custom top layers
- **Monitoring**: Real-time performance tracking
- **Storage**: Local file system + SQLite for metrics

### Technologies Used

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-009688?style=flat&logo=fastapi&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-3.x-F9D371?style=flat&logo=gradio&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24.x-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.x-11557c?style=flat)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployment-46E3B7?style=flat&logo=render&logoColor=white)

## Development Setup

1. Install development dependencies:
```bash
pip install -e ".[dev]"
```

2. Install pre-commit hooks:
```bash
pre-commit install
```

3. Run tests:
```bash
pytest tests/
```

## API Documentation

### Single Image Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/image.jpg"
```

### Batch Prediction

```bash
curl -X POST http://localhost:5000/batch_predict \
  -F "images=@/path/to/image1.jpg" \
  -F "images=@/path/to/image2.jpg"
```

### Performance Metrics

```bash
curl http://localhost:5000/metrics
```

## 📊 Model Performance

- **Accuracy**: ~85% on test set
- **F1 Score**: ~0.83 across all classes
- **Inference Time**: ~150ms per image
- **Model Architecture**: EfficientNetB3 with custom top layers
- **Input Size**: 384x384 pixels
- **Batch processing**: Up to 32 images
- **GPU utilization**: Optimized for NVIDIA GPUs
- **CPU fallback**: Automatically detects available hardware

## 🧠 Training Process

The model was trained on the [Solar Augmented Dataset](https://www.kaggle.com/datasets/gitenavnath/solar-augmented-dataset) from Kaggle, which contains images of solar panels in various conditions.

### Dataset Details

- **Classes**: 6 (Bird-drop, Clean, Dusty, Electrical-damage, Physical-damage, Snow-covered)
- **Images**: ~3,000 images across all classes
- **Split**: 70% training, 15% validation, 15% testing
- **Augmentation**: Rotation, shift, shear, zoom, and horizontal flip

### Training Strategy

1. **Transfer Learning**: Started with EfficientNetB3 pre-trained on ImageNet
2. **Initial Training**: Trained the top layers with the base model frozen
3. **Fine-tuning**: Unfroze the last 30 layers and fine-tuned with a lower learning rate
4. **Regularization**: Used dropout (0.3) and early stopping to prevent overfitting

### Training Parameters

- **Batch Size**: 32
- **Image Size**: 384x384 pixels
- **Initial Learning Rate**: 0.001
- **Fine-tuning Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Loss Function**: Categorical Cross-Entropy

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:
```bash
cp .env.example .env
```

Key configurations:
- `PORT`: API port (default: 5000)
- `MODEL_BATCH_SIZE`: Maximum batch size
- `ENABLE_GPU`: Enable/disable GPU acceleration

### Streamlit Configuration

Located in `.streamlit/config.toml`:
- Theme customization
- Server settings
- Security configurations

## Docker Support

1. Build the image:
```bash
docker build -t solar-panel-detector .
```

2. Run the container:
```bash
docker run -p 5000:5000 -p 8501:8501 solar-panel-detector
```

## Deployment

![Render](https://img.shields.io/badge/Render-Deployment-46E3B7?style=flat&logo=render&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![CI/CD](https://img.shields.io/badge/CI%2FCD-Automated-4CAF50?style=flat&logo=github-actions&logoColor=white)
![Git LFS](https://img.shields.io/badge/Git%20LFS-Enabled-8A2BE2?style=flat&logo=git-lfs&logoColor=white)

### Model Storage

The model files are stored externally on [Hugging Face Model Hub](https://huggingface.co/VishalGorule09/SolarPanelModel) due to GitHub's file size limitations (100MB max). The deployment process automatically downloads the model files during build.

### Render Deployment

1. Fork this repository
2. Connect to Render
3. Deploy using the provided `render.yaml` (the model will be automatically downloaded from Hugging Face)

The deployment is configured to download the model from Hugging Face Model Hub. If you want to use your own model:

1. Upload your model file to Hugging Face or another storage service
2. Update the `MODEL_URL` environment variable in `render.yaml` with your model download URL

### Configuration

#### Environment Variables

- `PORT`: The port on which the application will run (default: 7860)
- `MODEL_URL`: URL to download the model file (defaults to Hugging Face URL)

### Local Deployment

Use `waitress` (Windows) or `gunicorn` (Linux) for production:

```bash
# Windows
waitress-serve --port=5000 app:app

# Linux
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests: `pytest tests/`
4. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

- Author: Vishal Gorule
- Email: [gorulevishal984@gmail.com](mailto:gorulevishal984@gmail.com)
- GitHub: [@VisionExpo](https://github.com/VisionExpo)

---

Made with ❤️ by [Vishal Gorule](https://github.com/VisionExpo)