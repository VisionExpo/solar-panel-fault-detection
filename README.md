# Solar Panel Fault Detection System

![Version](https://img.shields.io/badge/version-1.0-blue?style=flat)
![Status](https://img.shields.io/badge/status-active-success?style=flat)
![License](https://img.shields.io/badge/license-Apache%202.0-blue?style=flat)

A deep learning system for detecting and classifying faults in solar panels using computer vision.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95.0-009688?style=flat&logo=fastapi&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-3.x-F9D371?style=flat&logo=gradio&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24.x-013243?style=flat&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7.x-11557c?style=flat)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deployment-46E3B7?style=flat&logo=render&logoColor=white)

## Features

- 🔍 Real-time fault detection
- 📊 Interactive web interface built with Streamlit
- 🚀 RESTful API with batch processing support
- 📈 Real-time performance monitoring
- 🎯 Support for 6 fault categories
- ⚡ GPU-accelerated inference
- 📱 Mobile-friendly interface

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

## Supported Fault Categories

![Bird Droppings](https://img.shields.io/badge/Category-Bird%20Droppings-yellow?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiNGRkQxMDAiLz48L3N2Zz4=)
![Clean](https://img.shields.io/badge/Category-Clean-brightgreen?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiM0Q0FGNTAiLz48L3N2Zz4=)
![Dusty](https://img.shields.io/badge/Category-Dusty-lightgrey?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiM5RTlFOUUiLz48L3N2Zz4=)
![Electrical Damage](https://img.shields.io/badge/Category-Electrical%20Damage-red?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiNGNDQzMzYiLz48L3N2Zz4=)
![Physical Damage](https://img.shields.io/badge/Category-Physical%20Damage-orange?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiNGRjk4MDAiLz48L3N2Zz4=)
![Snow Coverage](https://img.shields.io/badge/Category-Snow%20Coverage-blue?style=flat&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGQ9Ik0xMiAyYzUuNTIgMCAxMCA0LjQ4IDEwIDEwcy00LjQ4IDEwLTEwIDEwUzIgMTcuNTIgMiAxMiA2LjQ4IDIgMTIgMnoiIGZpbGw9IiMyMTk2RjMiLz48L3N2Zz4=)

1. 🦅 Bird droppings
2. ✨ Clean panels (no faults)
3. 🌫️ Dusty panels
4. ⚡ Electrical damage
5. 💢 Physical damage
6. ❄️ Snow coverage

## System Architecture

### Components

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

## Model Performance

- Average inference time: ~150ms
- Batch processing: Up to 32 images
- GPU utilization: Optimized for NVIDIA GPUs
- CPU fallback: Automatically detects available hardware

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

The model files are stored externally due to GitHub's file size limitations (100MB max). The deployment process automatically downloads the model files during build.

### Render Deployment

1. Fork this repository
2. Upload your model file to Google Drive or another storage service
3. Update the `MODEL_URL` environment variable in `render.yaml` with your model download URL
4. Connect to Render
5. Deploy using the provided `render.yaml`

### Environment Variables

- `PORT`: The port on which the application will run (default: 7860)
- `MODEL_URL`: URL to download the model file (required for deployment)

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
- Email: gorulevishal984@gmail.com
- GitHub: [@VisionExpo](https://github.com/VisionExpo)