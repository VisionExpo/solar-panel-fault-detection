# Solar Panel Fault Detection System

A deep learning system for detecting and classifying faults in solar panels using computer vision.

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

- Python 3.8+
- TensorFlow 2.x
- Flask + Flask-RESTx
- Streamlit
- OpenCV
- Plotly
- MLflow

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

### Render Deployment

1. Fork this repository
2. Connect to Render
3. Add required environment variables
4. Deploy using the provided `render.yaml`

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

- Author: Your Name
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)