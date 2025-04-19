# Solar Panel Fault Detection System

An end-to-end deep learning system for detecting and classifying faults in solar panels using computer vision.

## Features

- Multi-class fault detection (Bird-drop, Clean, Dusty, Electrical-damage, Physical-Damage, Snow-Covered)
- Automated data collection and augmentation pipeline
- Advanced model architecture using EfficientNetB0
- MLflow and Weights & Biases integration for experiment tracking
- Optimized inference with batch processing and caching
- REST API with both single and batch prediction endpoints
- Comprehensive test suite
- Docker containerization
- Ready for deployment on Render

## Project Structure

```
src/
├── solar_panel_detector/
│   ├── components/
│   │   ├── data_ingestion.py     # Data collection and downloading
│   │   ├── data_preparation.py   # Data preprocessing and augmentation
│   │   └── model.py             # Model architecture and training
│   ├── config/
│   │   └── configuration.py     # Configuration management
│   ├── pipeline/
│   │   └── train_pipeline.py    # Training orchestration
│   └── utils/
│       └── logger.py            # Custom logging
tests/
├── test_model.py               # Model tests
└── test_data_and_api.py       # Data pipeline and API tests
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/solar-panel-fault-detection.git
cd solar-panel-fault-detection
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training Pipeline

1. Configure the training parameters in `src/solar_panel_detector/config/configuration.py`

2. Run the training pipeline:
```bash
python -m src.solar_panel_detector.pipeline.train_pipeline
```

The training process includes:
- Automated data collection up to 5000 images per category
- Advanced data augmentation
- Transfer learning with EfficientNetB0
- Hyperparameter optimization
- Metrics tracking with MLflow and W&B

## Model Monitoring and Analysis

### Training Metrics
- Loss vs. Epochs
- Accuracy vs. Epochs
- Validation metrics
- Per-class F1 scores
- Confusion matrix

### Available in MLflow:
- Model artifacts
- Training parameters
- Performance metrics
- Learning curves

### Weights & Biases Integration:
- Real-time training monitoring
- Experiment comparison
- Model versioning

## API Endpoints

### Single Image Prediction
```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@path/to/image.jpg"
```

### Batch Prediction
```bash
curl -X POST http://localhost:5000/batch_predict \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg"
```

### Health Check
```bash
curl http://localhost:5000/health
```

## Deployment

### Local Deployment with Docker

1. Build the Docker image:
```bash
docker build -t solar-panel-detector .
```

2. Run the container:
```bash
docker run -p 5000:10000 solar-panel-detector
```

### Deployment on Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the following:
   - Environment: Docker
   - Build Command: (automatic with Dockerfile)
   - Start Command: (automatic with Dockerfile)
   - Environment Variables:
     - PORT=10000
     - WANDB_API_KEY=your_key
     - MLFLOW_TRACKING_URI=your_uri

## Performance Optimization

The system includes several optimizations:
- Model quantization for faster inference
- Response caching
- Parallel image processing
- Batch prediction support
- Efficient model loading
- Custom data augmentation pipeline

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.

## Acknowledgments

- TensorFlow team for the EfficientNet implementation
- MLflow team for experiment tracking capabilities
- Weights & Biases for advanced monitoring features