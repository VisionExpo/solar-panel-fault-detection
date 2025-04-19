# Solar Panel Fault Detection

An end-to-end deep learning system for detecting faults in solar panels using computer vision. The system can identify six different categories of solar panel conditions:
- Bird droppings
- Clean panels
- Dusty panels
- Electrical damage
- Physical damage
- Snow coverage

## Features

- **Advanced Model Architecture**: Uses EfficientNetB0 with custom top layers for efficient and accurate fault detection
- **Data Pipeline**: Automated data preparation with advanced augmentation techniques
- **Model Monitoring**: Real-time performance monitoring and metrics tracking
- **Optimization**: Multiple optimization techniques including TensorRT, quantization, and graph optimization
- **API Endpoints**: RESTful API with both single image and batch prediction capabilities
- **Performance Dashboard**: Real-time monitoring of model performance and resource usage

## Project Structure

```
├── src/
│   └── solar_panel_detector/
│       ├── components/      # Core components
│       ├── config/         # Configuration management
│       ├── utils/          # Utility functions
│       └── pipeline/       # Training and inference pipelines
├── scripts/               # Training and evaluation scripts
├── tests/                # Test suites
├── artifacts/            # Model artifacts and metrics
└── Faulty_solar_panel/  # Dataset directory
```

## Setup

1. Create a virtual environment:
```bash
make setup
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

3. Install additional dependencies (optional):
```bash
pip install -e .[dev]  # Install development dependencies
```

## Training

1. Run the training pipeline:
```bash
make train
```

2. Run hyperparameter optimization:
```bash
python scripts/optimize_model.py
```

3. Evaluate model performance:
```bash
make evaluate
```

## Deployment

### Local Development

1. Run the Flask application:
```bash
python app.py
```

2. Access the API endpoints:
- Health check: `GET /health`
- Single prediction: `POST /predict`
- Batch prediction: `POST /batch_predict`
- Metrics: `GET /metrics`
- Dashboard: `GET /dashboard`

### Docker Deployment

1. Build the Docker image:
```bash
make docker-build
```

2. Run the container:
```bash
make docker-run
```

### Render Deployment

1. Fork this repository

2. Connect to Render:
   - Create a new Web Service
   - Connect your repository
   - Choose "Docker" as the environment
   - Use the following environment variables:
     - `PORT`: 10000
     - `GOOGLE_API_KEY`: Your Google API key (for data collection)
     - `WANDB_API_KEY`: Your Weights & Biases API key
     - `MLFLOW_TRACKING_URI`: MLflow tracking URI

## API Documentation

### Single Image Prediction

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/image.jpg"
```

Response:
```json
{
  "prediction": "Clean",
  "confidence": 0.95,
  "inference_time_ms": 150,
  "top_3_predictions": [
    {"class": "Clean", "confidence": 0.95},
    {"class": "Dusty", "confidence": 0.03},
    {"class": "Bird-drop", "confidence": 0.02}
  ]
}
```

### Batch Prediction

```bash
curl -X POST http://localhost:5000/batch_predict \
  -F "images=@/path/to/image1.jpg" \
  -F "images=@/path/to/image2.jpg"
```

Response:
```json
{
  "results": [
    {
      "prediction": "Clean",
      "confidence": 0.95,
      "top_3_predictions": [...]
    },
    {
      "prediction": "Dusty",
      "confidence": 0.87,
      "top_3_predictions": [...]
    }
  ],
  "inference_time_ms": 250,
  "batch_size": 2
}
```

## Performance Monitoring

Access the performance dashboard at `http://localhost:5000/dashboard` to view:
- Inference time distribution
- Predictions by class
- Resource usage (CPU, Memory, GPU)
- Batch size distribution

## Model Performance

Current model performance metrics:
- Average inference time: ~150ms
- P95 inference time: ~200ms
- Throughput: ~200 images/second
- Accuracy: >90% (varies by class)

## Contributing

1. Fork the repository
2. Create your feature branch
3. Run tests: `make test`
4. Submit a pull request

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details