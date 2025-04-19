# Solar Panel Fault Detector

This is a deployment package for the Solar Panel Fault Detector model.

## Overview

The Solar Panel Fault Detector is a deep learning model that can identify various types of faults in solar panels from images. It can classify solar panels into the following categories:

- Bird-drop: Solar panels with bird droppings
- Clean: Solar panels with no faults
- Dusty: Solar panels covered with dust
- Electrical-damage: Solar panels with electrical damage
- Physical-damage: Solar panels with physical damage
- Snow-covered: Solar panels covered with snow

## Contents

- `model/`: The trained TensorFlow model
- `label_mapping.json`: Mapping of class indices to class names
- `inference.py`: Inference script with the SolarPanelFaultDetector class
- `app.py`: Gradio web interface for the model
- `app_fastapi.py`: FastAPI API for the model
- `requirements.txt`: Required Python packages
- `Dockerfile`: Docker configuration for containerization
- `docker-compose.yml`: Docker Compose configuration for running the services

## Installation

### Local Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/solar-panel-fault-detector.git
   cd solar-panel-fault-detector
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

### Docker Installation

1. Build and run the Docker containers:
   ```
   docker-compose up -d
   ```

## Usage

### Command Line Interface

You can use the model from the command line:

```python
from inference import SolarPanelFaultDetector

# Initialize detector
detector = SolarPanelFaultDetector("model", "label_mapping.json")

# Predict on a single image
predictions = detector.predict("path/to/image.jpg")
print(predictions)

# Get top prediction
top_class, confidence = detector.classify_image("path/to/image.jpg")
print(f"Prediction: {top_class} (confidence: {confidence:.4f})")
```

### Web Interface

The web interface is available at http://localhost:7860 when running with Docker, or you can start it manually:

```
python app.py
```

### API

The API is available at http://localhost:8000 when running with Docker, or you can start it manually:

```
python app_fastapi.py
```

#### API Endpoints

- `GET /`: Root endpoint with API information
- `GET /health`: Health check endpoint
- `GET /info`: Model information
- `POST /predict`: Predict fault type for an image
- `POST /predict_batch`: Predict fault types for multiple images
- `POST /predict_base64`: Predict fault type for a base64-encoded image

## Model Information

- Architecture: EfficientNetB3 with custom classification head
- Input shape: 300x300x3 (RGB image)
- Output: 6 classes (Bird-drop, Clean, Dusty, Electrical-damage, Physical-damage, Snow-covered)
- Performance:
  - Accuracy: ~50%
  - Top-3 Accuracy: ~80%

## API Reference

### SolarPanelFaultDetector Class

The `SolarPanelFaultDetector` class provides the following methods:

- `predict(image)`: Returns a dictionary of class probabilities
- `predict_batch(images)`: Predicts on a batch of images
- `classify_image(image, threshold=0.5)`: Returns the most likely class and confidence
- `get_top_k_predictions(image, k=3)`: Returns the top k predictions

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The model was trained on a dataset of solar panel images with various fault types.
- The model architecture is based on EfficientNet by Google.
