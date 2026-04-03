# ⚡ Quick Start Guide

Get the Solar Panel Fault Detection API running in 5 minutes.

## Installation (2 min)

```bash
# Clone and navigate
git clone https://github.com/VisionExpo/solar-panel-fault-detection.git
cd solar-panel-fault-detection

# Setup environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements-prod.txt
pip install -e .
```

## Start the API (1 min)

```bash
python -m uvicorn apps.api.fastapi_app:app --reload
```

Open your browser: **http://localhost:8000/docs**

## Make Your First Prediction (2 min)

### Using Python

```python
import requests
from pathlib import Path

# Prepare image
image_path = Path("test_image.jpg")

# Send request
with open(image_path, "rb") as img:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": img}
    )

# Get prediction
result = response.json()
print(f"Predicted class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### Using cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -F "file=@test_image.jpg"
```

### Using Python Requests

```bash
python -c "
import requests
response = requests.post(
    'http://localhost:8000/predict',
    files={'file': open('test_image.jpg', 'rb')}
)
print(response.json())
"
```

## Check API Status

```bash
curl http://localhost:8000/health
```

Response:
```json
{"status": "ok", "model_loaded": true}
```

---

## Next Steps

- 📚 Read full [Setup Guide](SETUP.md)
- 📖 Explore [REST API Documentation](../api/REST_API.md)
- 🚀 Deploy to [Production](DEPLOYMENT.md)
- 🔧 Configure [Settings](CONFIGURATION.md)

---

## Common Commands

| Command | Purpose |
|---------|---------|
| `python -m uvicorn apps.api.fastapi_app:app` | Start API server |
| `python -m solar_fault_detector.utils.download_model` | Download model |
| `curl http://localhost:8000/health` | Check API health |
| `docker build -t detector .` | Build Docker image |
| `docker run -p 5000:5000 detector` | Run Docker container |

---

## Troubleshooting

**Model not found?**
```bash
python -m solar_fault_detector.utils.download_model
```

**Port already in use?**
```bash
python -m uvicorn apps.api.fastapi_app:app --port 8001
```

**Import errors?**
```bash
pip install -e .
```

