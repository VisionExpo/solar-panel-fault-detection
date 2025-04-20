# Solar Panel Fault Detection API Documentation

## API Endpoints

### 1. Single Image Prediction
`POST /predict`

Analyzes a single solar panel image and returns fault detection results.

#### Request
- Content-Type: `multipart/form-data`
- Body:
  - `image`: Image file (JPG/PNG)

#### Response
```json
{
    "prediction": "Clean",
    "confidence": 0.95,
    "processing_time": "150ms",
    "fault_probabilities": {
        "Bird-drop": 0.01,
        "Clean": 0.95,
        "Dusty": 0.02,
        "Electrical-damage": 0.01,
        "Physical-Damage": 0.005,
        "Snow-Covered": 0.005
    }
}
```

### 2. Batch Prediction
`POST /batch_predict`

Processes multiple solar panel images in a single request.

#### Request
- Content-Type: `multipart/form-data`
- Body:
  - `images`: Array of image files
  - `batch_size`: (optional) Integer (default: 32)

#### Response
```json
{
    "predictions": [
        {
            "filename": "panel1.jpg",
            "prediction": "Clean",
            "confidence": 0.95
        },
        {
            "filename": "panel2.jpg",
            "prediction": "Dusty",
            "confidence": 0.87
        }
    ],
    "processing_time": "350ms"
}
```

### 3. Model Performance Metrics
`GET /metrics`

Returns current model performance statistics.

#### Response
```json
{
    "total_predictions": 1500,
    "average_inference_time": "150ms",
    "accuracy_last_24h": 0.94,
    "error_rate": 0.06,
    "system_health": {
        "gpu_utilization": "45%",
        "memory_usage": "2.1GB",
        "uptime": "5d 12h"
    }
}
```

## Authentication

API requests require an API key passed in the header:
```
Authorization: Bearer your_api_key_here
```

## Rate Limiting
- Free tier: 100 requests/hour
- Premium tier: 1000 requests/hour

## Error Codes
- 400: Bad Request (invalid image format)
- 401: Unauthorized (invalid API key)
- 429: Too Many Requests
- 500: Internal Server Error

## Sample Usage

### Python
```python
import requests

url = "http://api.solarpanel-detector.com/predict"
headers = {"Authorization": "Bearer your_api_key_here"}
files = {"image": open("panel.jpg", "rb")}

response = requests.post(url, headers=headers, files=files)
result = response.json()
print(result["prediction"])
```

### cURL
```bash
curl -X POST \
  -H "Authorization: Bearer your_api_key_here" \
  -F "image=@panel.jpg" \
  http://api.solarpanel-detector.com/predict
```