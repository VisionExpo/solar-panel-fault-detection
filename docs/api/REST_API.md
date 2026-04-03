# 📡 REST API Documentation

Complete API reference for the Solar Panel Fault Detection system.

## Base URL

**Development**: `http://localhost:8000`  
**Production**: `https://<your-render-app>.onrender.com`

---

## Authentication

Currently, the API is open (no authentication). For production, consider adding:
- API Key validation
- Rate limiting
- HTTPS enforcement

---

## Endpoints

### 1. Health Check

Check if the service is running and model is loaded.

**Endpoint**: `GET /health`

**Request**:
```bash
curl http://localhost:8000/health
```

**Response** (200 OK):
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**Response** (Model not loaded):
```json
{
  "status": "model_not_ready",
  "model_loaded": false
}
```

---

### 2. Predict - Single Image

Analyze a single solar panel image and return fault classification.

**Endpoint**: `POST /predict`

**Content-Type**: `multipart/form-data`

**Parameters**:
- `file` (required): Image file (JPG or PNG)
  - Max size: 25MB (configurable)
  - Supported formats: `.jpg`, `.jpeg`, `.png`

**Request** (cURL):
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@path/to/solar_panel.jpg" \
  -H "accept: application/json"
```

**Request** (Python):
```python
import requests

with open("solar_panel.jpg", "rb") as img:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": img}
    )
print(response.json())
```

**Request** (JavaScript):
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

const response = await fetch('http://localhost:8000/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

**Response** (200 OK):
```json
{
  "image": "solar_panel.jpg",
  "predicted_class": 2,
  "confidence": 0.9847,
  "probabilities": [
    0.0023,
    0.0089,
    0.9847,
    0.0031,
    0.0008,
    0.0002
  ]
}
```

**Response Fields**:
- `image` (string): Uploaded filename
- `predicted_class` (integer): Class index (0-5)
  - `0`: Normal
  - `1`: Dust
  - `2`: High Temperature
  - `3`: Cracks
  - `4`: Water Damage
  - `5`: Delamination
- `confidence` (float): Probability of predicted class (0-1)
- `probabilities` (array): Confidence for all 6 classes

**Error Responses**:

**400 Bad Request** - Invalid file format:
```json
{
  "detail": "Unsupported file type. Upload a JPG or PNG image."
}
```

**413 Payload Too Large** - File exceeds size limit:
```json
{
  "detail": "File too large. Maximum size is 25MB."
}
```

**503 Service Unavailable** - Model not loaded:
```json
{
  "detail": "Model not loaded. Service unavailable. Check server logs."
}
```

**500 Internal Server Error** - Processing error:
```json
{
  "detail": "Error processing image: [error details]"
}
```

---

## HTTP Status Codes

| Code | Meaning | Scenario |
|------|---------|----------|
| 200 | OK | Prediction successful |
| 400 | Bad Request | Invalid file format or parameters |
| 413 | Payload Too Large | File exceeds maximum size |
| 503 | Service Unavailable | Model not loaded |
| 500 | Internal Server Error | Server error during processing |

---

## Rate Limiting (Future)

Currently no rate limiting, but recommended for production:

- Per IP: 100 requests/minute
- Per API Key: 1000 requests/minute
- Burst: 10 requests/second

---

## Response Times

Typical response times (varies by hardware):

| Operation | Time |
|-----------|------|
| Image upload | 100-500ms |
| Model inference | 50-150ms |
| Total request | 200-800ms |

---

## Usage Examples

### Example 1: Batch Process Multiple Images

```python
import requests
from pathlib import Path

image_dir = Path("test_images")

for image_file in image_dir.glob("*.jpg"):
    with open(image_file, "rb") as f:
        response = requests.post(
            "http://localhost:8000/predict",
            files={"file": f}
        )
    
    result = response.json()
    print(f"{image_file.name}: Class {result['predicted_class']} "
          f"({result['confidence']:.1%})")
```

### Example 2: Web Application Integration

```javascript
async function predictFault() {
  const fileInput = document.getElementById('imageInput');
  const file = fileInput.files[0];
  
  const formData = new FormData();
  formData.append('file', file);
  
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    const classNames = [
      'Normal', 'Dust', 'High Temperature', 
      'Cracks', 'Water Damage', 'Delamination'
    ];
    
    document.getElementById('result').innerHTML = `
      <p>Predicted: ${classNames[data.predicted_class]}</p>
      <p>Confidence: ${(data.confidence * 100).toFixed(2)}%</p>
    `;
  } catch (error) {
    console.error('Error:', error);
  }
}
```

### Example 3: Command Line Tool

```bash
#!/bin/bash

API_URL="http://localhost:8000/predict"
IMAGE_FILE=$1

if [ -z "$IMAGE_FILE" ]; then
  echo "Usage: $0 <image_path>"
  exit 1
fi

result=$(curl -s -X POST "$API_URL" \
  -F "file=@$IMAGE_FILE")

echo "$result" | python -m json.tool
```

---

## API Documentation (Auto-generated)

Visit the interactive API docs:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## Error Handling Best Practices

```python
import requests
from requests.exceptions import RequestException

def predict_with_retry(image_path, max_retries=3):
    url = "http://localhost:8000/predict"
    
    for attempt in range(max_retries):
        try:
            with open(image_path, "rb") as f:
                response = requests.post(
                    url,
                    files={"file": f},
                    timeout=30
                )
            
            if response.status_code == 200:
                return response.json()
            
            elif response.status_code == 503:
                print(f"Model not ready, retrying... ({attempt + 1}/{max_retries})")
                time.sleep(2 ** attempt)  # Exponential backoff
            
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        
        except RequestException as e:
            print(f"Request failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                raise
    
    raise Exception("Failed after all retries")
```

