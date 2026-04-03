# Solar Panel Fault Detection API SDK

A comprehensive Python SDK for interacting with the Solar Panel Fault Detection REST API.

## Installation

```bash
pip install solar-fault-detection-sdk
# or
pip install git+https://github.com/your-repo/solar-panel-fault-detection.git#subdirectory=sdk
```

## Quick Start

```python
from solar_fault_detection_sdk import SolarFaultDetectionClient

# Initialize client
client = SolarFaultDetectionClient(base_url="http://your-api-server.com")

# Make a prediction
result = client.predict("path/to/solar_panel_image.jpg")
print(f"Predicted fault: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2f}")
```

## Features

- ✅ **Single Image Prediction**: Classify individual solar panel images
- ✅ **Batch Processing**: Process multiple images efficiently
- ✅ **URL Prediction**: Predict from image URLs
- ✅ **Health Monitoring**: Check API status and metrics
- ✅ **Async Support**: Asynchronous operations for high-throughput applications
- ✅ **Error Handling**: Robust error handling with retries
- ✅ **Authentication**: API key support for secure access

## API Reference

### SolarFaultDetectionClient

Main client class for interacting with the API.

#### Initialization

```python
client = SolarFaultDetectionClient(
    base_url="http://localhost:5000",  # API server URL
    api_key="your-api-key",           # Optional authentication
    timeout=30,                       # Request timeout
    retries=3                         # Retry attempts
)
```

#### Methods

##### `health_check() -> dict`

Check API server health status.

```python
health = client.health_check()
print(health)  # {"status": "healthy", "timestamp": 1234567890}
```

##### `predict(image_path, return_probabilities=True) -> dict`

Predict fault type for a single image.

```python
result = client.predict("solar_panel.jpg")
# Returns:
{
    "image": "solar_panel.jpg",
    "predicted_class": 2,
    "confidence": 0.87,
    "probabilities": [0.02, 0.05, 0.87, 0.03, 0.02, 0.01]
}
```

##### `predict_batch(image_paths, batch_size=10) -> list`

Process multiple images in batches.

```python
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
results = client.predict_batch(image_paths, batch_size=5)

for result in results:
    print(f"{result['image']}: Class {result['predicted_class']}")
```

##### `predict_url(image_url) -> dict`

Predict from an image URL.

```python
result = client.predict_url("https://example.com/solar_panel.jpg")
```

##### `get_model_info() -> dict`

Get information about the deployed model.

```python
info = client.get_model_info()
print(f"Model: {info['model_name']}, Classes: {info['num_classes']}")
```

##### `get_metrics() -> dict`

Get API performance metrics.

```python
metrics = client.get_metrics()
print(f"Total predictions: {metrics['total_predictions']}")
```

### AsyncSolarFaultDetectionClient

Asynchronous version for high-performance applications.

```python
import asyncio
from solar_fault_detection_sdk import AsyncSolarFaultDetectionClient

async def main():
    client = AsyncSolarFaultDetectionClient()

    # Async prediction
    result = await client.predict_async("image.jpg")
    print(result)

    # Cleanup
    await client.close()

asyncio.run(main())
```

## Error Handling

The SDK includes comprehensive error handling:

```python
from solar_fault_detection_sdk import SolarFaultDetectionClient
import requests

client = SolarFaultDetectionClient()

try:
    result = client.predict("image.jpg")
except requests.exceptions.ConnectionError:
    print("Cannot connect to API server")
except requests.exceptions.Timeout:
    print("Request timed out")
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        print("Rate limit exceeded")
    else:
        print(f"HTTP error: {e}")
except FileNotFoundError:
    print("Image file not found")
```

## Authentication

For secured API endpoints:

```python
client = SolarFaultDetectionClient(
    api_key="your-secret-api-key"
)
```

## Advanced Usage

### Custom Request Configuration

```python
import requests

# Custom session configuration
client = SolarFaultDetectionClient()
client.session.headers.update({"X-Custom-Header": "value"})
client.session.verify = False  # Disable SSL verification
```

### Monitoring and Metrics

```python
# Get API health and metrics
health = client.health_check()
metrics = client.get_metrics()

print(f"API Status: {health['status']}")
print(f"Total Requests: {metrics['total_requests']}")
print(f"Average Latency: {metrics['avg_latency']}ms")
```

### Batch Processing with Progress

```python
from tqdm import tqdm

image_paths = [f"images/{i}.jpg" for i in range(100)]
results = []

for i in tqdm(range(0, len(image_paths), 10)):
    batch = image_paths[i:i+10]
    batch_results = client.predict_batch(batch)
    results.extend(batch_results)
```

## Integration Examples

### Flask Web Application

```python
from flask import Flask, request, jsonify
from solar_fault_detection_sdk import SolarFaultDetectionClient

app = Flask(__name__)
client = SolarFaultDetectionClient()

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    # Save temporarily
    temp_path = f"/tmp/{file.filename}"
    file.save(temp_path)

    try:
        result = client.predict(temp_path)
        return jsonify(result)
    finally:
        os.remove(temp_path)
```

### Streamlit Dashboard

```python
import streamlit as st
from solar_fault_detection_sdk import SolarFaultDetectionClient

client = SolarFaultDetectionClient()

st.title("Solar Panel Fault Detection")

uploaded_file = st.file_uploader("Choose an image...")
if uploaded_file is not None:
    # Save temporarily
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getvalue())

    # Predict
    result = client.predict("temp.jpg")

    st.image(uploaded_file, caption="Uploaded Image")
    st.write(f"**Predicted Class:** {result['predicted_class']}")
    st.write(f"**Confidence:** {result['confidence']:.2%}")

    # Show probabilities
    st.bar_chart(result['probabilities'])
```

### Command Line Tool

```python
#!/usr/bin/env python3
import argparse
from solar_fault_detection_sdk import quick_predict

def main():
    parser = argparse.ArgumentParser(description='Solar Panel Fault Detection')
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--url', help='API server URL')

    args = parser.parse_args()

    try:
        result = quick_predict(args.image, base_url=args.url)
        print(f"Prediction: Class {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.3f}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Check if API server is running
   - Verify the base_url is correct

2. **Timeout Errors**
   - Increase timeout parameter
   - Check network connectivity

3. **Authentication Errors**
   - Verify API key is correct
   - Check if API requires authentication

4. **File Not Found**
   - Ensure image file exists
   - Check file permissions

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

client = SolarFaultDetectionClient()
```

## Contributing

To contribute to the SDK:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

This SDK is released under the MIT License. See LICENSE file for details.