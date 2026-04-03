# 💡 Code Examples

Practical examples for using the Solar Panel Fault Detection system.

## Example 1: Basic Inference

```python
from pathlib import Path
from solar_fault_detector import Predictor, Config

# Load configuration
config = Config()

# Create predictor
predictor = Predictor(
    model_path=Path("artifacts/models/best_model.h5"),
    config=config.model
)

# Make prediction
result = predictor.predict(Path("test.jpg"))

print(f"Class: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## Example 2: Batch Processing

```python
from pathlib import Path
from solar_fault_detector import BatchInferenceEngine, Config

config = Config()

# Create batch engine
engine = BatchInferenceEngine(
    model_path=Path("artifacts/models/best_model.h5"),
    config=config.model
)

# Process directory
results = engine.predict_directory(Path("solar_images"))

# Analyze results
for result in results:
    class_name = ["Normal", "Dust", "High Temp", "Cracks", "Water", "Delamination"][result['predicted_class']]
    print(f"{result['image']}: {class_name} ({result['confidence']:.1%})")
```

## Example 3: Using the REST API

### Python

```python
import requests
import json

# Point to local or remote API
API_URL = "http://localhost:8000"

# Make prediction
with open("solar_panel.jpg", "rb") as f:
    response = requests.post(
        f"{API_URL}/predict",
        files={"file": f}
    )

# Check result
if response.status_code == 200:
    result = response.json()
    print(json.dumps(result, indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

### JavaScript/Node.js

```javascript
const fs = require('fs');
const FormData = require('form-data');
const axios = require('axios');

async function predictFault() {
    const form = new FormData();
    form.append('file', fs.createReadStream('solar_panel.jpg'));
    
    try {
        const response = await axios.post(
            'http://localhost:8000/predict',
            form,
            { headers: form.getHeaders() }
        );
        console.log(response.data);
    } catch (error) {
        console.error('Prediction failed:', error.message);
    }
}

predictFault();
```

### cURL

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@solar_panel.jpg" \
  -H "accept: application/json" | jq .
```

## Example 4: Custom Data Pipeline

```python
from pathlib import Path
from solar_fault_detector.data import ImagePreprocessor
from solar_fault_detector.config import Config

config = Config()
preprocessor = ImagePreprocessor(config.model)

# Load single image
image_array = preprocessor.load_and_preprocess("solar_panel.jpg")
print(f"Shape: {image_array.shape}")

# Load batch
image_paths = list(Path("images").glob("*.jpg"))
batch = preprocessor.preprocess_batch(image_paths)
print(f"Batch shape: {batch.shape}")

# Generator for large datasets
for batch in preprocessor.preprocess_batch_generator(image_paths, batch_size=32):
    print(f"Processing batch: {batch.shape}")
    # Process batch
```

## Example 5: Flask Web Application

```python
from flask import Flask, request, jsonify
from pathlib import Path
import tempfile

from solar_fault_detector import Predictor, Config

app = Flask(__name__)

# Initialize
config = Config()
predictor = Predictor(
    model_path=Path("artifacts/models/best_model.h5"),
    config=config.model
)

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Save temporarily
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        file.save(tmp.name)
        
        try:
            result = predictor.predict(Path(tmp.name))
            return jsonify(result)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            Path(tmp.name).unlink()

if __name__ == '__main__':
    app.run(debug=False, port=5000)
```

## Example 6: Streamlit Interactive App

```python
import streamlit as st
from pathlib import Path
from PIL import Image

from solar_fault_detector import Predictor, Config

# Page setup
st.set_page_config(page_title="Solar Panel Fault Detector")

st.title("🔍 Solar Panel Fault Detection")
st.write("Upload an image to detect faults in solar panels")

# Initialize
@st.cache_resource
def load_predictor():
    config = Config()
    return Predictor(
        model_path=Path("artifacts/models/best_model.h5"),
        config=config.model
    )

predictor = load_predictor()

# Class labels
class_names = [
    "✅ Normal",
    "💨 Dust",
    "🔥 High Temperature",
    "💔 Cracks",
    "💧 Water Damage",
    "📄 Delamination"
]

# File upload
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Make prediction
    with st.spinner("Analyzing..."):
        result = predictor.predict(Path(uploaded_file.name))
    
    # Display results
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            pred_class = result['predicted_class']
            st.metric(
                "Prediction",
                class_names[pred_class],
                f"{result['confidence']:.1%}"
            )
        
        with col2:
            st.metric("Confidence", f"{result['confidence']:.2%}")
    
    # Probability chart
    st.bar_chart(
        {class_names[i]: result['probabilities'][i] 
         for i in range(len(class_names))}
    )
```

## Example 7: Command-Line Tool

```python
# cli.py
import click
from pathlib import Path
from solar_fault_detector import Predictor, BatchInferenceEngine, Config

@click.group()
def cli():
    """Solar Panel Fault Detection CLI"""
    pass

@cli.command()
@click.argument('image_path', type=click.Path(exists=True))
def predict(image_path):
    """Predict fault for single image"""
    config = Config()
    predictor = Predictor(
        model_path=Path("artifacts/models/best_model.h5"),
        config=config.model
    )
    
    result = predictor.predict(Path(image_path))
    
    click.echo(f"Image: {result['image']}")
    click.echo(f"Class: {result['predicted_class']}")
    click.echo(f"Confidence: {result['confidence']:.2%}")

@cli.command()
@click.argument('directory', type=click.Path(exists=True))
def batch_predict(directory):
    """Batch predict for all images in directory"""
    config = Config()
    engine = BatchInferenceEngine(
        model_path=Path("artifacts/models/best_model.h5"),
        config=config.model
    )
    
    results = engine.predict_directory(Path(directory))
    
    click.echo(f"Processed {len(results)} images")
    for result in results:
        click.echo(f"{result['image']}: Class {result['predicted_class']}")

if __name__ == '__main__':
    cli()
```

Run with:
```bash
python cli.py predict image.jpg
python cli.py batch-predict ./images
```

## Example 8: Docker Deployment

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

# Copy code
COPY . .
RUN pip install -e .

# Download model
RUN python -m solar_fault_detector.utils.download_model

# Run API
CMD ["uvicorn", "apps.api.fastapi_app:app", "--host", "0.0.0.0", "--port", "5000"]
```

Build and run:
```bash
docker build -t detector .
docker run -p 5000:5000 detector
```

