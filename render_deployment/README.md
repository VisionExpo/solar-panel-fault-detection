# Solar Panel Fault Detector - Render Deployment

This repository contains the files needed to deploy the Solar Panel Fault Detector on Render.

## Deployment Instructions

1. Create a new Web Service on Render.
2. Connect your GitHub repository.
3. Use the following settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python app.py`
   - **Environment Variables**:
     - `PORT`: 7860
     - `PYTHON_VERSION`: 3.9.0

## API Endpoints

The application provides both a web interface and a REST API:

- Web Interface: `https://your-app-name.onrender.com`
- REST API: `https://your-app-name.onrender.com/api`

## API Documentation

When the application is running, you can access the API documentation at:
`https://your-app-name.onrender.com/docs`

## Files

- `app.py`: Gradio web application
- `app_fastapi.py`: FastAPI server
- `inference.py`: Core inference module
- `model/`: Directory containing the trained model
- `label_mapping.json`: Mapping of class indices to class names
- `static/`: Directory containing visualizations
- `requirements.txt`: Python dependencies
- `render.yaml`: Render deployment configuration
- `Procfile`: Process file for Render
- `runtime.txt`: Python runtime version
