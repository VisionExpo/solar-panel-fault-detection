import os
import sys
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
import json
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from io import BytesIO
from PIL import Image
import base64

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the inference module
from inference import SolarPanelFaultDetector

# Path to model and label mapping
MODEL_PATH = os.environ.get("MODEL_PATH", "deployment/model")
LABEL_MAPPING_PATH = os.environ.get("LABEL_MAPPING_PATH", "deployment/label_mapping.json")

# Initialize detector
detector = SolarPanelFaultDetector(MODEL_PATH, LABEL_MAPPING_PATH)

# Create FastAPI app
app = FastAPI(
    title="Solar Panel Fault Detector API",
    description="API for detecting faults in solar panels",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Solar Panel Fault Detector API",
        "version": "1.0.0",
        "endpoints": {
            "/predict": "Predict fault type for an image",
            "/health": "Check API health",
            "/info": "Get model information",
        }
    }

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}

@app.get("/info")
async def info():
    """Get model information"""
    return {
        "model_path": MODEL_PATH,
        "label_mapping_path": LABEL_MAPPING_PATH,
        "classes": list(detector.label_mapping.keys()),
        "input_shape": detector.input_shape,
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Predict fault type for an image

    Args:
        file: Image file

    Returns:
        JSON response with predictions
    """
    try:
        # Read image
        contents = await file.read()
        image = Image.open(BytesIO(contents))

        # Convert to numpy array
        image_np = np.array(image)

        # Convert to RGB if needed
        if len(image_np.shape) == 2:  # Grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        # Get predictions
        predictions = detector.predict(image_np)

        # Get top prediction
        top_class, top_confidence = detector.classify_image(image_np)

        # Get top 3 predictions
        top_3 = detector.get_top_k_predictions(image_np, k=3)

        # Format response
        response = {
            "prediction": {
                "class": top_class,
                "confidence": float(top_confidence),
            },
            "top_3": [
                {"class": class_name, "confidence": float(conf)}
                for class_name, conf in top_3
            ],
            "all_predictions": {
                class_name: float(conf)
                for class_name, conf in predictions.items()
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch(files: list[UploadFile] = File(...)):
    """
    Predict fault types for a batch of images

    Args:
        files: List of image files

    Returns:
        JSON response with predictions for each image
    """
    try:
        results = []

        for file in files:
            # Read image
            contents = await file.read()
            image = Image.open(BytesIO(contents))

            # Convert to numpy array
            image_np = np.array(image)

            # Convert to RGB if needed
            if len(image_np.shape) == 2:  # Grayscale
                image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
            elif image_np.shape[2] == 4:  # RGBA
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

            # Get top prediction
            top_class, top_confidence = detector.classify_image(image_np)

            # Add to results
            results.append({
                "filename": file.filename,
                "prediction": {
                    "class": top_class,
                    "confidence": float(top_confidence),
                }
            })

        return JSONResponse(content={"results": results})

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_base64")
async def predict_base64(data: dict):
    """
    Predict fault type for a base64-encoded image

    Args:
        data: Dictionary with base64-encoded image

    Returns:
        JSON response with predictions
    """
    try:
        # Get base64 image
        base64_image = data.get("image")
        if not base64_image:
            raise HTTPException(status_code=400, detail="No image provided")

        # Decode base64 image
        image_data = base64.b64decode(base64_image)
        image = Image.open(BytesIO(image_data))

        # Convert to numpy array
        image_np = np.array(image)

        # Convert to RGB if needed
        if len(image_np.shape) == 2:  # Grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
        elif image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

        # Get predictions
        predictions = detector.predict(image_np)

        # Get top prediction
        top_class, top_confidence = detector.classify_image(image_np)

        # Get top 3 predictions
        top_3 = detector.get_top_k_predictions(image_np, k=3)

        # Format response
        response = {
            "prediction": {
                "class": top_class,
                "confidence": float(top_confidence),
            },
            "top_3": [
                {"class": class_name, "confidence": float(conf)}
                for class_name, conf in top_3
            ],
            "all_predictions": {
                class_name: float(conf)
                for class_name, conf in predictions.items()
            }
        }

        return JSONResponse(content=response)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Run FastAPI app
    uvicorn.run(app, host="0.0.0.0", port=8000)
