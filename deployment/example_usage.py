import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import json
import base64
from io import BytesIO
from PIL import Image

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Example usage of Solar Panel Fault Detector')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--api_url', type=str, default='http://localhost:8000', help='URL of the API server')
    parser.add_argument('--mode', type=str, choices=['local', 'api'], default='local', help='Mode to run the example')
    return parser.parse_args()

def local_inference(image_path):
    """
    Run inference locally using the SolarPanelFaultDetector class.

    Args:
        image_path: Path to the image file

    Returns:
        Prediction results
    """
    # Import the inference module
    from inference import SolarPanelFaultDetector

    # Initialize detector
    detector = SolarPanelFaultDetector('deployment/model', 'deployment/label_mapping.json')

    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None

    # Get predictions
    predictions = detector.predict(image)

    # Get top prediction
    top_class, confidence = detector.classify_image(image)

    # Get top 3 predictions
    top_3 = detector.get_top_k_predictions(image, k=3)

    # Create result dictionary
    result = {
        "prediction": {
            "class": top_class,
            "confidence": float(confidence),
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

    return result

def api_inference(image_path, api_url):
    """
    Run inference using the API.

    Args:
        image_path: Path to the image file
        api_url: URL of the API server

    Returns:
        Prediction results
    """
    # Load image
    with open(image_path, 'rb') as f:
        files = {'file': (os.path.basename(image_path), f, 'image/jpeg')}

        # Send request to API
        response = requests.post(f"{api_url}/predict", files=files)

        # Check if request was successful
        if response.status_code != 200:
            print(f"Error: API request failed with status code {response.status_code}")
            print(response.text)
            return None

        # Parse response
        result = response.json()

        return result

def api_inference_base64(image_path, api_url):
    """
    Run inference using the API with base64 encoding.

    Args:
        image_path: Path to the image file
        api_url: URL of the API server

    Returns:
        Prediction results
    """
    # Load image and convert to base64
    with open(image_path, 'rb') as f:
        image_data = f.read()
        base64_image = base64.b64encode(image_data).decode('utf-8')

        # Send request to API
        response = requests.post(
            f"{api_url}/predict_base64",
            json={"image": base64_image}
        )

        # Check if request was successful
        if response.status_code != 200:
            print(f"Error: API request failed with status code {response.status_code}")
            print(response.text)
            return None

        # Parse response
        result = response.json()

        return result

def visualize_result(image_path, result):
    """
    Visualize the prediction result.

    Args:
        image_path: Path to the image file
        result: Prediction result
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')

    # Plot predictions
    plt.subplot(1, 2, 2)
    classes = [p["class"] for p in result["top_3"]]
    confidences = [p["confidence"] for p in result["top_3"]]

    # Create horizontal bar chart
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, confidences, align='center', alpha=0.5)
    plt.yticks(y_pos, classes)
    plt.xlabel('Confidence')
    plt.title(f'Prediction: {result["prediction"]["class"]} ({result["prediction"]["confidence"]:.2f})')

    plt.tight_layout()
    plt.show()

def main():
    """Main function."""
    args = parse_args()

    # Check if image exists
    if not os.path.isfile(args.image):
        print(f"Error: Image file {args.image} does not exist")
        return

    # Run inference
    if args.mode == 'local':
        print("Running local inference...")
        result = local_inference(args.image)
    else:
        print(f"Running API inference using {args.api_url}...")
        result = api_inference(args.image, args.api_url)

        if result is None:
            print("Trying base64 API endpoint...")
            result = api_inference_base64(args.image, args.api_url)

    if result is None:
        print("Error: Inference failed")
        return

    # Print results
    print("\nPrediction Results:")
    print(f"  Class: {result['prediction']['class']}")
    print(f"  Confidence: {result['prediction']['confidence']:.4f}")

    print("\nTop 3 Predictions:")
    for i, pred in enumerate(result["top_3"]):
        print(f"  {i+1}. {pred['class']}: {pred['confidence']:.4f}")

    # Visualize result
    visualize_result(args.image, result)

if __name__ == "__main__":
    main()
