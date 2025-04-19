import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the inference module
from inference import SolarPanelFaultDetector

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the Solar Panel Fault Detector')
    parser.add_argument('--model', type=str, default='model', help='Path to the model directory')
    parser.add_argument('--labels', type=str, default='label_mapping.json', help='Path to the label mapping file')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file or directory')
    parser.add_argument('--output', type=str, default=None, help='Path to save the output visualization')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top predictions to show')
    return parser.parse_args()

def visualize_prediction(image, predictions, top_k=3):
    """
    Visualize the prediction results.

    Args:
        image: Input image
        predictions: Dictionary of class probabilities
        top_k: Number of top predictions to show

    Returns:
        Visualization image
    """
    # Sort predictions by confidence
    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    top_k_predictions = sorted_predictions[:top_k]

    # Create figure
    plt.figure(figsize=(12, 6))

    # Plot image
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')

    # Plot predictions
    plt.subplot(1, 2, 2)
    classes = [p[0] for p in top_k_predictions]
    confidences = [p[1] for p in top_k_predictions]

    # Create horizontal bar chart
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, confidences, align='center', alpha=0.5)
    plt.yticks(y_pos, classes)
    plt.xlabel('Confidence')
    plt.title('Top {} Predictions'.format(top_k))

    plt.tight_layout()
    return plt

def main():
    """Main function."""
    args = parse_args()

    # Initialize detector
    detector = SolarPanelFaultDetector(args.model, args.labels)

    # Check if image is a directory
    if os.path.isdir(args.image):
        # Process all images in directory
        image_files = [os.path.join(args.image, f) for f in os.listdir(args.image)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for image_file in image_files:
            process_single_image(detector, image_file, args.output, args.top_k)
    else:
        # Process single image
        process_single_image(detector, args.image, args.output, args.top_k)

def process_single_image(detector, image_path, output_path=None, top_k=3):
    """
    Process a single image.

    Args:
        detector: SolarPanelFaultDetector instance
        image_path: Path to the image file
        output_path: Path to save the output visualization
        top_k: Number of top predictions to show
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return

    # Get predictions
    predictions = detector.predict(image)

    # Get top prediction
    top_class, confidence = detector.classify_image(image)

    # Print results
    print(f"\nImage: {image_path}")
    print(f"Prediction: {top_class} (confidence: {confidence:.4f})")

    # Print top k predictions
    top_k_predictions = detector.get_top_k_predictions(image, k=top_k)
    print(f"Top {top_k} predictions:")
    for class_name, conf in top_k_predictions:
        print(f"  {class_name}: {conf:.4f}")

    # Visualize results
    plt = visualize_prediction(image, predictions, top_k)

    # Save or show visualization
    if output_path:
        # Create output directory if it doesn't exist
        parent_dir = os.path.dirname(output_path)
        if parent_dir:  # Only create if parent_dir is not empty
            os.makedirs(parent_dir, exist_ok=True)

        # Generate output filename
        if os.path.isdir(output_path):
            base_name = os.path.basename(image_path)
            output_file = os.path.join(output_path, f"pred_{base_name}")
        else:
            output_file = output_path

        plt.savefig(output_file)
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
