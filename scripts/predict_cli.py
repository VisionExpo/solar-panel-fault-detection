import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.inference import SolarPanelFaultDetector

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Solar Panel Fault Detection CLI')
    parser.add_argument('--image', type=str, required=True, help='Path to the image file')
    parser.add_argument('--model', type=str, default='artifacts/models/final_model', help='Path to the model directory')
    parser.add_argument('--labels', type=str, default='artifacts/models/label_mapping.json', help='Path to the label mapping file')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for classification')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top predictions to show')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image file not found: {args.image}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model directory not found: {args.model}")
        return
    
    # Check if label mapping exists
    if not os.path.exists(args.labels):
        print(f"Error: Label mapping file not found: {args.labels}")
        return
    
    # Initialize detector
    detector = SolarPanelFaultDetector(args.model, args.labels)
    
    # Load image
    image = cv2.imread(args.image)
    if image is None:
        print(f"Error: Failed to load image: {args.image}")
        return
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get predictions
    predictions = detector.predict(image_rgb)
    
    # Get top k predictions
    top_k = detector.get_top_k_predictions(image_rgb, k=args.top_k)
    
    # Get classification
    predicted_class, confidence = detector.classify_image(image_rgb, threshold=args.threshold)
    
    # Print results
    print(f"\nImage: {args.image}")
    print(f"Classification: {predicted_class} (confidence: {confidence:.4f})")
    
    print(f"\nTop {args.top_k} predictions:")
    for class_name, prob in top_k:
        print(f"  {class_name}: {prob:.4f}")
    
    # Visualize results if requested
    if args.visualize:
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot image
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title(f"Prediction: {predicted_class} ({confidence:.2f})")
        plt.axis('off')
        
        # Plot bar chart of top predictions
        plt.subplot(1, 2, 2)
        classes = [class_name for class_name, _ in top_k]
        probs = [prob for _, prob in top_k]
        
        # Create horizontal bar chart
        bars = plt.barh(classes, probs, color=['green' if cls == predicted_class else 'blue' for cls in classes])
        plt.xlim(0, 1)
        plt.title('Prediction Probabilities')
        
        # Add confidence values to bars
        for bar, conf in zip(bars, probs):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f"{conf:.2f}", 
                    va='center')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
