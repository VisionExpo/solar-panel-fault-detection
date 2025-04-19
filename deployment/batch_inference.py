import os
import sys
import argparse
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import json
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import the inference module
from inference import SolarPanelFaultDetector

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Batch inference for Solar Panel Fault Detector')
    parser.add_argument('--model', type=str, default='model', help='Path to the model directory')
    parser.add_argument('--labels', type=str, default='label_mapping.json', help='Path to the label mapping file')
    parser.add_argument('--input', type=str, required=True, help='Path to the input directory containing images')
    parser.add_argument('--output', type=str, required=True, help='Path to the output directory for results')
    parser.add_argument('--visualize', action='store_true', help='Generate visualization images')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--threshold', type=float, default=0.5, help='Confidence threshold for classification')
    return parser.parse_args()

def process_batch(detector, image_paths, batch_size=16, threshold=0.5):
    """
    Process a batch of images.
    
    Args:
        detector: SolarPanelFaultDetector instance
        image_paths: List of image paths
        batch_size: Batch size for inference
        threshold: Confidence threshold for classification
        
    Returns:
        List of dictionaries with prediction results
    """
    results = []
    
    # Process images in batches
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        
        # Load images
        images = []
        for path in batch_paths:
            img = cv2.imread(path)
            if img is not None:
                images.append(img)
            else:
                print(f"Warning: Could not load image {path}")
        
        if not images:
            continue
        
        # Get predictions
        batch_predictions = detector.predict_batch(images)
        
        # Process predictions
        for j, (path, predictions) in enumerate(zip(batch_paths, batch_predictions)):
            # Get top prediction
            top_class = max(predictions.items(), key=lambda x: x[1])
            class_name, confidence = top_class
            
            # Skip if confidence is below threshold
            if confidence < threshold:
                class_name = "Unknown"
            
            # Get top 3 predictions
            top_3 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]
            
            # Add to results
            results.append({
                "image_path": path,
                "prediction": class_name,
                "confidence": float(confidence),
                "top_3": [{"class": c, "confidence": float(conf)} for c, conf in top_3],
                "all_predictions": {c: float(conf) for c, conf in predictions.items()}
            })
    
    return results

def visualize_results(image_path, prediction, output_dir):
    """
    Create visualization for a prediction.
    
    Args:
        image_path: Path to the image
        prediction: Prediction dictionary
        output_dir: Output directory for visualization
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
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
    classes = [p["class"] for p in prediction["top_3"]]
    confidences = [p["confidence"] for p in prediction["top_3"]]
    
    # Create horizontal bar chart
    y_pos = np.arange(len(classes))
    plt.barh(y_pos, confidences, align='center', alpha=0.5)
    plt.yticks(y_pos, classes)
    plt.xlabel('Confidence')
    plt.title(f'Prediction: {prediction["prediction"]} ({prediction["confidence"]:.2f})')
    
    # Save figure
    output_path = os.path.join(output_dir, os.path.basename(image_path).replace('.', '_pred.'))
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    """Main function."""
    args = parse_args()
    
    # Initialize detector
    detector = SolarPanelFaultDetector(args.model, args.labels)
    
    # Get image paths
    image_paths = []
    for root, _, files in os.walk(args.input):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print(f"No images found in {args.input}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Create visualization directory if needed
    if args.visualize:
        vis_dir = os.path.join(args.output, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
    
    # Process images
    start_time = time.time()
    results = process_batch(detector, image_paths, args.batch_size, args.threshold)
    end_time = time.time()
    
    # Print statistics
    print(f"Processed {len(results)} images in {end_time - start_time:.2f} seconds")
    print(f"Average time per image: {(end_time - start_time) / len(results):.4f} seconds")
    
    # Count predictions by class
    class_counts = {}
    for result in results:
        class_name = result["prediction"]
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
    
    print("\nPrediction counts:")
    for class_name, count in sorted(class_counts.items()):
        print(f"  {class_name}: {count} ({count / len(results) * 100:.1f}%)")
    
    # Save results to CSV
    csv_path = os.path.join(args.output, "predictions.csv")
    df = pd.DataFrame([{
        "image": os.path.basename(r["image_path"]),
        "prediction": r["prediction"],
        "confidence": r["confidence"]
    } for r in results])
    df.to_csv(csv_path, index=False)
    print(f"\nSaved predictions to {csv_path}")
    
    # Save detailed results to JSON
    json_path = os.path.join(args.output, "predictions_detailed.json")
    with open(json_path, 'w') as f:
        json.dump({os.path.basename(r["image_path"]): {
            "prediction": r["prediction"],
            "confidence": r["confidence"],
            "top_3": r["top_3"]
        } for r in results}, f, indent=2)
    print(f"Saved detailed predictions to {json_path}")
    
    # Create visualizations if requested
    if args.visualize:
        print("\nCreating visualizations...")
        for result in tqdm(results):
            visualize_results(result["image_path"], result, vis_dir)
        print(f"Saved visualizations to {vis_dir}")

if __name__ == "__main__":
    main()
