import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from collections import Counter
import json

# Set style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = Path('visualization/static')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def count_images_by_class(data_dir):
    """Count the number of images in each class"""
    class_counts = {}
    
    # Get all subdirectories in the data directory
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
        
        # Count images in the directory
        image_files = [f for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(image_files)
    
    return class_counts

def plot_class_distribution(class_counts, output_dir):
    """Plot the distribution of images across classes"""
    plt.figure(figsize=(12, 6))
    
    # Sort classes by count
    sorted_counts = {k: v for k, v in sorted(class_counts.items(), 
                                            key=lambda item: item[1], 
                                            reverse=True)}
    
    # Create bar plot
    bars = plt.bar(sorted_counts.keys(), sorted_counts.values(), color='skyblue')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height}', ha='center', va='bottom')
    
    plt.title('Number of Images per Class', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Number of Images', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_image_dimensions(data_dir):
    """Analyze the dimensions of images in the dataset"""
    widths = []
    heights = []
    aspect_ratios = []
    
    # Process all images in all subdirectories
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
        
        # Get all image files
        image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sample images if there are too many
        if len(image_files) > 100:
            import random
            image_files = random.sample(image_files, 100)
        
        # Process each image
        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    return widths, heights, aspect_ratios

def plot_dimension_distributions(widths, heights, aspect_ratios, output_dir):
    """Plot distributions of image dimensions"""
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot width distribution
    sns.histplot(widths, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title('Image Width Distribution', fontsize=14)
    axes[0].set_xlabel('Width (pixels)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    
    # Plot height distribution
    sns.histplot(heights, kde=True, ax=axes[1], color='lightgreen')
    axes[1].set_title('Image Height Distribution', fontsize=14)
    axes[1].set_xlabel('Height (pixels)', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    
    # Plot aspect ratio distribution
    sns.histplot(aspect_ratios, kde=True, ax=axes[2], color='salmon')
    axes[2].set_title('Aspect Ratio Distribution', fontsize=14)
    axes[2].set_xlabel('Aspect Ratio (width/height)', fontsize=12)
    axes[2].set_ylabel('Count', fontsize=12)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / 'dimension_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_image_brightness(data_dir):
    """Analyze the brightness of images in the dataset"""
    brightness_by_class = {}
    
    # Process all images in all subdirectories
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        
        # Skip if not a directory
        if not os.path.isdir(class_dir):
            continue
        
        # Get all image files
        image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sample images if there are too many
        if len(image_files) > 50:
            import random
            image_files = random.sample(image_files, 50)
        
        brightness_values = []
        
        # Process each image
        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert to grayscale
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # Calculate average brightness
                    brightness = np.mean(gray)
                    brightness_values.append(brightness)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if brightness_values:
            brightness_by_class[class_name] = brightness_values
    
    return brightness_by_class

def plot_brightness_distribution(brightness_by_class, output_dir):
    """Plot distribution of image brightness by class"""
    plt.figure(figsize=(12, 8))
    
    # Create box plot
    data = []
    labels = []
    
    for class_name, values in brightness_by_class.items():
        data.append(values)
        labels.append(class_name)
    
    plt.boxplot(data, labels=labels, patch_artist=True)
    
    plt.title('Image Brightness Distribution by Class', fontsize=16)
    plt.xlabel('Class', fontsize=14)
    plt.ylabel('Brightness (0-255)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / 'brightness_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_sample_grid(data_dir, output_dir):
    """Create a grid of sample images from each class"""
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    # Number of samples per class
    n_samples = 5
    
    # Create a grid with one row per class
    n_rows = len(class_dirs)
    n_cols = n_samples
    
    plt.figure(figsize=(n_cols * 3, n_rows * 3))
    
    for i, class_name in enumerate(class_dirs):
        class_dir = os.path.join(data_dir, class_name)
        
        # Get all image files
        image_files = [os.path.join(class_dir, f) for f in os.listdir(class_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Sample images
        if len(image_files) > n_samples:
            import random
            sampled_files = random.sample(image_files, n_samples)
        else:
            sampled_files = image_files
        
        # Plot each sample
        for j, img_path in enumerate(sampled_files):
            if j >= n_samples:
                break
                
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    # Convert BGR to RGB
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Create subplot
                    plt.subplot(n_rows, n_cols, i * n_cols + j + 1)
                    plt.imshow(img_rgb)
                    plt.axis('off')
                    
                    # Add class name to the first image in each row
                    if j == 0:
                        plt.title(class_name, fontsize=12)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_batch_predictions(predictions_file, output_dir):
    """Analyze batch prediction results"""
    # Load predictions
    with open(predictions_file, 'r') as f:
        predictions = json.load(f)
    
    # Count predictions by class
    pred_counts = Counter()
    confidence_by_class = {}
    
    for img_name, pred_data in predictions.items():
        pred_class = pred_data['prediction']
        confidence = pred_data['confidence']
        
        pred_counts[pred_class] += 1
        
        if pred_class not in confidence_by_class:
            confidence_by_class[pred_class] = []
        
        confidence_by_class[pred_class].append(confidence)
    
    # Plot prediction distribution
    plt.figure(figsize=(12, 6))
    
    # Sort classes by count
    sorted_counts = {k: v for k, v in sorted(pred_counts.items(), 
                                            key=lambda item: item[1], 
                                            reverse=True)}
    
    # Create bar plot
    bars = plt.bar(sorted_counts.keys(), sorted_counts.values(), color='lightcoral')
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height}', ha='center', va='bottom')
    
    plt.title('Prediction Distribution', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot confidence distribution by class
    plt.figure(figsize=(12, 6))
    
    data = []
    labels = []
    
    for class_name, values in confidence_by_class.items():
        if values:  # Only include classes with predictions
            data.append(values)
            labels.append(class_name)
    
    plt.boxplot(data, labels=labels, patch_artist=True)
    
    plt.title('Prediction Confidence by Class', fontsize=16)
    plt.xlabel('Predicted Class', fontsize=14)
    plt.ylabel('Confidence', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Create output directory
    output_dir = create_output_dir()
    
    # Data directories
    train_dir = 'data/processed/train'
    val_dir = 'data/processed/val'
    test_dir = 'data/processed/test'
    
    # Check if directories exist
    if not os.path.exists(train_dir):
        train_dir = 'Faulty_solar_panel'  # Use original data if processed not available
    
    # Generate visualizations for training data
    if os.path.exists(train_dir):
        print(f"Analyzing training data in {train_dir}...")
        
        # Class distribution
        class_counts = count_images_by_class(train_dir)
        plot_class_distribution(class_counts, output_dir)
        
        # Image dimensions
        widths, heights, aspect_ratios = analyze_image_dimensions(train_dir)
        plot_dimension_distributions(widths, heights, aspect_ratios, output_dir)
        
        # Image brightness
        brightness_by_class = analyze_image_brightness(train_dir)
        plot_brightness_distribution(brightness_by_class, output_dir)
        
        # Sample images
        create_sample_grid(train_dir, output_dir)
    
    # Analyze batch predictions if available
    predictions_file = 'batch_output/predictions_detailed.json'
    if os.path.exists(predictions_file):
        print(f"Analyzing batch predictions from {predictions_file}...")
        analyze_batch_predictions(predictions_file, output_dir)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
