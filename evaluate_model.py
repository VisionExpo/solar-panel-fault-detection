import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

def load_test_data(test_dir, img_size=(384, 384), batch_size=32):
    """
    Load test data using ImageDataGenerator.
    
    Args:
        test_dir: Directory containing test data
        img_size: Size to resize images to
        batch_size: Batch size for evaluation
        
    Returns:
        Test data generator and class indices
    """
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )
    
    return test_generator

def evaluate_model(model, test_generator):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        test_generator: Test data generator
        
    Returns:
        Test loss, test accuracy, predictions, and true labels
    """
    # Get predictions
    steps = test_generator.samples // test_generator.batch_size + 1
    predictions = model.predict(test_generator, steps=steps)
    
    # Get true labels
    true_labels = test_generator.classes
    
    # Get predicted labels
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(test_generator)
    
    return test_loss, test_accuracy, predicted_labels, true_labels, predictions

def plot_confusion_matrix(true_labels, predicted_labels, class_names, output_dir):
    """
    Plot confusion matrix.
    
    Args:
        true_labels: True labels
        predicted_labels: Predicted labels
        class_names: List of class names
        output_dir: Directory to save the plot
    """
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Normalize confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def plot_class_metrics(classification_rep, class_names, output_dir):
    """
    Plot class-wise metrics.
    
    Args:
        classification_rep: Classification report from sklearn
        class_names: List of class names
        output_dir: Directory to save the plot
    """
    # Extract metrics from classification report
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = {}
    
    for class_name in class_names:
        class_metrics[class_name] = {metric: classification_rep[class_name][metric] for metric in metrics}
    
    # Create DataFrame for plotting
    import pandas as pd
    df = pd.DataFrame(class_metrics).T
    
    # Plot class metrics
    plt.figure(figsize=(12, 6))
    df.plot(kind='bar', figsize=(12, 6))
    plt.title('Class-wise Metrics')
    plt.xlabel('Class')
    plt.ylabel('Score')
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_metrics.png'))
    plt.close()

def plot_sample_predictions(model, test_generator, class_names, output_dir, num_samples=16):
    """
    Plot sample predictions.
    
    Args:
        model: Trained model
        test_generator: Test data generator
        class_names: List of class names
        output_dir: Directory to save the plot
        num_samples: Number of samples to plot
    """
    # Reset the generator
    test_generator.reset()
    
    # Get a batch of images and labels
    images, labels = next(test_generator)
    
    # Make predictions
    predictions = model.predict(images)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = np.argmax(labels, axis=1)
    
    # Plot images with predictions
    plt.figure(figsize=(15, 15))
    for i in range(min(num_samples, len(images))):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i])
        
        # Get class names
        true_class = class_names[true_classes[i]]
        pred_class = class_names[predicted_classes[i]]
        confidence = predictions[i][predicted_classes[i]]
        
        # Set title color based on correctness
        title_color = 'green' if true_class == pred_class else 'red'
        
        plt.title(f"True: {true_class}\nPred: {pred_class} ({confidence:.2f})", 
                  color=title_color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sample_predictions.png'))
    plt.close()

def main():
    # Directories
    data_dir = Path("data/solar_panels/organized")
    test_dir = data_dir / "test"
    model_dir = Path("src/models")
    vis_dir = Path("visualization/static")
    
    # Check if directories exist
    if not test_dir.exists():
        print("Test directory not found. Please run download_dataset.py first.")
        return 1
    
    if not (model_dir / "final_model").exists():
        print("Model not found. Please run train_model.py first.")
        return 1
    
    os.makedirs(vis_dir, exist_ok=True)
    
    # Load label mapping
    with open(model_dir / "label_mapping.json", "r") as f:
        label_mapping = json.load(f)
    
    class_names = [label_mapping[str(i)] for i in range(len(label_mapping))]
    print(f"Class names: {class_names}")
    
    # Load model
    print("Loading model...")
    model = load_model(model_dir / "final_model")
    
    # Load test data
    print("Loading test data...")
    test_generator = load_test_data(test_dir)
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy, predicted_labels, true_labels, predictions = evaluate_model(model, test_generator)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate classification report
    print("Generating classification report...")
    report = classification_report(true_labels, predicted_labels, target_names=class_names, output_dict=True)
    print(classification_report(true_labels, predicted_labels, target_names=class_names))
    
    # Save classification report
    with open(vis_dir / "classification_report.json", "w") as f:
        json.dump(report, f)
    
    # Plot confusion matrix
    print("Plotting confusion matrix...")
    plot_confusion_matrix(true_labels, predicted_labels, class_names, vis_dir)
    
    # Plot class metrics
    print("Plotting class metrics...")
    plot_class_metrics(report, class_names, vis_dir)
    
    # Plot sample predictions
    print("Plotting sample predictions...")
    plot_sample_predictions(model, test_generator, class_names, vis_dir)
    
    print("Evaluation completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
