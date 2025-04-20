import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Set style for plots
plt.style.use('ggplot')
sns.set(style="whitegrid")

def create_output_dir():
    """Create output directory for visualizations"""
    output_dir = Path('visualization/static')
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def load_training_history(history_file):
    """Load training history from JSON file"""
    with open(history_file, 'r') as f:
        history = json.load(f)
    return history

def plot_training_curves(history, output_dir):
    """Plot training and validation curves"""
    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot accuracy
    axes[0].plot(history['accuracy'], label='Training Accuracy', marker='o', linestyle='-', color='blue')
    axes[0].plot(history['val_accuracy'], label='Validation Accuracy', marker='o', linestyle='-', color='orange')
    axes[0].set_title('Model Accuracy', fontsize=16)
    axes[0].set_xlabel('Epoch', fontsize=14)
    axes[0].set_ylabel('Accuracy', fontsize=14)
    axes[0].legend(loc='lower right', fontsize=12)
    axes[0].grid(True)
    
    # Plot loss
    axes[1].plot(history['loss'], label='Training Loss', marker='o', linestyle='-', color='blue')
    axes[1].plot(history['val_loss'], label='Validation Loss', marker='o', linestyle='-', color='orange')
    axes[1].set_title('Model Loss', fontsize=16)
    axes[1].set_xlabel('Epoch', fontsize=14)
    axes[1].set_ylabel('Loss', fontsize=14)
    axes[1].legend(loc='upper right', fontsize=12)
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_learning_rate(history, output_dir):
    """Plot learning rate if available"""
    if 'lr' in history:
        plt.figure(figsize=(12, 6))
        
        plt.plot(history['lr'], marker='o', linestyle='-', color='green')
        plt.title('Learning Rate Schedule', fontsize=16)
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Learning Rate', fontsize=14)
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / 'learning_rate.png', dpi=300, bbox_inches='tight')
        plt.close()

def create_training_summary(history, output_dir):
    """Create a summary of training metrics"""
    # Extract key metrics
    final_train_acc = history['accuracy'][-1]
    final_val_acc = history['val_accuracy'][-1]
    final_train_loss = history['loss'][-1]
    final_val_loss = history['val_loss'][-1]
    
    best_val_acc = max(history['val_accuracy'])
    best_val_acc_epoch = history['val_accuracy'].index(best_val_acc) + 1
    
    best_val_loss = min(history['val_loss'])
    best_val_loss_epoch = history['val_loss'].index(best_val_loss) + 1
    
    # Create summary dictionary
    summary = {
        'epochs': len(history['accuracy']),
        'final_train_acc': final_train_acc,
        'final_val_acc': final_val_acc,
        'final_train_loss': final_train_loss,
        'final_val_loss': final_val_loss,
        'best_val_acc': best_val_acc,
        'best_val_acc_epoch': best_val_acc_epoch,
        'best_val_loss': best_val_loss,
        'best_val_loss_epoch': best_val_loss_epoch
    }
    
    # Save summary to JSON
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary

def plot_confusion_matrix(confusion_matrix_file, output_dir):
    """Plot confusion matrix from file"""
    try:
        # Load confusion matrix
        with open(confusion_matrix_file, 'r') as f:
            cm_data = json.load(f)
        
        confusion_matrix = np.array(cm_data['confusion_matrix'])
        class_names = cm_data['class_names']
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_norm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(cm_norm, annot=confusion_matrix, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")

def plot_class_metrics(metrics_file, output_dir):
    """Plot per-class metrics from file"""
    try:
        # Load metrics
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Extract class names and metrics
        class_names = list(metrics.keys())
        precision = [metrics[cls]['precision'] for cls in class_names]
        recall = [metrics[cls]['recall'] for cls in class_names]
        f1_score = [metrics[cls]['f1-score'] for cls in class_names]
        
        # Create figure with subplots
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Set width of bars
        bar_width = 0.25
        
        # Set positions of bars on X axis
        r1 = np.arange(len(class_names))
        r2 = [x + bar_width for x in r1]
        r3 = [x + bar_width for x in r2]
        
        # Create bars
        ax.bar(r1, precision, width=bar_width, label='Precision', color='skyblue')
        ax.bar(r2, recall, width=bar_width, label='Recall', color='lightgreen')
        ax.bar(r3, f1_score, width=bar_width, label='F1-Score', color='salmon')
        
        # Add labels and title
        ax.set_xlabel('Class', fontsize=14)
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title('Per-Class Metrics', fontsize=16)
        ax.set_xticks([r + bar_width for r in range(len(class_names))])
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.legend()
        
        plt.tight_layout()
        
        # Save the plot
        plt.savefig(output_dir / 'class_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Error plotting class metrics: {e}")

def create_synthetic_history():
    """Create synthetic training history for demonstration"""
    np.random.seed(42)
    epochs = 20
    
    # Generate synthetic data
    accuracy = [0.5 + 0.4 * (1 - np.exp(-0.2 * i)) + 0.05 * np.random.randn() for i in range(epochs)]
    val_accuracy = [0.5 + 0.35 * (1 - np.exp(-0.2 * i)) + 0.07 * np.random.randn() for i in range(epochs)]
    
    loss = [0.8 * np.exp(-0.1 * i) + 0.1 + 0.05 * np.random.randn() for i in range(epochs)]
    val_loss = [0.8 * np.exp(-0.1 * i) + 0.15 + 0.07 * np.random.randn() for i in range(epochs)]
    
    # Ensure values are within reasonable ranges
    accuracy = [min(max(acc, 0.5), 0.99) for acc in accuracy]
    val_accuracy = [min(max(acc, 0.45), 0.95) for acc in val_accuracy]
    loss = [max(l, 0.1) for l in loss]
    val_loss = [max(l, 0.15) for l in val_loss]
    
    # Create learning rate schedule
    lr = [0.001 * np.exp(-0.1 * i) for i in range(epochs)]
    
    # Create history dictionary
    history = {
        'accuracy': accuracy,
        'val_accuracy': val_accuracy,
        'loss': loss,
        'val_loss': val_loss,
        'lr': lr
    }
    
    return history

def create_synthetic_confusion_matrix():
    """Create synthetic confusion matrix for demonstration"""
    np.random.seed(42)
    
    # Define class names
    class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-damage', 'Snow-covered']
    n_classes = len(class_names)
    
    # Create confusion matrix with diagonal dominance
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    # Fill diagonal with higher values (correct predictions)
    for i in range(n_classes):
        cm[i, i] = np.random.randint(70, 100)
    
    # Fill off-diagonal with lower values (incorrect predictions)
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                cm[i, j] = np.random.randint(0, 20)
    
    # Create confusion matrix data
    cm_data = {
        'confusion_matrix': cm.tolist(),
        'class_names': class_names
    }
    
    return cm_data

def create_synthetic_class_metrics():
    """Create synthetic class metrics for demonstration"""
    np.random.seed(42)
    
    # Define class names
    class_names = ['Bird-drop', 'Clean', 'Dusty', 'Electrical-damage', 'Physical-damage', 'Snow-covered']
    
    # Create metrics dictionary
    metrics = {}
    
    for cls in class_names:
        precision = 0.7 + 0.2 * np.random.random()
        recall = 0.7 + 0.2 * np.random.random()
        f1 = 2 * precision * recall / (precision + recall)
        
        metrics[cls] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1,
            'support': np.random.randint(50, 150)
        }
    
    return metrics

def main():
    # Create output directory
    output_dir = create_output_dir()
    
    # Check for training history file
    history_file = 'artifacts/training_history.json'
    
    if os.path.exists(history_file):
        print(f"Loading training history from {history_file}...")
        history = load_training_history(history_file)
    else:
        print("Training history file not found. Creating synthetic data for demonstration...")
        history = create_synthetic_history()
        
        # Save synthetic history for reference
        with open(output_dir / 'synthetic_history.json', 'w') as f:
            json.dump(history, f, indent=4)
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    # Plot learning rate if available
    plot_learning_rate(history, output_dir)
    
    # Create training summary
    summary = create_training_summary(history, output_dir)
    print("Training summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Check for confusion matrix file
    confusion_matrix_file = 'artifacts/confusion_matrix.json'
    
    if os.path.exists(confusion_matrix_file):
        print(f"Loading confusion matrix from {confusion_matrix_file}...")
        plot_confusion_matrix(confusion_matrix_file, output_dir)
    else:
        print("Confusion matrix file not found. Creating synthetic data for demonstration...")
        cm_data = create_synthetic_confusion_matrix()
        
        # Save synthetic confusion matrix for reference
        with open(output_dir / 'synthetic_confusion_matrix.json', 'w') as f:
            json.dump(cm_data, f, indent=4)
        
        # Plot synthetic confusion matrix
        with open(output_dir / 'synthetic_confusion_matrix.json', 'r') as f:
            plot_confusion_matrix(output_dir / 'synthetic_confusion_matrix.json', output_dir)
    
    # Check for class metrics file
    metrics_file = 'artifacts/class_metrics.json'
    
    if os.path.exists(metrics_file):
        print(f"Loading class metrics from {metrics_file}...")
        plot_class_metrics(metrics_file, output_dir)
    else:
        print("Class metrics file not found. Creating synthetic data for demonstration...")
        metrics = create_synthetic_class_metrics()
        
        # Save synthetic metrics for reference
        with open(output_dir / 'synthetic_class_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Plot synthetic class metrics
        plot_class_metrics(output_dir / 'synthetic_class_metrics.json', output_dir)
    
    print(f"Visualizations saved to {output_dir}")

if __name__ == "__main__":
    main()
