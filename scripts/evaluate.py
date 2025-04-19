import os
import sys
from pathlib import Path
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import mlflow
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.config.configuration import Config
from src.solar_panel_detector.components.data_preparation import DataPreparation
from src.solar_panel_detector.utils.logger import logger

def plot_confusion_matrix(cm, class_names, save_path):
    """Plot confusion matrix with proper formatting"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_class_distribution(class_counts, save_path):
    """Plot class distribution"""
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
    plt.xticks(rotation=45)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    try:
        # Initialize configuration
        config = Config()
        
        # Load the model
        model = tf.keras.models.load_model(str(config.model.best_model_path))
        
        # Load label mapping
        with open(config.model.model_dir / 'label_mapping.json', 'r') as f:
            label_mapping = json.load(f)
        
        # Prepare evaluation data
        data_prep = DataPreparation(config)
        _, _, test_ds, _ = data_prep.prepare_data()
        
        # Make predictions
        all_predictions = []
        all_labels = []
        
        for images, labels in test_ds:
            predictions = model.predict(images)
            pred_classes = np.argmax(predictions, axis=1)
            all_predictions.extend(pred_classes)
            all_labels.extend(labels.numpy())
        
        # Calculate metrics
        cm = confusion_matrix(all_labels, all_predictions)
        class_names = list(label_mapping.keys())
        report = classification_report(all_labels, all_predictions, 
                                    target_names=class_names,
                                    output_dict=True)
        
        # Create evaluation directory
        eval_dir = Path('artifacts/evaluation')
        eval_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save confusion matrix plot
        cm_path = eval_dir / f'confusion_matrix_{timestamp}.png'
        plot_confusion_matrix(cm, class_names, cm_path)
        
        # Calculate and plot class distribution
        class_counts = {name: len([x for x in all_labels if x == idx])
                       for name, idx in label_mapping.items()}
        dist_path = eval_dir / f'class_distribution_{timestamp}.png'
        plot_class_distribution(class_counts, dist_path)
        
        # Save metrics to file
        metrics_path = eval_dir / f'metrics_{timestamp}.json'
        metrics = {
            'classification_report': report,
            'class_distribution': class_counts
        }
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Log to MLflow
        mlflow.start_run()
        mlflow.log_metrics({
            f"{class_name}_f1": metrics['f1-score']
            for class_name, metrics in report.items()
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']
        })
        mlflow.log_artifact(str(cm_path))
        mlflow.log_artifact(str(dist_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.end_run()
        
        logger.info(f"Evaluation completed. Results saved to {eval_dir}")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise e

if __name__ == "__main__":
    main()