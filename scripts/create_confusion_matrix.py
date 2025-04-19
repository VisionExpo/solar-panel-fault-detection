import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.config.configuration import Config
from src.solar_panel_detector.components.data_preparation import DataPreparation
from src.solar_panel_detector.utils.logger import logger
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
import pandas as pd

def main():
    try:
        # Initialize configuration
        config = Config()
        
        # Load label mapping
        with open("artifacts/models/label_mapping.json", 'r') as f:
            label_mapping = json.load(f)
            
        # Prepare test dataset
        logger.info("Preparing test dataset...")
        data_preparation = DataPreparation(config)
        _, _, test_ds, _ = data_preparation.prepare_data()
        
        # Load model
        logger.info("Loading model...")
        model_path = "artifacts/models/final_model"
        model = tf.keras.models.load_model(model_path)
        
        # Get predictions
        logger.info("Making predictions...")
        y_pred_proba = model.predict(test_ds)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Get true labels
        y_true = []
        for _, labels in test_ds:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure for visualization
        plt.figure(figsize=(12, 10))
        
        # Plot normalized confusion matrix
        sns.heatmap(
            cm_norm, 
            annot=True, 
            fmt='.2f', 
            cmap='Blues',
            xticklabels=list(label_mapping.keys()),
            yticklabels=list(label_mapping.keys())
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()
        
        # Save normalized confusion matrix
        plt.savefig("artifacts/models/normalized_confusion_matrix.png")
        logger.info("Normalized confusion matrix saved to artifacts/models/normalized_confusion_matrix.png")
        
        # Plot raw confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=list(label_mapping.keys()),
            yticklabels=list(label_mapping.keys())
        )
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix (Raw Counts)')
        plt.tight_layout()
        
        # Save raw confusion matrix
        plt.savefig("artifacts/models/raw_confusion_matrix.png")
        logger.info("Raw confusion matrix saved to artifacts/models/raw_confusion_matrix.png")
        
        # Calculate and print classification report
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=list(label_mapping.keys()),
            output_dict=True
        )
        
        # Save classification report as CSV
        report_df = pd.DataFrame(report).transpose()
        report_df.to_csv("artifacts/models/classification_report.csv")
        logger.info("Classification report saved to artifacts/models/classification_report.csv")
        
        # Print summary
        print("\nClassification Report:")
        print(report_df.round(3))
        
        # Calculate per-class metrics
        class_metrics = {}
        for class_name, class_idx in label_mapping.items():
            # True positives, false positives, false negatives
            tp = cm[class_idx, class_idx]
            fp = cm[:, class_idx].sum() - tp
            fn = cm[class_idx, :].sum() - tp
            
            # Calculate precision, recall, f1-score
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[class_name] = {
                'precision': precision,
                'recall': recall,
                'f1-score': f1,
                'support': cm[class_idx, :].sum()
            }
        
        # Create a bar chart for recall by class
        plt.figure(figsize=(12, 6))
        recalls = [metrics['recall'] for class_name, metrics in class_metrics.items()]
        plt.bar(list(label_mapping.keys()), recalls)
        plt.xlabel('Class')
        plt.ylabel('Recall')
        plt.title('Recall by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("artifacts/models/recall_by_class.png")
        logger.info("Recall by class chart saved to artifacts/models/recall_by_class.png")
        
        # Create a bar chart for precision by class
        plt.figure(figsize=(12, 6))
        precisions = [metrics['precision'] for class_name, metrics in class_metrics.items()]
        plt.bar(list(label_mapping.keys()), precisions)
        plt.xlabel('Class')
        plt.ylabel('Precision')
        plt.title('Precision by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("artifacts/models/precision_by_class.png")
        logger.info("Precision by class chart saved to artifacts/models/precision_by_class.png")
        
        # Create a bar chart for F1-score by class
        plt.figure(figsize=(12, 6))
        f1_scores = [metrics['f1-score'] for class_name, metrics in class_metrics.items()]
        plt.bar(list(label_mapping.keys()), f1_scores)
        plt.xlabel('Class')
        plt.ylabel('F1-Score')
        plt.title('F1-Score by Class')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("artifacts/models/f1_score_by_class.png")
        logger.info("F1-score by class chart saved to artifacts/models/f1_score_by_class.png")
        
    except Exception as e:
        logger.error(f"Error in creating confusion matrix: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
