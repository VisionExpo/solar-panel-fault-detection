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
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
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
        
        # Evaluate model
        logger.info("Evaluating model...")
        results = model.evaluate(test_ds)
        metrics = dict(zip(model.metrics_names, results))
        
        print("\nModel Evaluation Metrics:")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        
        # Get predictions
        y_pred_proba = model.predict(test_ds)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Get true labels
        y_true = []
        for _, labels in test_ds:
            y_true.extend(labels.numpy())
        y_true = np.array(y_true)
        
        # Calculate metrics
        report = classification_report(
            y_true, 
            y_pred, 
            target_names=list(label_mapping.keys()),
            output_dict=True
        )
        
        # Print classification report
        print("\nClassification Report:")
        report_df = pd.DataFrame(report).transpose()
        print(report_df.round(3))
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
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
        plt.title('Confusion Matrix')
        plt.tight_layout()
        
        # Save plot
        cm_plot_path = Path("artifacts/models/evaluation_confusion_matrix.png")
        plt.savefig(cm_plot_path)
        
        logger.info(f"Confusion matrix saved to {cm_plot_path}")
        
        # Calculate per-class metrics
        print("\nPer-Class Metrics:")
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"{class_name}:")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall: {metrics['recall']:.4f}")
                print(f"  F1-score: {metrics['f1-score']:.4f}")
                print(f"  Support: {metrics['support']}")
                print()
        
        # Calculate overall accuracy
        print(f"Overall Accuracy: {report['accuracy']:.4f}")
        
    except Exception as e:
        logger.error(f"Error in evaluation: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
