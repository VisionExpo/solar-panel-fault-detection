import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.config.configuration import Config
from src.solar_panel_detector.components.data_preparation import DataPreparation
from src.solar_panel_detector.components.model_factory import ModelFactory
from src.solar_panel_detector.utils.logger import logger
import mlflow
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

def main():
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.training.mlflow_tracking_uri)
        mlflow.set_experiment(f"{config.training.experiment_name}_improved")
        
        with mlflow.start_run(run_name="improved_model"):
            # Prepare data
            logger.info("Preparing datasets...")
            data_preparation = DataPreparation(config)
            train_ds, val_ds, test_ds, label_mapping = data_preparation.prepare_data()
            
            # Save label mapping
            os.makedirs("artifacts/models", exist_ok=True)
            with open("artifacts/models/label_mapping.json", 'w') as f:
                json.dump(label_mapping, f)
            
            # Create model
            logger.info("Creating model...")
            factory = ModelFactory(config)
            model = factory.create_model('efficientnetv2s')
            
            # Create callbacks
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    "artifacts/models/best_model.h5",
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=8,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-7
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir="artifacts/logs",
                    histogram_freq=1
                )
            ]
            
            # Train model
            logger.info("Training model...")
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=50,
                callbacks=callbacks
            )
            
            # Evaluate model
            logger.info("Evaluating model...")
            results = model.evaluate(test_ds)
            
            # Log metrics
            metrics = dict(zip(model.metrics_names, results))
            for name, value in metrics.items():
                mlflow.log_metric(name, value)
                
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
            cm_plot_path = Path("artifacts/models/confusion_matrix.png")
            plt.savefig(cm_plot_path)
            
            # Log to MLflow
            mlflow.log_artifact(str(cm_plot_path))
            
            # Log metrics
            for class_name, metrics in report.items():
                if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                    mlflow.log_metrics({
                        f"{class_name}_precision": metrics['precision'],
                        f"{class_name}_recall": metrics['recall'],
                        f"{class_name}_f1": metrics['f1-score']
                    })
            
            # Save model
            logger.info("Saving model...")
            model.save("artifacts/models/final_model")
            
            logger.info("Training completed successfully")
            
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
