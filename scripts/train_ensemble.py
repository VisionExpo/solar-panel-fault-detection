import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.config.configuration import Config
from src.solar_panel_detector.components.data_preparation import DataPreparation
from src.solar_panel_detector.components.model_factory import ModelFactory, EnsembleModel
from src.solar_panel_detector.utils.logger import logger
import mlflow
from dotenv import load_dotenv
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json

def train_model(config, train_ds, val_ds, model_type, epochs=30):
    """Train a single model"""
    logger.info(f"Training {model_type} model...")
    
    # Create model
    factory = ModelFactory(config)
    model = factory.create_model(model_type)
    
    # Create callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            f"artifacts/models/{model_type}_best.h5",
            save_best_only=True,
            monitor='val_accuracy'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=1e-7
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=f"artifacts/logs/{model_type}",
            histogram_freq=1
        )
    ]
    
    # Train model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    
    # Load best weights
    model.load_weights(f"artifacts/models/{model_type}_best.h5")
    
    return model, history

def evaluate_ensemble(ensemble, test_ds, label_mapping):
    """Evaluate ensemble model"""
    logger.info("Evaluating ensemble model...")
    
    # Get predictions
    y_pred_proba = ensemble.predict(test_ds)
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
    cm_plot_path = Path("artifacts/models/ensemble_confusion_matrix.png")
    plt.savefig(cm_plot_path)
    
    # Log to MLflow
    mlflow.log_artifact(str(cm_plot_path))
    
    # Log metrics
    for class_name, metrics in report.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            mlflow.log_metrics({
                f"ensemble_{class_name}_precision": metrics['precision'],
                f"ensemble_{class_name}_recall": metrics['recall'],
                f"ensemble_{class_name}_f1": metrics['f1-score']
            })
    
    mlflow.log_metric("ensemble_accuracy", report['accuracy'])
    
    logger.info(f"Ensemble Evaluation Report:\n{report}")
    
    return report

def main():
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.training.mlflow_tracking_uri)
        mlflow.set_experiment(f"{config.training.experiment_name}_ensemble")
        
        with mlflow.start_run(run_name="ensemble_training"):
            # Prepare data
            logger.info("Preparing datasets...")
            data_preparation = DataPreparation(config)
            train_ds, val_ds, test_ds, label_mapping = data_preparation.prepare_data()
            
            # Save label mapping
            os.makedirs("artifacts/models", exist_ok=True)
            with open("artifacts/models/label_mapping.json", 'w') as f:
                json.dump(label_mapping, f)
            
            # Train individual models
            model_types = ['efficientnetv2s', 'resnet50v2', 'densenet121']
            trained_models = []
            
            for model_type in model_types:
                with mlflow.start_run(run_name=f"{model_type}_training", nested=True):
                    model, _ = train_model(config, train_ds, val_ds, model_type)
                    trained_models.append(model)
            
            # Create ensemble
            logger.info("Creating ensemble model...")
            ensemble = EnsembleModel(trained_models)
            
            # Evaluate ensemble
            evaluation_report = evaluate_ensemble(ensemble, test_ds, label_mapping)
            
            # Save ensemble
            logger.info("Saving ensemble model...")
            ensemble_path = "artifacts/models/ensemble"
            os.makedirs(ensemble_path, exist_ok=True)
            ensemble.save(ensemble_path)
            
            logger.info("Ensemble training completed successfully")
            
    except Exception as e:
        logger.error(f"Error in ensemble training: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
