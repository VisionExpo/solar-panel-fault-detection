import tensorflow as tf
from tensorflow.keras import layers, models
import tensorflow_addons as tfa
from tensorflow.keras.applications import EfficientNetB0
import mlflow
import wandb
from ..utils.logger import logger
from ..config.configuration import Config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path
import json

class SolarPanelModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = self._build_model()
        self.history = None
        
    def _build_model(self):
        """Build EfficientNet model with custom top layers"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.model.img_size[0], 
                        self.config.model.img_size[1], 
                        self.config.model.num_channels)
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.config.model.num_classes, activation='softmax')
        ])
        
        # Compile with advanced optimizer and learning rate schedule
        initial_learning_rate = self.config.model.learning_rate
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=1000,
            decay_rate=0.9,
            staircase=True
        )
        
        optimizer = tfa.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=0.0001
        )
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )
        
        return model
    
    def train(self, train_ds, val_ds, label_mapping):
        """Train the model with advanced callbacks and monitoring"""
        # Create model directory
        self.config.model.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Start nested MLflow run
        with mlflow.start_run(nested=True):
            # Initialize callbacks
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    str(self.config.model.best_model_path),
                    save_best_only=True,
                    monitor='val_accuracy'
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.model.early_stopping_patience,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3,
                    min_lr=1e-6
                )
            ]
            
            # Try to initialize W&B if available
            try:
                import wandb
                wandb.init(
                    project=self.config.training.wandb_project,
                    entity=self.config.training.wandb_entity,
                    config={
                        "learning_rate": self.config.model.learning_rate,
                        "epochs": self.config.model.epochs,
                        "batch_size": self.config.model.batch_size,
                        "img_size": self.config.model.img_size,
                        "architecture": "EfficientNetB0"
                    }
                )
                callbacks.append(wandb.keras.WandbMetricsLogger())
                wandb_enabled = True
            except Exception as e:
                logger.warning(f"WandB initialization failed, continuing without WandB logging: {str(e)}")
                wandb_enabled = False
            
            # Train model
            self.history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.model.epochs,
                callbacks=callbacks
            )
            
            # Save label mapping
            with open(self.config.model.model_dir / 'label_mapping.json', 'w') as f:
                json.dump(label_mapping, f)
            
            # Log metrics and artifacts
            self._log_training_results()
            
            if wandb_enabled:
                wandb.finish()
        
    def _log_training_results(self):
        """Log training metrics and generate visualization artifacts"""
        # Plot and save training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        # Save plot
        history_plot_path = self.config.model.model_dir / 'training_history.png'
        plt.savefig(history_plot_path)
        plt.close()
        
        # Log metrics to MLflow
        mlflow.log_metrics({
            'final_train_loss': self.history.history['loss'][-1],
            'final_train_accuracy': self.history.history['accuracy'][-1],
            'final_val_loss': self.history.history['val_loss'][-1],
            'final_val_accuracy': self.history.history['val_accuracy'][-1]
        })
        
        # Log artifacts
        mlflow.log_artifact(str(history_plot_path))
        mlflow.log_artifact(str(self.config.model.best_model_path))
        
    def evaluate(self, test_ds, label_mapping):
        """Evaluate model on test set and generate detailed metrics"""
        # Get predictions
        y_pred = []
        y_true = []
        
        for images, labels in test_ds:
            predictions = self.model.predict(images)
            y_pred.extend(np.argmax(predictions, axis=1))
            y_true.extend(labels.numpy())
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=list(label_mapping.keys()),
                   yticklabels=list(label_mapping.keys()))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        
        # Save confusion matrix plot
        cm_plot_path = self.config.model.model_dir / 'confusion_matrix.png'
        plt.savefig(cm_plot_path)
        plt.close()
        
        # Generate classification report
        report = classification_report(y_true, y_pred, 
                                    target_names=list(label_mapping.keys()),
                                    output_dict=True)
        
        # Log evaluation metrics
        mlflow.log_metrics({
            f"{class_name}_f1": metrics['f1-score']
            for class_name, metrics in report.items()
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']
        })
        
        mlflow.log_artifact(str(cm_plot_path))
        
        return report
    
    def save_model_for_serving(self):
        """Save model in TF SavedModel format for deployment"""
        serving_path = self.config.model.model_dir / 'serving'
        tf.saved_model.save(self.model, str(serving_path))
        logger.info(f"Model saved for serving at {serving_path}")
        return serving_path