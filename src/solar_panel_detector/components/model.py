import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0, EfficientNetV2S
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
from ..utils.losses import FocalLoss, weighted_categorical_crossentropy

class SolarPanelModel:
    def __init__(self, config: Config):
        self.config = config
        self.model = self._build_model()
        self.history = None

    def _build_model(self):
        """Build EfficientNetV2S model with custom top layers and partial unfreezing"""
        # Use EfficientNetV2S for better performance
        base_model = EfficientNetV2S(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.model.img_size[0],
                        self.config.model.img_size[1],
                        self.config.model.num_channels)
        )

        # Freeze early layers but unfreeze later layers for fine-tuning
        # This allows the model to adapt to our specific dataset
        # EfficientNetV2S has more layers, so we unfreeze more
        for layer in base_model.layers[:-30]:  # Freeze all except last 30 layers
            layer.trainable = False

        # Use Functional API for more flexibility
        inputs = tf.keras.Input(shape=(self.config.model.img_size[0],
                                      self.config.model.img_size[1],
                                      self.config.model.num_channels))
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)

        # Add more capacity with wider layers
        x = layers.Dense(1024, activation='relu')(x)  # Increased capacity
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Dense(512, activation='relu')(x)  # Additional layer
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)

        # Use a smaller learning rate for the final classification layer
        outputs = layers.Dense(self.config.model.num_classes, activation='softmax',
                             kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)

        model = tf.keras.Model(inputs, outputs)

        # Compile with optimizer and learning rate
        # Using native TF optimizer instead of TFA to avoid compatibility issues
        initial_learning_rate = self.config.model.learning_rate

        # Use standard Adam optimizer instead of AdamW from TFA
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=initial_learning_rate
        )

        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
        )

        return model

    def train(self, train_ds, val_ds, label_mapping):
        """Train the model with advanced callbacks, monitoring, and class weights"""
        # Create model directory
        self.config.model.model_dir.mkdir(parents=True, exist_ok=True)

        # Calculate class weights to handle imbalance
        class_counts = {}
        for _, labels in train_ds:
            for label in labels.numpy():
                if label not in class_counts:
                    class_counts[label] = 0
                class_counts[label] += 1

        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)

        # Compute balanced class weights
        class_weights = {}
        for class_idx, count in class_counts.items():
            # Formula: total_samples / (num_classes * count)
            class_weights[class_idx] = total_samples / (num_classes * count)

        logger.info(f"Using class weights: {class_weights}")

        # Start nested MLflow run
        with mlflow.start_run(nested=True):
            # Initialize callbacks with improved learning rate schedule
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
                    patience=2,  # More aggressive LR reduction
                    min_lr=1e-7
                ),
                # Add TensorBoard callback for better visualization
                tf.keras.callbacks.TensorBoard(
                    log_dir=str(self.config.model.model_dir / 'logs'),
                    histogram_freq=1
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
                        "architecture": "EfficientNetB0",
                        "class_weights": class_weights
                    }
                )
                callbacks.append(wandb.keras.WandbMetricsLogger())
                wandb_enabled = True
            except Exception as e:
                logger.warning(f"WandB initialization failed, continuing without WandB logging: {str(e)}")
                wandb_enabled = False

            # Log class weights to MLflow
            mlflow.log_params({f"class_weight_{k}": v for k, v in class_weights.items()})

            # Train model with class weights
            self.history = self.model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=self.config.model.epochs,
                callbacks=callbacks,
                class_weight=class_weights  # Apply class weights
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