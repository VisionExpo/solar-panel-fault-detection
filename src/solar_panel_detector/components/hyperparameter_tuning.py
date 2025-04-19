import optuna
from optuna.integration import TFKerasPruningCallback
from ..config.configuration import Config
from ..utils.logger import logger
import mlflow
import tensorflow as tf
from typing import Tuple, Dict, Any
import wandb
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import tensorflow_addons as tfa

class HyperParameterTuner:
    def __init__(self, config: Config, train_ds, val_ds):
        self.config = config
        self.train_ds = train_ds
        self.val_ds = val_ds
        
    def objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function for hyperparameter optimization"""
        # Define hyperparameter search space
        hp_values = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'adamw', 'radam']),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'dense_units': trial.suggest_int('dense_units', 128, 512, step=64),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
        }
        
        # Build model with trial hyperparameters
        model = self._build_model(hp_values)
        
        # Train for a few epochs
        history = model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=10,
            callbacks=[
                TFKerasPruningCallback(trial, 'val_accuracy'),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            ]
        )
        
        # Log metrics to MLflow
        metrics = {
            'val_accuracy': max(history.history['val_accuracy']),
            'val_loss': min(history.history['val_loss']),
            **hp_values
        }
        mlflow.log_metrics(metrics)
        
        return metrics['val_accuracy']
    
    def _build_model(self, hp_values: Dict[str, Any]) -> tf.keras.Model:
        """Build model with given hyperparameters"""
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.model.img_size[0], 
                        self.config.model.img_size[1], 
                        self.config.model.num_channels)
        )
        
        # Freeze base model
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(hp_values['dropout_rate']),
            layers.Dense(hp_values['dense_units'], activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(hp_values['dropout_rate']),
            layers.Dense(self.config.model.num_classes, activation='softmax')
        ])
        
        # Configure optimizer
        if hp_values['optimizer'] == 'adam':
            optimizer = tf.keras.optimizers.Adam(learning_rate=hp_values['learning_rate'])
        elif hp_values['optimizer'] == 'adamw':
            optimizer = tfa.optimizers.AdamW(
                learning_rate=hp_values['learning_rate'],
                weight_decay=hp_values['weight_decay']
            )
        else:  # radam
            optimizer = tfa.optimizers.RectifiedAdam(learning_rate=hp_values['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def find_best_hyperparameters(self, n_trials: int = 50) -> Dict[str, Any]:
        """Run hyperparameter optimization study"""
        mlflow.start_run(nested=True)
        
        study = optuna.create_study(
            direction='maximize',
            pruner=optuna.pruners.MedianPruner()
        )
        
        study.optimize(
            self.objective,
            n_trials=n_trials,
            callbacks=[
                lambda study, trial: wandb.log({
                    "best_value": study.best_value,
                    "best_trial": study.best_trial.number
                })
            ]
        )
        
        # Log best parameters
        best_params = study.best_params
        mlflow.log_params(best_params)
        
        # Create optimization history plot
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("artifacts/optimization_history.png")
        mlflow.log_artifact("artifacts/optimization_history.png")
        
        # Create parameter importance plot
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("artifacts/param_importances.png")
        mlflow.log_artifact("artifacts/param_importances.png")
        
        mlflow.end_run()
        
        return best_params