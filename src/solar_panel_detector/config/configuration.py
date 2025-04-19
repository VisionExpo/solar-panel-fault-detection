from dataclasses import dataclass
from pathlib import Path
import os

@dataclass
class DataConfig:
    root_dir: Path = Path("artifacts")
    data_dir: Path = Path("Faulty_solar_panel")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_state: int = 42

@dataclass
class ModelConfig:
    img_size: tuple = (384, 384)  # Increased image size for better performance
    num_channels: int = 3
    num_classes: int = 6
    batch_size: int = 16  # Reduced batch size due to larger model
    epochs: int = 50
    learning_rate: float = 0.0005  # Reduced learning rate for stability
    early_stopping_patience: int = 8  # Increased patience
    model_dir: Path = Path("artifacts/models")
    best_model_path: Path = Path("artifacts/models/best_model.h5")

@dataclass
class TrainingConfig:
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "solar_panel_fault_detection"
    run_name: str = "baseline_run"
    wandb_project: str = "solar_panel_fault_detection"
    wandb_entity: str = "gorulevishal984"  # Add your wandb username here instead of organization

@dataclass
class AugmentationConfig:
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    fill_mode: str = "nearest"

@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    augmentation: AugmentationConfig = AugmentationConfig()