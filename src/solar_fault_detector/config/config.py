from dataclasses import dataclass
from pathlib import Path

# ======================
# Data Configuration
# ======================
@dataclass
class DataConfig:
    root_dir: Path = Path("artifacts")
    data_dir: Path = Path("Faulty_solar_panel")
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_state: int = 42


# ======================
# Model Configuration
# ======================
@dataclass
class ModelConfig:
    img_size: tuple = (384, 384)
    num_channels: int = 3
    num_classes: int = 6
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 5e-4
    early_stopping_patience: int = 8

    model_dir: Path = Path("artifacts/models")
    best_model_path: Path = Path("artifacts/models/best_model.h5")


# ======================
# Training / Experiment Tracking
# ======================
@dataclass
class TrainingConfig:
    use_wandb: bool = True
    wandb_project: str = "solar_panel_fault_detection"
    wandb_entity: str = "gorulevishal984"
    wandb_run_group: str = "baseline"
    wandb_mode: str = "online"   # online | offline | disabled
    log_images: bool = True
    log_confusion_matrix: bool = True


# ======================
# Data Augmentation
# ======================
@dataclass
class AugmentationConfig:
    rotation_range: int = 20
    width_shift_range: float = 0.2
    height_shift_range: float = 0.2
    shear_range: float = 0.2
    zoom_range: float = 0.2
    horizontal_flip: bool = True
    fill_mode: str = "nearest"


# ======================
# Unified Config
# ======================
@dataclass
class Config:
    data: DataConfig = DataConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    augmentation: AugmentationConfig = AugmentationConfig()
