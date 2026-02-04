import wandb
import tensorflow as tf

from solar_fault_detector.config.config import Config
from solar_fault_detector.training.trainer import Trainer


def sweep_train():
    """
    Training function executed by W&B sweep agent.
    Hyperparameters are injected via wandb.config.
    """
    wandb.init()

    cfg = Config()

    # Override config values from sweep
    cfg.model.learning_rate = wandb.config.learning_rate
    cfg.model.batch_size = wandb.config.batch_size
    cfg.model.epochs = wandb.config.epochs
    model_type = wandb.config.model_type

    trainer = Trainer(
        config=cfg,
        model_type=model_type,
        num_ensemble_models=wandb.config.get("num_ensemble_models", 3),
    )

    # NOTE:
    # train_dataset and val_dataset should be constructed
    # in the pipeline layer and passed here if needed.
    train_dataset = wandb.config.train_dataset
    val_dataset = wandb.config.val_dataset

    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


def run_sweep(sweep_config: dict, project: str):
    """
    Launch a W&B sweep.
    """
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
    )
    wandb.agent(sweep_id, function=sweep_train)
