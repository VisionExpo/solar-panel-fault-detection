from solar_fault_detector.config.config import Config
from solar_fault_detector.data.ingestion import DataIngestor
from solar_fault_detector.data.augmentation import ImageAugmentor
from solar_fault_detector.training.trainer import Trainer

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def run_training(
    model_type: str = "cnn",
    num_ensemble_models: int = 3,
):
    """
    End-to-end training pipeline.
    """
    config = Config()

    # ======================
    # Data ingestion
    # ======================
    ingestor = DataIngestor(config.data)
    data_dirs = ingestor.ingest()

    # ======================
    # Data generators
    # ======================
    augmentor = ImageAugmentor(config.augmentation)

    train_gen: ImageDataGenerator = augmentor.get_train_augmentor()
    val_gen: ImageDataGenerator = augmentor.get_validation_augmentor()

    train_dataset = train_gen.flow_from_directory(
        data_dirs["train"],
        target_size=config.model.img_size,
        batch_size=config.model.batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    val_dataset = val_gen.flow_from_directory(
        data_dirs["val"],
        target_size=config.model.img_size,
        batch_size=config.model.batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    # ======================
    # Training
    # ======================
    trainer = Trainer(
        config=config,
        model_type=model_type,
        num_ensemble_models=num_ensemble_models,
    )

    trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )


if __name__ == "__main__":
    run_training()
