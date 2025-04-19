import os
from ..components.data_ingestion import DataIngestion
from ..components.data_preparation import DataPreparation
from ..components.model import SolarPanelModel
from ..config.configuration import Config
from ..utils.logger import logger
import mlflow
import wandb

class TrainingPipeline:
    def __init__(self, config: Config):
        self.config = config

    def run(self, skip_data_ingestion=False):
        """Execute the complete training pipeline

        Args:
            skip_data_ingestion (bool, optional): Skip the data ingestion phase. Defaults to False.
        """
        try:
            logger.info("Starting training pipeline")

            # Initialize MLflow
            mlflow.set_tracking_uri(self.config.training.mlflow_tracking_uri)
            mlflow.set_experiment(self.config.training.experiment_name)

            if not skip_data_ingestion:
                # Data ingestion
                logger.info("Starting data ingestion")
                data_ingestion = DataIngestion(self.config)
                category_counts = data_ingestion.count_category_data()
                logger.info(f"Current dataset statistics: {category_counts}")

                # Download more images if needed
                for category in data_ingestion.categories:
                    if category_counts.get(category, 0) < 5000:
                        logger.info(f"Downloading additional images for {category}")
                        data_ingestion.download_category_images(category)
            else:
                logger.info("Skipping data ingestion phase")

            # Data preparation
            logger.info("Preparing datasets")
            data_preparation = DataPreparation(self.config)
            train_ds, val_ds, test_ds, label_mapping = data_preparation.prepare_data()

            # Model training
            logger.info("Starting model training")
            model = SolarPanelModel(self.config)
            model.train(train_ds, val_ds, label_mapping)

            # Model evaluation
            logger.info("Evaluating model")
            evaluation_report = model.evaluate(test_ds, label_mapping)
            logger.info(f"Evaluation Report:\n{evaluation_report}")

            # Save model for deployment
            logger.info("Saving model for deployment")
            serving_path = model.save_model_for_serving()
            logger.info(f"Model saved at: {serving_path}")

            logger.info("Training pipeline completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            raise e