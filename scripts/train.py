import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.config.configuration import Config
from src.solar_panel_detector.pipeline.train_pipeline import TrainingPipeline
from src.solar_panel_detector.utils.logger import logger
import mlflow
from dotenv import load_dotenv

def main():
    # Load environment variables
    load_dotenv()
    
    try:
        # Initialize configuration
        config = Config()
        
        # Initialize MLflow
        mlflow.set_tracking_uri(config.training.mlflow_tracking_uri)
        mlflow.set_experiment(config.training.experiment_name)
        
        # Run training pipeline
        pipeline = TrainingPipeline(config)
        pipeline.run()
        
        logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Error in training: {str(e)}")
        raise e

if __name__ == "__main__":
    main()