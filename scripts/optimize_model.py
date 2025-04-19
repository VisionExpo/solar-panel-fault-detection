import os
import sys
from pathlib import Path
import tensorflow as tf
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.config.configuration import Config
from src.solar_panel_detector.components.data_preparation import DataPreparation
from src.solar_panel_detector.utils.model_optimization import ModelOptimizer
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
        mlflow.set_experiment(f"{config.training.experiment_name}_optimization")
        
        # Load data for calibration
        data_prep = DataPreparation(config)
        _, val_ds, test_ds, _ = data_prep.prepare_data()
        
        # Paths for optimized models
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        opt_dir = Path('artifacts/models/optimized') / timestamp
        opt_dir.mkdir(parents=True, exist_ok=True)
        
        # Original model path
        original_model_path = config.model.best_model_path
        
        with mlflow.start_run():
            # Benchmark original model
            logger.info("Benchmarking original model...")
            original_metrics = ModelOptimizer.benchmark_model(
                original_model_path, test_ds
            )
            mlflow.log_metrics({
                'original_' + k: v for k, v in original_metrics.items()
            })
            
            # TensorRT optimization
            logger.info("Optimizing model with TensorRT...")
            tensorrt_path = opt_dir / 'tensorrt'
            ModelOptimizer.optimize_for_inference(
                original_model_path, tensorrt_path
            )
            
            # Benchmark TensorRT model
            tensorrt_metrics = ModelOptimizer.benchmark_model(
                tensorrt_path, test_ds
            )
            mlflow.log_metrics({
                'tensorrt_' + k: v for k, v in tensorrt_metrics.items()
            })
            
            # Quantization
            logger.info("Quantizing model...")
            quantized_path = opt_dir / 'quantized/model.tflite'
            ModelOptimizer.quantize_model(
                original_model_path, quantized_path, val_ds
            )
            
            # Graph optimization
            logger.info("Optimizing computation graph...")
            graph_opt_path = opt_dir / 'graph_optimized'
            ModelOptimizer.optimize_graph(
                original_model_path, graph_opt_path
            )
            
            # Benchmark graph optimized model
            graph_opt_metrics = ModelOptimizer.benchmark_model(
                graph_opt_path, test_ds
            )
            mlflow.log_metrics({
                'graph_opt_' + k: v for k, v in graph_opt_metrics.items()
            })
            
            # Log artifacts
            mlflow.log_artifacts(str(opt_dir))
            
            # Log optimization summary
            speedup_tensorrt = original_metrics['mean_inference_time_ms'] / \
                             tensorrt_metrics['mean_inference_time_ms']
            speedup_graph = original_metrics['mean_inference_time_ms'] / \
                          graph_opt_metrics['mean_inference_time_ms']
            
            logger.info(f"TensorRT Speedup: {speedup_tensorrt:.2f}x")
            logger.info(f"Graph Optimization Speedup: {speedup_graph:.2f}x")
            
            # Copy best performing model to serving directory
            best_speedup = max(speedup_tensorrt, speedup_graph)
            if best_speedup > 1.2:  # Only use optimized if >20% faster
                if speedup_tensorrt > speedup_graph:
                    serving_path = tensorrt_path
                    logger.info("Using TensorRT optimized model for serving")
                else:
                    serving_path = graph_opt_path
                    logger.info("Using Graph optimized model for serving")
                
                # Copy to serving location
                serving_dir = Path("artifacts/models/serving")
                if serving_dir.exists():
                    import shutil
                    shutil.rmtree(serving_dir)
                shutil.copytree(serving_path, serving_dir)
            else:
                logger.info("Using original model for serving (optimization gains insufficient)")
        
        logger.info("Model optimization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model optimization: {str(e)}")
        raise e

if __name__ == "__main__":
    main()