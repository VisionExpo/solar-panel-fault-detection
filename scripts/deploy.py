import os
import sys
from pathlib import Path
import subprocess
import json
import time
from datetime import datetime

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.solar_panel_detector.utils.logger import logger
from src.solar_panel_detector.components.data_preparation import DataPreparation
from src.solar_panel_detector.config.configuration import Config
from src.solar_panel_detector.utils.model_optimization import ModelOptimizer
import requests

def validate_model_performance(config: Config) -> bool:
    """Validate model performance meets requirements"""
    try:
        # Load test data
        data_prep = DataPreparation(config)
        _, _, test_ds, _ = data_prep.prepare_data()
        
        # Run benchmarks
        metrics = ModelOptimizer.benchmark_model(
            config.model.best_model_path,
            test_ds
        )
        
        # Define performance requirements
        requirements = {
            'mean_inference_time_ms': 200,  # Max 200ms inference time
            'throughput_imgs_per_sec': 100,  # Min 100 images per second
        }
        
        # Check if performance meets requirements
        meets_requirements = all(
            metrics.get(metric, float('inf')) <= threshold 
            if 'time' in metric else 
            metrics.get(metric, 0) >= threshold
            for metric, threshold in requirements.items()
        )
        
        if meets_requirements:
            logger.info("Model performance meets requirements")
        else:
            logger.error("Model performance does not meet requirements")
            logger.error(f"Required: {requirements}")
            logger.error(f"Actual: {metrics}")
        
        return meets_requirements
        
    except Exception as e:
        logger.error(f"Error validating model performance: {str(e)}")
        return False

def optimize_for_deployment(config: Config) -> bool:
    """Optimize model for deployment"""
    try:
        # Run optimization script
        optimize_script = project_root / 'scripts' / 'optimize_model.py'
        result = subprocess.run(
            [sys.executable, str(optimize_script)],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Optimization failed: {result.stderr}")
            return False
        
        logger.info("Model optimization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error optimizing model: {str(e)}")
        return False

def deploy_to_render() -> bool:
    """Deploy application to Render"""
    try:
        # Check for required environment variables
        required_vars = ['RENDER_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        # Deploy using Render CLI
        result = subprocess.run(
            ['render', 'deploy'],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.error(f"Deployment failed: {result.stderr}")
            return False
        
        logger.info("Deployment to Render completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error deploying to Render: {str(e)}")
        return False

def verify_deployment(service_url: str, timeout: int = 300) -> bool:
    """Verify deployment is successful and service is healthy"""
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            # Check health endpoint
            response = requests.get(f"{service_url}/health")
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get('status') == 'healthy':
                    logger.info("Deployment verification successful")
                    return True
        except:
            pass
        
        time.sleep(10)
        logger.info("Waiting for service to become healthy...")
    
    logger.error("Deployment verification timed out")
    return False

def main():
    try:
        config = Config()
        deployment_log = {
            'timestamp': datetime.now().isoformat(),
            'steps': []
        }
        
        # Step 1: Validate model performance
        logger.info("Step 1: Validating model performance")
        if not validate_model_performance(config):
            raise ValueError("Model validation failed")
        deployment_log['steps'].append({
            'step': 'validate_model',
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
        # Step 2: Optimize model
        logger.info("Step 2: Optimizing model for deployment")
        if not optimize_for_deployment(config):
            raise ValueError("Model optimization failed")
        deployment_log['steps'].append({
            'step': 'optimize_model',
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
        # Step 3: Deploy to Render
        logger.info("Step 3: Deploying to Render")
        if not deploy_to_render():
            raise ValueError("Deployment failed")
        deployment_log['steps'].append({
            'step': 'deploy',
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
        # Step 4: Verify deployment
        logger.info("Step 4: Verifying deployment")
        service_url = os.getenv('RENDER_SERVICE_URL')
        if not service_url:
            raise ValueError("RENDER_SERVICE_URL environment variable not set")
            
        if not verify_deployment(service_url):
            raise ValueError("Deployment verification failed")
        deployment_log['steps'].append({
            'step': 'verify',
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        })
        
        # Save deployment log
        log_dir = Path('artifacts/deployment_logs')
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"deployment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump(deployment_log, f, indent=4)
            
        logger.info("Deployment completed successfully")
        
    except Exception as e:
        logger.error(f"Deployment failed: {str(e)}")
        raise e

if __name__ == "__main__":
    main()