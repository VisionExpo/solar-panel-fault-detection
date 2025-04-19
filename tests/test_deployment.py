import os
import sys
import pytest
import requests
from pathlib import Path
import json
import time
from PIL import Image
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def test_service_health(base_url):
    """Test the health check endpoint"""
    response = requests.get(f"{base_url}/health")
    assert response.status_code == 200
    data = response.json()
    
    assert 'status' in data
    assert data['status'] == 'healthy'
    assert 'metrics' in data
    assert 'timestamp' in data

def test_single_prediction(base_url):
    """Test single image prediction endpoint"""
    # Get a test image
    test_image_path = project_root / 'Faulty_solar_panel/Clean/Clean (1).jpg'
    assert test_image_path.exists(), "Test image not found"
    
    with open(test_image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{base_url}/predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert 'prediction' in data
    assert 'confidence' in data
    assert 'inference_time_ms' in data
    assert 'top_3_predictions' in data
    
    # Validate prediction
    assert data['prediction'] in ['Bird-drop', 'Clean', 'Dusty', 
                                'Electrical-damage', 'Physical-Damage', 'Snow-Covered']
    assert 0 <= data['confidence'] <= 1
    assert data['inference_time_ms'] > 0
    assert len(data['top_3_predictions']) == 3

def test_batch_prediction(base_url):
    """Test batch prediction endpoint"""
    # Get test images
    test_image_paths = [
        project_root / 'Faulty_solar_panel/Clean/Clean (1).jpg',
        project_root / 'Faulty_solar_panel/Dusty/Dusty (1).jpg'
    ]
    
    for path in test_image_paths:
        assert path.exists(), f"Test image not found: {path}"
    
    files = [('images', open(path, 'rb')) for path in test_image_paths]
    response = requests.post(f"{base_url}/batch_predict", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert 'results' in data
    assert 'inference_time_ms' in data
    assert 'batch_size' in data
    
    # Validate results
    assert len(data['results']) == len(test_image_paths)
    assert data['batch_size'] == len(test_image_paths)
    assert data['inference_time_ms'] > 0
    
    for result in data['results']:
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'top_3_predictions' in result
        assert 0 <= result['confidence'] <= 1

def test_metrics_endpoint(base_url):
    """Test metrics endpoint"""
    response = requests.get(f"{base_url}/metrics")
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert 'performance_metrics' in data
    assert 'resource_usage' in data
    
    metrics = data['performance_metrics']
    assert 'average_inference_time_ms' in metrics
    assert 'predictions_per_class' in metrics

def test_dashboard_endpoint(base_url):
    """Test dashboard endpoint"""
    response = requests.get(f"{base_url}/dashboard")
    assert response.status_code in [200, 404]  # 404 is acceptable if no dashboard yet
    
    if response.status_code == 200:
        assert response.headers['content-type'] == 'text/html; charset=utf-8'

def test_rate_limiting(base_url):
    """Test rate limiting functionality"""
    # Test single prediction rate limit
    responses = []
    for _ in range(35):  # Limit is 30 per minute
        with open(project_root / 'Faulty_solar_panel/Clean/Clean (1).jpg', 'rb') as f:
            response = requests.post(f"{base_url}/predict", files={'image': f})
            responses.append(response.status_code)
    
    assert 429 in responses  # Should hit rate limit
    
    # Test batch prediction rate limit
    responses = []
    files = [('images', open(project_root / 'Faulty_solar_panel/Clean/Clean (1).jpg', 'rb'))]
    for _ in range(15):  # Limit is 10 per minute
        response = requests.post(f"{base_url}/batch_predict", files=files)
        responses.append(response.status_code)
    
    assert 429 in responses  # Should hit rate limit

def test_error_handling(base_url):
    """Test error handling"""
    # Test invalid image
    with open(__file__, 'rb') as f:  # Send a Python file instead of an image
        response = requests.post(f"{base_url}/predict", files={'image': f})
    assert response.status_code == 400
    
    # Test missing image
    response = requests.post(f"{base_url}/predict")
    assert response.status_code == 400
    
    # Test batch size limit
    files = [('images', open(project_root / 'Faulty_solar_panel/Clean/Clean (1).jpg', 'rb')) 
             for _ in range(50)]  # Default limit is 32
    response = requests.post(f"{base_url}/batch_predict", files=files)
    assert response.status_code == 400

@pytest.fixture
def base_url():
    """Get base URL for testing"""
    url = os.getenv('TEST_API_URL', 'http://localhost:5000')
    return url.rstrip('/')

def verify_deployment_readiness():
    """Verify all requirements for deployment are met"""
    validations = []
    
    # Check required files exist
    required_files = [
        'app.py',
        'Dockerfile',
        'requirements.txt',
        'render.yaml',
        '.env.example'
    ]
    
    for file in required_files:
        exists = Path(project_root / file).exists()
        validations.append({
            'check': f"Required file: {file}",
            'status': 'PASS' if exists else 'FAIL'
        })
    
    # Check model artifacts
    model_path = project_root / 'artifacts/models/serving'
    label_mapping_path = model_path / 'label_mapping.json'
    
    validations.extend([
        {
            'check': 'Model artifacts directory',
            'status': 'PASS' if model_path.exists() else 'FAIL'
        },
        {
            'check': 'Label mapping file',
            'status': 'PASS' if label_mapping_path.exists() else 'FAIL'
        }
    ])
    
    # Check environment variables
    required_env_vars = [
        'PORT',
        'MODEL_BATCH_SIZE',
        'MODEL_NUM_WORKERS'
    ]
    
    for var in required_env_vars:
        exists = var in os.environ
        validations.append({
            'check': f"Environment variable: {var}",
            'status': 'PASS' if exists else 'WARN'
        })
    
    return validations

if __name__ == '__main__':
    # Run deployment validation
    print("\nRunning deployment validation checks...")
    validations = verify_deployment_readiness()
    
    # Print results
    print("\nDeployment Validation Results:")
    print("-" * 50)
    
    has_failures = False
    for check in validations:
        status = check['status']
        if status == 'FAIL':
            has_failures = True
        
        status_color = {
            'PASS': '\033[92m',  # Green
            'FAIL': '\033[91m',  # Red
            'WARN': '\033[93m'   # Yellow
        }.get(status, '')
        
        print(f"{check['check']}: {status_color}{status}\033[0m")
    
    print("-" * 50)
    if has_failures:
        print("\n❌ Deployment validation failed. Please fix the issues above.")
        sys.exit(1)
    else:
        print("\n✅ Deployment validation passed. Ready to deploy!")