import sys
import os
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import io
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import Streamlit app functions
from streamlit_app import process_image, get_metrics, create_performance_plots

@pytest.fixture
def mock_streamlit():
    """Mock Streamlit functions"""
    with patch('streamlit.image') as mock_image, \
         patch('streamlit.success') as mock_success, \
         patch('streamlit.error') as mock_error, \
         patch('streamlit.spinner') as mock_spinner, \
         patch('streamlit.progress') as mock_progress:
        yield {
            'image': mock_image,
            'success': mock_success,
            'error': mock_error,
            'spinner': mock_spinner,
            'progress': mock_progress
        }

@pytest.fixture
def sample_image():
    """Create a sample test image"""
    img = Image.new('RGB', (224, 224), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)
    return img_byte_arr

@pytest.fixture
def sample_metrics():
    """Create sample metrics data"""
    return {
        'performance_metrics': {
            'average_inference_time_ms': 150.5,
            'p95_inference_time_ms': 200.3,
            'predictions_per_class': {
                'Bird-drop': 100,
                'Clean': 200,
                'Dusty': 150,
                'Electrical-damage': 120,
                'Physical-Damage': 130,
                'Snow-Covered': 90
            }
        },
        'resource_usage': {
            'average_cpu_percent': 45.5,
            'average_memory_percent': 60.2,
            'average_gpu_percent': 80.1
        }
    }

def test_process_image(sample_image):
    """Test image processing function"""
    # Convert bytes to PIL Image
    image = Image.open(sample_image)
    
    # Process image
    processed = process_image(image)
    
    # Check if image was processed correctly
    assert isinstance(processed, Image.Image)
    assert processed.mode == 'RGB'
    assert max(processed.size) <= 800

@patch('requests.get')
def test_get_metrics_success(mock_get, sample_metrics):
    """Test successful metrics retrieval"""
    # Mock successful API response
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = sample_metrics
    
    # Get metrics
    metrics = get_metrics()
    
    # Verify metrics
    assert metrics == sample_metrics
    assert 'performance_metrics' in metrics
    assert 'resource_usage' in metrics

@patch('requests.get')
def test_get_metrics_failure(mock_get):
    """Test metrics retrieval failure"""
    # Mock failed API response
    mock_get.return_value.status_code = 500
    
    # Get metrics
    metrics = get_metrics()
    
    # Verify None is returned on failure
    assert metrics is None

def test_create_performance_plots(mock_streamlit, sample_metrics):
    """Test performance plots creation"""
    # Create plots
    create_performance_plots(sample_metrics)
    
    # Verify metrics were displayed
    mock_streamlit['success'].assert_called()

@pytest.mark.integration
def test_streamlit_api_integration():
    """Integration test for Streamlit and API interaction"""
    import requests
    
    # Test API health
    try:
        response = requests.get("http://localhost:5000/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")

@pytest.mark.integration
def test_prediction_workflow(sample_image):
    """Test complete prediction workflow"""
    import requests
    
    try:
        # Test single prediction
        files = {'image': ('test.jpg', sample_image, 'image/jpeg')}
        response = requests.post("http://localhost:5000/predict", files=files)
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify response structure
        assert 'prediction' in result
        assert 'confidence' in result
        assert 'inference_time_ms' in result
        assert 'top_3_predictions' in result
        
        # Verify prediction values
        assert isinstance(result['confidence'], float)
        assert 0 <= result['confidence'] <= 1
        assert len(result['top_3_predictions']) == 3
        
    except requests.exceptions.ConnectionError:
        pytest.skip("API server not running")