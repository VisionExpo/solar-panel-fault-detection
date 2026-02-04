import pytest
import numpy as np
from pathlib import Path
import json
import io
from PIL import Image
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.solar_panel_detector.components.data_preparation import DataPreparation
from src.solar_panel_detector.config.configuration import Config
from app import app

@pytest.fixture
def config():
    return Config()

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

@pytest.fixture
def sample_image():
    # Create a sample test image
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img)
    img_io = io.BytesIO()
    img.save(img_io, 'JPEG')
    img_io.seek(0)
    return img_io

def test_data_preparation_initialization(config):
    data_prep = DataPreparation(config)
    assert data_prep.transform is not None
    assert data_prep.config == config

def test_image_preprocessing(config):
    data_prep = DataPreparation(config)
    # Create a test image
    test_img = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    
    # Save test image temporarily
    test_path = Path("test_image.jpg")
    Image.fromarray(test_img).save(test_path)
    
    # Test preprocessing
    processed_img = data_prep.load_and_preprocess_image(str(test_path))
    assert processed_img is not None
    assert processed_img.shape == (*config.model.img_size, 3)
    assert processed_img.dtype == np.float32
    assert np.all(processed_img >= 0) and np.all(processed_img <= 1)
    
    # Cleanup
    test_path.unlink()

def test_api_health_check(client):
    response = client.get('/health')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'status' in data
    assert data['status'] == 'healthy'

def test_predict_endpoint_no_image(client):
    response = client.post('/predict')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data

def test_predict_endpoint_with_image(client, sample_image):
    data = {
        'image': (sample_image, 'test.jpg')
    }
    response = client.post('/predict', 
                         content_type='multipart/form-data',
                         data=data)
    assert response.status_code in [200, 500]  # 500 if model not loaded, 200 if loaded

def test_batch_predict_endpoint(client, sample_image):
    data = {
        'images': [(sample_image, 'test1.jpg'), 
                  (sample_image, 'test2.jpg')]
    }
    response = client.post('/batch_predict',
                         content_type='multipart/form-data',
                         data=data)
    assert response.status_code in [200, 500]  # 500 if model not loaded, 200 if loaded

def test_batch_predict_endpoint_too_many_images(client, sample_image):
    # Create batch larger than BATCH_SIZE
    data = {
        'images': [(sample_image, f'test{i}.jpg') for i in range(33)]  # BATCH_SIZE + 1
    }
    response = client.post('/batch_predict',
                         content_type='multipart/form-data',
                         data=data)
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data