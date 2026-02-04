from fastapi.testclient import TestClient
from unittest.mock import patch
from pathlib import Path
import io

from apps.api.fastapi_app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("apps.api.fastapi_app.predictor")
def test_predict_endpoint(mock_predictor):
    mock_predictor.predict.return_value = {
        "image": "test.jpg",
        "predicted_class": 1,
        "confidence": 0.92,
        "probabilities": [0.01, 0.92, 0.02, 0.02, 0.01, 0.02],
    }

    fake_image = io.BytesIO(b"fake image bytes")

    response = client.post(
        "/predict",
        files={"file": ("test.jpg", fake_image, "image/jpeg")},
    )

    assert response.status_code == 200
    data = response.json()

    assert "predicted_class" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert data["predicted_class"] == 1