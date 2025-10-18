# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.api import app
from app.train import train_model
import os
import json
import numpy as np

# Use TestClient for synchronous testing of the FastAPI app
client = TestClient(app)

# Cleanup function to ensure a fresh environment
@pytest.fixture(scope="session", autouse=True)
def setup_model_for_tests():
    """Ensures the model is trained and loaded before running API tests."""
    # 1. Train the v0.1 model artifact
    print("Setting up model for API tests...")
    train_model(version="v0.1") 
    
    # 2. Set the environment variable to mock the version used by the API
    os.environ['MODEL_VERSION'] = 'v0.1'
    
    # The API will attempt to load the model on initialization
    yield
    
    # 3. Cleanup environment variable (good practice)
    del os.environ['MODEL_VERSION']


def test_health_check_endpoint():
    """Tests the GET /health endpoint for correct status and version."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_version"] == "v0.1" # Based on the setup fixture

def test_predict_endpoint_success():
    """Tests the POST /predict endpoint with valid data."""
    sample_payload = {
        "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02,
        "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001
    }
    
    response = client.post("/predict", json=sample_payload)
    assert response.status_code == 200
    data = response.json()
    
    # Check for the required response shape and type
    assert "prediction" in data
    assert isinstance(data["prediction"], float)
    
    # Check prediction result against expected value from test_model.py
    EXPECTED_PREDICTION_V01 = 154.673 
    assert np.isclose(data["prediction"], EXPECTED_PREDICTION_V01, atol=0.001)

def test_predict_endpoint_bad_input():
    """Tests POST /predict with missing data to ensure Pydantic validation works."""
    bad_payload = {
        "age": 0.02, 
        "sex": -0.044, 
        # Missing all other features
    }
    
    response = client.post("/predict", json=bad_payload)
    # Pydantic/FastAPI should return 422 Unprocessable Entity for validation errors
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
