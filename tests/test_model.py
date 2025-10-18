# tests/test_model.py
import pytest
import os
import numpy as np
import json
from app.train import train_model
from app.model import load_model, predict
from app.data_schema import DiabetesFeatures

# Define the expected RMSE for the v0.1 model (Linear Regression)
# This value is based on running train.py with RANDOM_SEED=42
EXPECTED_V01_RMSE = 54.814 

# Sample data mimicking the scaled input features of the Diabetes dataset
SAMPLE_FEATURE_DATA = {
    "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02,
    "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001
}

# Cleanup function to run after tests
@pytest.fixture(autouse=True)
def cleanup_artifacts():
    """Ensure artifacts are cleaned up before and after tests."""
    files_to_remove = ["model.joblib", "metrics.json"]
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)
    yield
    for f in files_to_remove:
        if os.path.exists(f):
            os.remove(f)

def test_train_script_produces_artifacts_and_correct_metrics():
    """Tests that train_model runs and produces expected artifacts and metrics."""
    metrics = train_model(version="v0.1")
    
    # 1. Check for files
    assert os.path.exists("model.joblib")
    assert os.path.exists("metrics.json")
    
    # 2. Check metrics content (Reproducibility check)
    assert metrics["model_version"] == "v0.1"
    assert metrics["model_type"] == "LinearRegression"
    
    # Check RMSE within a small tolerance (to ensure deterministic training)
    assert np.isclose(metrics["rmse"], EXPECTED_V01_RMSE, atol=0.001)

def test_model_loading_and_prediction():
    """Tests that the saved model can be loaded and makes a valid prediction."""
    # Ensure model is trained and saved first
    train_model(version="v0.1")
    
    # Load the model artifact
    load_model(path="model.joblib", version="v0.1")
    
    # Prepare the input data using the Pydantic schema
    data = DiabetesFeatures(**SAMPLE_FEATURE_DATA)
    
    # Make a prediction
    prediction_result = predict(data)
    
    # Check that the result is a non-negative float
    assert isinstance(prediction_result, float)
    assert prediction_result >= 0.0
    
    # Check prediction for the standard v0.1 sample data (Deterministic check)
    EXPECTED_PREDICTION_V01 = 154.673 
    assert np.isclose(prediction_result, EXPECTED_PREDICTION_V01, atol=0.001)
