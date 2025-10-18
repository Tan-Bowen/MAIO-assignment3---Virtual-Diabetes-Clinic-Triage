# app/model.py
import joblib
import pandas as pd
from app.data_schema import DiabetesFeatures

MODEL = None
MODEL_VERSION = "N/A" # Default

def load_model(path="model.joblib", version="v0.1"):
    """Loads the pre-trained model and sets the version."""
    global MODEL, MODEL_VERSION
    try:
        MODEL = joblib.load(path)
        MODEL_VERSION = version
        print(f"Model {MODEL_VERSION} loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {path}")
        raise

def predict(data: DiabetesFeatures) -> float:
    """Makes a prediction using the loaded model."""
    if MODEL is None:
        raise RuntimeError("Model not loaded.")
        
    # Convert Pydantic object to pandas DataFrame (1 row)
    input_data = pd.DataFrame([data.model_dump()])
    
    prediction = MODEL.predict(input_data)[0]
    
    # Prediction should be non-negative (progression index)
    return max(0.0, float(prediction))