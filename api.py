# app/api.py
from fastapi import FastAPI, HTTPException
from app.model import load_model, predict, MODEL_VERSION
from app.data_schema import DiabetesFeatures
import os

# Get model version from environment variable set during container build
MODEL_VERSION_ENV = os.environ.get("MODEL_VERSION", "v0.1")

# Load model upon startup
try:
    load_model(version=MODEL_VERSION_ENV)
except Exception as e:
    # A failure to load the model is a fatal error for the service
    print(f"FATAL: Could not load model: {e}")
    MODEL_VERSION = "LOAD_ERROR" # Reflect error in health check

app = FastAPI(
    title="Virtual Diabetes Clinic Triage ML Service",
    description="Predicts short-term disease progression risk.",
    version=MODEL_VERSION_ENV
)

@app.get("/health")
def health_check():
    """Acceptance: GET /health returns status and model version."""
    if MODEL_VERSION == "LOAD_ERROR":
        raise HTTPException(status_code=503, detail={"status": "error", "message": "Model failed to load"})
        
    return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict_risk(features: DiabetesFeatures):
    """Acceptance: POST /predict returns the continuous prediction score."""
    try:
        prediction = predict(features)
        return {"prediction": prediction}
    except RuntimeError as e:
        # Handles case where predict is called but model somehow wasn't loaded
        raise HTTPException(status_code=500, detail={"error": "Model not available"})
    except Exception as e:
        # Observability: Return JSON errors on bad input/internal errors
        # Pydantic handles most bad input, this catches others
        print(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail={"error": "Invalid input or internal processing error"})