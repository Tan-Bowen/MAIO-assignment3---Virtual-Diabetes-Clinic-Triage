# app/train.py
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import joblib
import numpy as np
import os
import json

# Configuration
RANDOM_SEED = 42
MODEL_PATH = "model.joblib"
METRICS_PATH = "metrics.json"

def train_model(version="v0.1"):
    """
    Loads data, trains the model pipeline, saves the model, and logs metrics.
    """
    np.random.seed(RANDOM_SEED)

    # 1. Load Data
    Xy = load_diabetes(as_frame=True)
    X = Xy.frame.drop(columns=["target"])
    y = Xy.frame["target"]

    # 2. Split Data (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # 3. Define and Train Pipeline (v0.1: StandardScaler + LinearRegression)
    if version == "v0.1":
        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', LinearRegression())
        ])
    elif version == "v0.2":
        # Placeholder for Iteration 2 logic (e.g., Ridge or RandomForest)
        from sklearn.linear_model import Ridge
        model_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', Ridge(alpha=0.5, random_state=RANDOM_SEED))
        ])

    model_pipeline.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model_pipeline.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # 5. Save Model
    joblib.dump(model_pipeline, MODEL_PATH)

    # 6. Log Metrics
    metrics = {
        "model_version": version,
        "rmse": rmse,
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "model_type": model_pipeline.steps[-1][1].__class__.__name__
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Model trained. RMSE: {rmse:.3f}. Model saved to {MODEL_PATH}.")
    return metrics

if __name__ == "__main__":
    train_model(version=os.environ.get("MODEL_VERSION", "v0.1"))