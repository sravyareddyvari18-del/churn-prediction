
from fastapi import FastAPI
import joblib
import pandas as pd
import os
import json
from datetime import datetime
import time

app = FastAPI(
    title="Churn Prediction API",
    description="FastAPI backend for Telco Customer Churn Prediction",
    version="1.0"
)

MODEL_PATH = "deployment/model.pkl"
PREPROCESSOR_PATH = "deployment/preprocessor.pkl"
LOG_PATH = "logs/inference/inference_log.jsonl"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

def save_inference_log(data, prediction, probability, latency_ms, status):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "input_data": data,
        "prediction": int(prediction),
        "churn_probability": float(probability)
            if probability is not None else None,
        "model_name": "churn_prediction_model",
        "model_version": "1",
        "run_id": "1553e62089a9444f8c05bfb5865cd607",
        "latency_ms": float(latency_ms),
        "status": status
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")


@app.get("/")
def home():
    return {
        "message": "Churn Prediction API is running",
        "status": "success"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.post("/predict")
def predict(data: dict):
    start_time = time.time()
    status = "success"

    try:
        df = pd.DataFrame([data])

        X = preprocessor.transform(df)
        prediction = model.predict(X)[0]

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(X)[0][1]
        else:
            probability = None

        latency_ms = (time.time() - start_time) * 1000

        save_inference_log(
            data, prediction, probability, latency_ms, status
        )

        return {
            "prediction": int(prediction),
            "churn_probability": float(probability)
                if probability is not None else None,
            "latency_ms": round(latency_ms, 3),
            "status": status
        }

    except Exception as e:
        latency_ms = (time.time() - start_time) * 1000
        status = "failure"

        return {
            "error": str(e),
            "latency_ms": round(latency_ms, 3),
            "status": status
        }