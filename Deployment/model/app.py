from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from tensorflow.keras.models import load_model


app = FastAPI(title="BISINDO Vertex AI Predictor")

LABELS = np.array([
    "Kita",
    "Kamu",
    "Siapa",
    "Nunggu",
    "Saya",
    "Sudah",
    "Hallo",
    "Dimana",
    "Terima Kasih",
    "Apa",
])

MODEL_PATH = Path(os.getenv("MODEL_PATH", "/app/model/model_cnn_lstm_isyarat.keras"))
model = None
TARGET_FRAMES = 90
TARGET_FEATURES = 126


class PredictRequest(BaseModel):
    instances: list[list[list[float]]] = Field(..., description="Batch of sequences with shape [frames, features]")


@app.on_event("startup")
def startup() -> None:
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = load_model(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


def predict_one(instance: list[list[float]]):
    if model is None:
        raise RuntimeError("Model has not been loaded")

    array = np.asarray(instance, dtype=np.float32)
    if array.ndim != 2:
        raise HTTPException(status_code=400, detail="Each instance must be a 2D sequence")

    if array.shape[1] != TARGET_FEATURES:
        if array.shape[1] > TARGET_FEATURES:
            array = array[:, :TARGET_FEATURES]
        else:
            padding = np.zeros((array.shape[0], TARGET_FEATURES - array.shape[1]), dtype=np.float32)
            array = np.concatenate([array, padding], axis=1)

    if array.shape[0] > TARGET_FRAMES:
        array = array[:TARGET_FRAMES]
    elif array.shape[0] < TARGET_FRAMES:
        padding = np.zeros((TARGET_FRAMES - array.shape[0], TARGET_FEATURES), dtype=np.float32)
        array = np.concatenate([array, padding], axis=0)

    prediction = model(array[np.newaxis, ...], training=False)[0].numpy()
    best_idx = int(np.argmax(prediction))
    best_confidence = float(prediction[best_idx])

    return {
        "label": LABELS[best_idx],
        "confidence": best_confidence,
        "scores": prediction.tolist(),
    }


@app.post("/predict")
def predict(request: PredictRequest):
    predictions = [predict_one(instance) for instance in request.instances]
    return {"predictions": predictions}
