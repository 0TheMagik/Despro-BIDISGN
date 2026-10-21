import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from torch import nn


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "model_cnn_lstm_isyarat.pt"))
FEATURES = 126
VIDEO_FRAMES = 90


class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int, **config: Any) -> None:
        super().__init__()
        filters_1 = config.get("filters_1", 64)
        filters_2 = config.get("filters_2", 64)
        lstm_1 = config.get("lstm_1", 64)
        lstm_2 = config.get("lstm_2", 64)
        lstm_3 = config.get("lstm_3", 64)
        dense_1 = config.get("dense_1", 64)
        dense_2 = config.get("dense_2", 32)
        dropout = config.get("dropout", 0.2)
        self.cnn = nn.Sequential(
            nn.Conv1d(FEATURES, filters_1, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(filters_1, filters_2, 3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
        )
        self.lstm = nn.LSTM(filters_2, lstm_1, batch_first=True)
        self.lstm_2 = nn.LSTM(lstm_1, lstm_2, batch_first=True)
        self.lstm_3 = nn.LSTM(lstm_2, lstm_3, batch_first=True)
        self.temporal_dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_3, dense_1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_1, dense_2),
            nn.ReLU(),
            nn.Linear(dense_2, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.cnn(inputs.transpose(1, 2)).transpose(1, 2)
        features, _ = self.lstm(features)
        features = self.temporal_dropout(torch.relu(features))
        features, _ = self.lstm_2(features)
        features = self.temporal_dropout(torch.relu(features))
        features, _ = self.lstm_3(features)
        features = self.temporal_dropout(torch.relu(features))
        return self.classifier(features[:, -1])


def load_model() -> tuple[nn.Module, list[str], int]:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint tidak ditemukan: {MODEL_PATH}")
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint bukan dictionary PyTorch yang valid.")
    actions = checkpoint.get("actions")
    if not isinstance(actions, list) or not all(isinstance(x, str) for x in actions):
        raise ValueError("Checkpoint tidak memiliki daftar actions yang valid.")
    if int(checkpoint.get("features", FEATURES)) != FEATURES:
        raise ValueError("Checkpoint tidak memakai 126 fitur per frame.")
    frames = int(checkpoint.get("video_frames", VIDEO_FRAMES))
    config = checkpoint.get("model_config", {})
    model = CNNLSTM(len(actions), **config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE).eval()
    return model, actions, frames


ai_model, class_labels, sequence_length = load_model()

app = FastAPI(title="Hand Skeleton AI Model on Cloud Run (PyTorch)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def run_model_inference(landmarks_data: list) -> dict:
    arr = np.asarray(landmarks_data, dtype=np.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2 or arr.shape[1] != FEATURES:
        raise ValueError(f"Input harus berbentuk [N, 126], bukan {arr.shape}.")
    if arr.shape[0] >= sequence_length:
        arr = arr[-sequence_length:]
    else:
        arr = np.pad(arr, ((sequence_length - arr.shape[0], 0), (0, 0)))

    input_tensor = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(ai_model(input_tensor), dim=1)[0]
    class_idx = int(torch.argmax(probabilities).item())
    return {
        "status": "success",
        "label": class_labels[class_idx],
        "confidence": round(float(probabilities[class_idx]) * 100, 2),
        "input_shape": list(input_tensor.shape),
    }


class LandmarkPayload(BaseModel):
    landmarks: list


@app.get("/")
@app.get("/health")
def health_check() -> dict:
    return {"status": "online", "vocabulary_size": len(class_labels)}


@app.post("/predict")
async def predict_http(payload: LandmarkPayload) -> dict:
    try:
        return run_model_inference(payload.landmarks)
    except (TypeError, ValueError, RuntimeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.websocket("/ws/predict")
async def predict_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data_str = await websocket.receive_text()
            try:
                payload = json.loads(data_str)

                if "landmarks" not in payload:
                    await websocket.send_json({"status": "error", "message": "Key 'landmarks' tidak ditemukan"})
                    continue

                result = run_model_inference(payload["landmarks"])
                await websocket.send_json(result)
            except Exception as inner_e:
                # Tangkap error inferensi di dalam loop agar koneksi WebSocket tidak putus
                await websocket.send_json({"status": "error", "message": str(inner_e)})

    except WebSocketDisconnect:
        print("Perangkat terputus dari WebSocket.")