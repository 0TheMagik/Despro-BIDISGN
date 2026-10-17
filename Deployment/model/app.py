import os
import json
import numpy as np
from typing import List, Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Hand Skeleton AI Model on Cloud Run")

# Izinkan browser HP mengakses Cloud Run tanpa blokir CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================================
# 1. MUAT MODEL AI ANDA SAAT SERVER MENYALA (COLD START)
# =====================================================================
# Sesuaikan daftar label kelas dengan urutan output model AI Anda
CLASS_LABELS = [
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
]

# UNCOMMENT SALAH SATU SESUAI FORMAT MODEL ANDA:
# --- A. Jika menggunakan TensorFlow / Keras (.h5 atau .keras) ---
import tensorflow as tf
# ai_model = tf.keras.models.load_model("model_cnn_lstm_isyarat.keras")

# --- B. Jika menggunakan Scikit-Learn / Joblib (.pkl) ---
# import joblib
# ai_model = joblib.load("model_anda.pkl")

# --- C. Jika menggunakan ONNX Runtime (.onnx) ---
# import onnxruntime as ort
# ort_session = ort.InferenceSession("model_anda.onnx")

ai_model = tf.keras.models.load_model("model_cnn_lstm_isyarat.keras")



# =====================================================================
# 2. FUNGSI INFERENSI (MENDUKUNG SINGLE FRAME & SEQUENCE VIDEO)
# =====================================================================
def run_model_inference(landmarks_data: list) -> dict:
    """
    Menerima input dari HP:
    - Single Frame: array 1D berisi 63 angka -> diubah ke shape (1, 63)
    - Video Sequence: array 2D berisi [N_frames, 63] -> diubah ke shape (1, N_frames, 63)
    """
    arr = np.array(landmarks_data, dtype=np.float32)

    # Otomatis sesuaikan dimensi batch (tambahkan dimensi batch = 1 di depan)
    if arr.ndim == 1:
        # Bentuk: (63,) -> (1, 63) untuk model Dense / MLP / RandomForest
        input_tensor = np.expand_dims(arr, axis=0)
    elif arr.ndim == 2:
        # Bentuk: (N_frames, 63) -> (1, N_frames, 63) untuk model LSTM / GRU / 1D-CNN
        input_tensor = np.expand_dims(arr, axis=0)
    else:
        raise ValueError(f"Dimensi input tidak valid: {arr.shape}")

    # --- JIKA MODEL ASLI SUDAH DIMUAT ---
    if ai_model is not None:
        # Contoh untuk Keras / TensorFlow:
        predictions = ai_model.predict(input_tensor, verbose=0)[0]
        class_idx = int(np.argmax(predictions))
        confidence = float(predictions[class_idx])
        label = CLASS_LABELS[class_idx] if class_idx < len(CLASS_LABELS) else f"Class_{class_idx}"
    else:
        # Simulasi jika Anda sedang mengetes koneksi sebelum memasukkan file model
        label = "Simulasi_Gestur_OK"
        confidence = 0.975

    return {
        "status": "success",
        "label": label,
        "confidence": round(confidence * 100, 2),
        "input_shape": list(input_tensor.shape)
    }


# =====================================================================
# 3. ENDPOINT HTTP POST & WEBSOCKET
# =====================================================================
class LandmarkPayload(BaseModel):
    # Bisa menerima [63 float] atau [[63 float], [63 float], ...]
    landmarks: Union[List[float], List[List[float]]]

@app.get("/")
def health_check():
    return {"status": "online", "service": "Cloud Run Hand AI Inference"}

@app.post("/predict")
async def predict_http(payload: LandmarkPayload):
    try:
        return run_model_inference(payload.landmarks)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.websocket("/ws/predict")
async def predict_websocket(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data_str = await websocket.receive_text()
            payload = json.loads(data_str)

            if "landmarks" not in payload:
                await websocket.send_json({"status": "error", "message": "Key 'landmarks' tidak ditemukan"})
                continue

            result = run_model_inference(payload["landmarks"])
            await websocket.send_json(result)

    except WebSocketDisconnect:
        print("Perangkat HP terputus dari WebSocket.")
    except Exception as e:
        await websocket.send_json({"status": "error", "message": str(e)})