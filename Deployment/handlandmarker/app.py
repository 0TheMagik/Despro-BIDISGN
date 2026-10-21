import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="HandLandmarker + AI Model API on Cloud Run")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Inisialisasi HandLandmarker saat container pertama kali menyala (Cold Start)
base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO, # Gunakan IMAGE agar stateless di Cloud
    num_hands=1,
    min_hand_detection_confidence=0.5
)
landmarker = vision.HandLandmarker.create_from_options(options)

# 2. Muat Model AI Anda di sini (contoh menggunakan Scikit-Learn / Keras / ONNX)
# import joblib
# ai_model = joblib.load("model_gestur.pkl")

def preprocess_landmarks(hand_landmarks) -> np.ndarray:
    """Normalisasi 21 titik ke pusat WRIST (0,0,0) -> Vektor (1, 63)"""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    coords = coords - coords[0] # Translasi terhadap titik 0 (WRIST)
    flat = coords.flatten()
    max_val = np.max(np.abs(flat))
    if max_val > 0:
        flat = flat / max_val
    return np.expand_dims(flat, axis=0)

def process_jpeg_bytes(jpeg_bytes: bytes) -> dict:
    np_arr = np.frombuffer(jpeg_bytes, np.uint8)
    bgr_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if bgr_frame is None:
        return {"detected": False, "error": "Invalid JPEG"}

    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect(mp_image)
    if not result.hand_landmarks:
        return {"detected": False, "label": None}

    hand_lms = result.hand_landmarks[0]
    
    # 1. Fitur 63 dimensi yang sudah dinormalisasi untuk Model AI Anda
    features_63 = preprocess_landmarks(hand_lms)

    # --- INFERENSI MODEL AI ANDA DI SINI ---
    # pred_label = ai_model.predict(features_63)[0]
    pred_label = "Gestur_Terdeteksi"
    confidence = 0.98

    # 2. Koordinat layar (x, y) [0.0 - 1.0] untuk digambar oleh aplikasi penguji di laptop
    raw_points = [{"x": round(lm.x, 4), "y": round(lm.y, 4)} for lm in hand_lms]

    return {
        "detected": True,
        "label": pred_label,
        "confidence": confidence,
        "points": raw_points,          # Untuk visualisasi titik di layar laptop
        "landmarks_63": features_63[0].tolist() # Untuk cek data yang masuk ke AI
    }

# Endpoint 1: HTTP POST (Cocok untuk pengujian atau pengiriman per-event)
@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    jpeg_bytes = await file.read()
    return process_jpeg_bytes(jpeg_bytes)

# Endpoint 2: WebSocket Binary (Cocok untuk real-time streaming dari ESP32-CAM / Web)
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Terima byte JPEG mentah langsung dari ESP32-CAM / Browser
            jpeg_bytes = await websocket.receive_bytes()
            response = process_jpeg_bytes(jpeg_bytes)
            await websocket.send_json(response)
    except WebSocketDisconnect:
        print("Client terputus dari Cloud Run.")