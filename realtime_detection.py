import cv2
import numpy as np
import os

# Paksa eksekusi di CPU dengan menonaktifkan GPU CUDA Visbility
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import deque
import threading
import time

# Setup TensorFlow CPU
print("Menjalankan model menggunakan CPU.")

# 1. Setup Class Labels berdasarkan folder dataset
# PENTING: Urutan ini HARUS sama persis dengan urutan label saat training di notebook Linux!
# Jika menggunakan os.listdir() di Windows urutannya akan abjad (salah).
actions = np.array(['Kita', 'Kamu', 'Siapa', 'Nunggu', 'Saya', 'Sudah', 'Hallo', 'Dimana', 'Terima Kasih', 'Apa'])

# 2. Muat Model Terbaik (Pastikan path sesuai)
MODEL_PATH = os.path.join('best_model', 'model_cnn_lstm_isyarat.keras')
if not os.path.exists(MODEL_PATH):
    print(f"Model tidak ditemukan di {MODEL_PATH}. Mencari di base_model...")
    MODEL_PATH = os.path.join('base_model', 'model_cnn_lstm_isyarat.keras')

model = load_model(MODEL_PATH)
print("Model berhasil dimuat!")
print("Daftar actions:", actions)

# 3. Konfigurasi Mediapipe HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_detection_confidence = 0.5
hand_presence_confidence = 0.5

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join('handlandmarker', 'hand_landmarker.task')
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=hand_detection_confidence,
    min_hand_presence_confidence=hand_presence_confidence,
    min_tracking_confidence=0.5
)

# 4. Fungsi Ekstraksi Keypoint (Sesuai dengan saat training/preprocessing)
def extract_keypoints(results):
    if results and getattr(results, "hand_landmarks", None):
        landmarks_list = []
        for hand in results.hand_landmarks[:2]: # Max 2 hands
            for landmark in hand:
                landmarks_list.extend([landmark.x, landmark.y, landmark.z])

        arr = np.array(landmarks_list, dtype=np.float32)
        # Pad with zeros if less than 126 (e.g. only 1 hand detected)
        if arr.shape[0] < 126:
            arr = np.concatenate([arr, np.zeros(126 - arr.shape[0], dtype=np.float32)])
        return arr[:126]
    return np.zeros(126, dtype=np.float32)

# 5. Program Utama Deteksi Real-time
sequence = deque(maxlen=90) # Buffer untuk menyimpan 90 frame terakhir
threshold = 0.7 # Confidence Threshold
predicted_action = "-"
predictions_history = deque(maxlen=5) # Histori untuk stabilisasi prediksi
frame_counter = 0 # Konter frame untuk menghemat komputasi prediksi
prediction_thread = None

cap = cv2.VideoCapture(1) # Ganti ke angka lain jika kamera tidak terdeteksi (misal 1 untuk external cam)

def predict_action(input_data):
    global predicted_action
    
    res = model(input_data, training=False)[0].numpy()
    best_idx = np.argmax(res)
    
    if res[best_idx] > threshold:
        current_pred = actions[best_idx]
    else:
        current_pred = "-"
        
    predictions_history.append(current_pred)
    
    # Menghindari teks lompat-lompat/flicker (ambil prediksi terbanyak dari 5 riwayat terakhir)
    if len(predictions_history) == 5:
        counts = {}
        for p in predictions_history:
            counts[p] = counts.get(p, 0) + 1
        predicted_action = max(counts, key=counts.get)
    else:
        predicted_action = current_pred

with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip frame secara horizontal untuk tampilan seperti cermin
        frame = cv2.flip(frame, 1)
        
        # Konversi ke RGB untuk Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Deteksi landmark (gunakan timestamp untuk mode VIDEO)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Gambar prediksi tangan di layar (visualisasi opsional)
        # Di sini kita tidak gambar landmark lengkap agar lebih efisien, 
        # tapi ekstraksi fiturnya yang terpenting
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        frame_counter += 1
        
        # Mulai Prediksi jika sequence sudah terkumpul penuh sesuai frame model (90 frames)
        # OPTIMASI 1: Lakukan prediksi setiap 5 frame (skip-frame) agar tidak membebani komputasi
        if len(sequence) == 90 and frame_counter % 5 == 0:
            input_data = np.expand_dims(list(sequence), axis=0)
            
            # Jalankan prediksi di thread terpisah agar frame tidak berhenti saat komputasi model berjalan
            if prediction_thread is None or not prediction_thread.is_alive():
                prediction_thread = threading.Thread(target=predict_action, args=(input_data,))
                prediction_thread.start()
        
        # Tampilkan teks hasil
        cv2.rectangle(frame, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(frame, f'Deteksi: {predicted_action}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        cv2.imshow('Deteksi BISINDO Realtime', frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
