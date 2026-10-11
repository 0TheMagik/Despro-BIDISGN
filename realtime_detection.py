import cv2
import numpy as np
import os

# Paksa eksekusi di CPU dengan menonaktifkan GPU CUDA Visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import mediapipe as mp
import torch
import torch.nn as nn
from collections import deque
import threading
import time

# Setup PyTorch CPU
device = torch.device("cpu")
print("Menjalankan model PyTorch menggunakan CPU.")

# 1. Setup Class Labels berdasarkan folder dataset (48 classes dari training)
# Di-load dari model checkpoint, default fallback berikut sebagai cadangan
DEFAULT_ACTIONS = np.array([
    'A', 'Aku', 'Anda', 'Apa', 'B', 'Baik', 'Berapa', 'Bodoh', 'C', 'D',
    'Darimana', 'Dimana', 'E', 'F', 'G', 'H', 'Hallo', 'I', 'J', 'K',
    'Kalian', 'Kamu', 'Kemana', 'Kenapa', 'Kita', 'L', 'M', 'N', 'Nunggu',
    'O', 'P', 'Pintar', 'Q', 'R', 'S', 'Saya', 'Semua', 'Siapa', 'Sombong',
    'Sudah', 'T', 'Terima Kasih', 'U', 'V', 'W', 'X', 'Y', 'Z'
])

# 2. Definisikan Arsitektur CNN + LSTM (sama dengan training)
FEATURES = 126
VIDEO_FRAMES = 90


def resample_to_90(sequence):
    """Resample sequence ke tepat 90 frame menggunakan linear interpolation."""
    if sequence.shape[0] == SEQ_LEN:
        return sequence.astype(np.float32)
    source_positions = np.linspace(0.0, 1.0, sequence.shape[0])
    target_positions = np.linspace(0.0, 1.0, SEQ_LEN)
    resampled = np.empty((SEQ_LEN, sequence.shape[1]), dtype=np.float32)
    for feature_index in range(sequence.shape[1]):
        resampled[:, feature_index] = np.interp(
            target_positions, source_positions, sequence[:, feature_index]
        )
    return resampled


def normalize_sequence(sequence):
    """Normalisasi landmark tangan - SAMA PERSIS dengan preprocessing training notebook.

    sequence shape: (num_frames, 126) — array of frames dari deque.
    Returns: (num_frames, 126) normalized.
    """
    if sequence.ndim == 1:
        sequence = sequence.reshape(1, -1)

    # Reshape: (num_frames, 126) -> (num_frames, 2, 21, 3)
    normalized = sequence.reshape(sequence.shape[0], 2, 21, 3).copy()

    for hand_index in range(2):
        hand = normalized[:, hand_index]  # shape: (num_frames, 21, 3)

        # Cek frame mana yang valid (bukan nol)
        valid_frames = np.abs(hand).sum(axis=(1, 2)) > 0
        if not np.any(valid_frames):
            continue

        # Kurangi dengan wrist (landmark 0) — per frame
        wrist = hand[:, :1, :].copy()  # shape: (num_frames, 1, 3)
        hand -= wrist

        # Scale normalization — max norm per frame, lalu expand dims agar broadcast
        scale = np.linalg.norm(hand, axis=2).max(axis=1, keepdims=True)  # shape: (num_frames, 1, 1)
        scale = np.where(scale > 1e-6, scale, 1.0)
        hand /= scale[:, :, None]

        # Set frame invalid ke nol
        normalized[~valid_frames, hand_index] = 0.0

    return normalized.reshape(sequence.shape[0], -1).astype(np.float32)


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes,
        filters_1=64,
        filters_2=64,
        lstm_1=64,
        lstm_2=64,
        lstm_3=64,
        dense_1=64,
        dense_2=32,
        dropout=0.2,
    ):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(FEATURES, filters_1, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Dropout(dropout),
            nn.Conv1d(filters_1, filters_2, kernel_size=3, padding="same"),
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

    def forward(self, inputs):
        features = self.cnn(inputs.transpose(1, 2)).transpose(1, 2)
        features, _ = self.lstm(features)
        features = self.temporal_dropout(torch.relu(features))
        features, _ = self.lstm_2(features)
        features = self.temporal_dropout(torch.relu(features))
        features, _ = self.lstm_3(features)
        features = self.temporal_dropout(torch.relu(features))
        return self.classifier(features[:, -1])


# 3. Muat Model PyTorch
MODEL_DIR = os.path.join('Prototipe kode versi pytorch', 'best_model')
MODEL_PATH = os.path.join(MODEL_DIR, 'model_cnn_lstm_isyarat.pt')

if not os.path.exists(MODEL_PATH):
    print(f"Model tidak ditemukan di {MODEL_PATH}, mencoba base_model...")
    MODEL_DIR = os.path.join('Prototipe kode versi pytorch', 'base_model')
    MODEL_PATH = os.path.join(MODEL_DIR, 'model_cnn_lstm_isyarat.pt')

checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
model_config = checkpoint.get("model_config", {})
saved_actions = checkpoint.get("actions", None)

if saved_actions is not None:
    actions = np.array(saved_actions)
    print(f"Actions dimuat dari model: {actions}")
else:
    actions = DEFAULT_ACTIONS
    print(f"Actions default digunakan: {actions}")

num_classes = len(actions)
print(f"Jumlah kelas: {num_classes}")

model = CNNLSTM(num_classes=num_classes, **model_config).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Model PyTorch berhasil dimuat!")
print("Daftar actions:", actions)

# 4. Konfigurasi Mediapipe HandLandmarker
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

hand_detection_confidence = 0.3
hand_presence_confidence = 0.3

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=os.path.join('handlandmarker', 'hand_landmarker.task')
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=hand_detection_confidence,
    min_hand_presence_confidence=hand_presence_confidence,
    min_tracking_confidence=0.3
)

# Koneksi antar landmark tangan untuk visualisasi skeleton tangan
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Ibu jari
    (0, 5), (5, 6), (6, 7), (7, 8),           # Telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),      # Jari tengah
    (9, 13), (13, 14), (14, 15), (15, 16),    # Jari manis
    (13, 17), (17, 18), (18, 19), (19, 20),   # Kelingking
    (0, 17)                                    # Telapak
]

def draw_hand_landmarks(image, results, mirror=False):
    """Gambar titik landmark + garis koneksi tangan ke frame (mengikuti gerakan tangan realtime).

    Jika mirror=True, koordinat x dibalik agar pas dengan frame display yang sudah di-flip
    secara horizontal (sementara MediaPipe tetap memproses frame asli).
    """
    if not (results and getattr(results, "hand_landmarks", None)):
        return
    h, w = image.shape[:2]
    for hand in results.hand_landmarks[:2]:
        if mirror:
            points = [(int((1.0 - lm.x) * w), int(lm.y * h)) for lm in hand]
        else:
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand]
        # Garis koneksi
        for a, b in HAND_CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(image, points[a], points[b], (0, 255, 0), 2)
        # Titik landmark
        for (x, y) in points:
            cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
            cv2.circle(image, (x, y), 6, (255, 255, 255), 1)


# 5. Fungsi Ekstraksi Keypoint (Sesuai dengan saat training/preprocessing)
last_valid_keypoints = None  # Simpan frame terakhir yang valid untuk fallback saat tracking hilang

def extract_keypoints(results):
    global last_valid_keypoints

    if results and getattr(results, "hand_landmarks", None):
        landmarks_list = []
        for hand in results.hand_landmarks[:2]:  # Max 2 hands
            for landmark in hand:
                landmarks_list.extend([landmark.x, landmark.y, landmark.z])

        arr = np.array(landmarks_list, dtype=np.float32)

        # Pad with zeros if less than 126 (e.g. only 1 hand detected)
        if arr.shape[0] < 126:
            arr = np.concatenate([arr, np.zeros(126 - arr.shape[0], dtype=np.float32)])

        # Simpan sebagai fallback (raw, belum dinormalisasi)
        last_valid_keypoints = arr[:126].copy()
        return arr[:126]

    # Jika tracking hilang, pakai frame terakhir yang valid (smooth interpolation)
    if last_valid_keypoints is not None:
        return last_valid_keypoints.copy()
    return np.zeros(126, dtype=np.float32)

# 6. Program Utama Deteksi Real-time
SEQ_LEN = 90
MIN_FRAMES = 30                          # Minimal frame untuk gestur pendek/cepat
MOVEMENT_THRESHOLD = 0.015               # Threshold gerakan untuk trigger gestur
STABILITY_THRESHOLD = 0.005              # Threshold gerakan untuk menganggap gestur selesai
STABILITY_FRAMES = 8                    # Jumlah frame stabil untuk trigger prediksi

threshold = 0.7                          # Confidence Threshold (dinaikkan ke 0.7)
predicted_action = "-"
predicted_confidence = 0.0
predictions_history = deque(maxlen=3)     # Histori untuk stabilisasi (diubah ke 3)
frame_counter = 0
prediction_thread = None
pred_lock = threading.Lock()

# --- Gesture-triggered state ---
sequence = deque(maxlen=SEQ_LEN)          # Buffer utama
last_frame = None                         # Frame sebelumnya untuk hitung movement
is_gesturing = False                      # Apakah sedang dalam gestur
stability_counter = 0                     # Counter frame stabil
gesture_started = False                   # Apakah gestur sudah dimulai

cap = cv2.VideoCapture(0)  # Ganti ke angka lain jika kamera tidak terdeteksi (misal 1, 2, 3)
if not cap.isOpened():
    print("Kamera indeks 0 tidak tersedia, mencoba indeks 1...")
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Kamera indeks 1 tidak tersedia, mencoba indeks 2...")
    cap = cv2.VideoCapture(2)

# Cek spesifikasi kamera
print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

fps_start_time = time.time()
fps_counter = 0

def predict_action(input_data):
    global predicted_action, predicted_confidence

    # Normalisasi SELEURUH sequence sekaligus — SAMA PERSIS dengan training notebook
    input_normalized = normalize_sequence(input_data[0])  # (90, 126)
    input_tensor = torch.from_numpy(input_normalized).unsqueeze(0).float().to(device)  # (1, 90, 126)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        best_idx = int(probs.argmax(dim=1).item())
        best_conf = float(probs[0, best_idx].item())

    if best_conf > threshold:
        current_pred = actions[best_idx]
    else:
        current_pred = "-"

    with pred_lock:
        predictions_history.append(current_pred)
        predicted_confidence = best_conf

        # Stabilisasi: ambil prediksi terbanyak dari 3 riwayat terakhir
        if len(predictions_history) == predictions_history.maxlen:
            counts = {}
            for p in predictions_history:
                counts[p] = counts.get(p, 0) + 1
            stable_pred = max(counts, key=counts.get)
        else:
            stable_pred = current_pred

        predicted_action = stable_pred


def put_text_with_bg(img, text, org, font_scale=0.7, color=(255, 255, 255),
                    bg=(0, 0, 0), thickness=2, pad=6):
    """Helper menggambar teks dengan background gelap agar mudah dibaca."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


with HandLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Konversi frame ASLI ke RGB untuk Mediapipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Deteksi landmark (timestamp untuk mode VIDEO)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Ekstraksi keypoints
        keypoints = extract_keypoints(results)

        # Buat frame display dengan mirror
        frame = cv2.flip(frame, 1)
        draw_hand_landmarks(frame, results, mirror=True)

        # ========== GESTURE-TRIGGERED DETECTION ==========
        # Hitung movement magnitude (L2 norm perubahan landmark)
        movement_magnitude = 0.0
        if last_frame is not None:
            diff = keypoints - last_frame
            movement_magnitude = float(np.linalg.norm(diff))

        last_frame = keypoints.copy()

        # Decision: mulai atau lanjut gestur?
        if movement_magnitude > MOVEMENT_THRESHOLD:
            is_gesturing = True
            stability_counter = 0
            if not gesture_started:
                gesture_started = True
                sequence.clear()  # Mulai fresh
        elif is_gesturing:
            if movement_magnitude < STABILITY_THRESHOLD:
                stability_counter += 1
            else:
                stability_counter = 0

        # Selalu simpan ke buffer
        sequence.append(keypoints)
        frame_counter += 1

        # Trigger prediksi: gestur mulai + cukup frame + cukup stabil
        can_predict = (
            gesture_started
            and len(sequence) >= MIN_FRAMES
            and stability_counter >= STABILITY_FRAMES
        )

        if can_predict and frame_counter % 3 == 0:  # prediksi tiap 3 frame
            # Resample jika kurang dari SEQ_LEN
            frames_array = np.array(list(sequence), dtype=np.float32)
            if frames_array.shape[0] < SEQ_LEN:
                frames_array = resample_to_90(frames_array)

            input_data = frames_array.reshape(1, SEQ_LEN, 126)
            if prediction_thread is None or not prediction_thread.is_alive():
                prediction_thread = threading.Thread(
                    target=predict_action, args=(input_data,), daemon=True
                )
                prediction_thread.start()

        # Reset state setelah gesture selesai (setelah prediksi)
        if is_gesturing and stability_counter >= STABILITY_FRAMES:
            is_gesturing = False
            gesture_started = False
            stability_counter = 0
        # =============================================

        # ----- UI Overlay -----
        # Bar atas: kata terdeteksi + confidence
        cv2.rectangle(frame, (0, 0), (w, 50), (245, 117, 16), -1)
        with pred_lock:
            cur_action = predicted_action
            cur_conf = predicted_confidence
        cv2.putText(frame, f'Deteksi: {cur_action}  ({cur_conf*100:.1f}%)',
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

        # FPS counter
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            print(f"FPS: {fps:.1f}")
            fps_counter = 0
            fps_start_time = time.time()

        # Progress bar + gesture status indicator (kanan atas)
        bar_w = 200
        bar_x = w - bar_w - 10
        bar_color = (0, 200, 0) if is_gesturing else (0, 165, 255)
        cv2.rectangle(frame, (bar_x, 12), (bar_x + bar_w, 38), (255, 255, 255), 1)
        fill_pixels = int(bar_w * min(len(sequence) / MIN_FRAMES, 1.0))
        cv2.rectangle(frame, (bar_x, 12), (bar_x + fill_pixels, 38), bar_color, -1)
        status_text = "GERAK" if is_gesturing else "DIAM"
        status_color = (0, 255, 0) if is_gesturing else (150, 150, 150)
        cv2.putText(frame, status_text, (bar_x + 5, 29),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, status_color, 1, cv2.LINE_AA)

        # Petunjuk tombol
        cv2.putText(frame, "[Q] Keluar",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow('Deteksi BISINDO Realtime', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
