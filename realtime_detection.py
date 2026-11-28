import cv2
import numpy as np
import os
import textwrap
from pathlib import Path

# Paksa eksekusi di CPU dengan menonaktifkan GPU CUDA Visibility
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import mediapipe as mp
import torch
import torch.nn as nn
from collections import deque
import threading
import time

# Setup PyTorch CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "4"))))
print(f"Menjalankan model PyTorch menggunakan {device}.")

# 1. Setup Class Labels berdasarkan folder dataset (88 classes dari training terbaru)
# Di-load dari model checkpoint, default fallback berikut sebagai cadangan
DEFAULT_ACTIONS = np.array([
    'A', 'Adik', 'Ajak', 'Aku', 'Anak', 'Anda', 'Apa', 'B', 'Baik', 'Bantu',
    'Bapak', 'Bayi', 'Berapa', 'Bodoh', 'Boros', 'C', 'Cerewet', 'D',
    'Darimana', 'Dimana', 'E', 'Egois', 'F', 'G', 'Ganteng', 'H', 'Hallo',
    'Hari', 'Hati-hati', 'I', 'Ibu', 'Ingat', 'J', 'K', 'Kakak', 'Kakek',
    'Kalian', 'Kamu', 'Kapan', 'Kemana', 'Kenapa', 'Keren', 'Kereta',
    'Kerja', 'Kita', 'Kuliah', 'L', 'Lupa', 'M', 'Maaf', 'Mau', 'Mobil',
    'Motor', 'N', 'Nakal', 'Nama', 'Nunggu', 'O', 'Om', 'P', 'Perkenalkan',
    'Pintar', 'Q', 'R', 'Ramah', 'S', 'Sabar', 'Sama-sama', 'Saudara',
    'Saya', 'Sederhana', 'Semua', 'Senang', 'Siapa', 'Sombong', 'Sudah',
    'T', 'Terima Kasih', 'Tidak', 'Tidak Mau', 'Tolong', 'Tuli', 'U', 'V',
    'W', 'X', 'Y', 'Z'
])

# 2. Definisikan Arsitektur CNN + LSTM (sama dengan training)
FEATURES = 126
VIDEO_FRAMES = 90


def normalize_sequence(sequence):
    """Normalisasi landmark tangan - SAM dengan preprocessing training notebook."""
    # sequence shape: (126,) per frame, kita treat sebagai single frame
    # Normalisasi per-frame: reshape ke (2, 21, 3) [2 tangan, 21 landmarks, 3 koordinat]
    normalized = sequence.reshape(2, 21, 3).copy()

    for hand_index in range(2):
        hand = normalized[hand_index]
        # Cek apakah landmark valid (tidak nol)
        valid = np.abs(hand).sum() > 0
        if not valid:
            continue

        # Kurangi dengan wrist (landmark 0)
        wrist = hand[0:1, :].copy()
        hand -= wrist

        # Scale normalization
        scale = np.linalg.norm(hand, axis=1).max()
        if scale > 1e-6:
            hand /= scale

        normalized[hand_index] = hand

    return normalized.reshape(-1)


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
ROOT_DIR = Path(__file__).resolve().parent
MODEL_VARIANT = os.getenv("REALTIME_MODEL", "best").strip().lower()
MODEL_DIRECTORIES = {
    "best": "best_model",
    "base": "base_model",
}
if MODEL_VARIANT not in MODEL_DIRECTORIES:
    raise ValueError("REALTIME_MODEL harus bernilai 'best' atau 'base'.")

MODEL_PATH = (
    ROOT_DIR
    / 'Prototipe kode versi pytorch'
    / MODEL_DIRECTORIES[MODEL_VARIANT]
    / 'model_cnn_lstm_isyarat.pt'
)
if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Checkpoint {MODEL_VARIANT}_model tidak ditemukan: {MODEL_PATH}"
    )

checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
print(f"Checkpoint realtime: {MODEL_VARIANT} ({MODEL_PATH})")
model_config = checkpoint.get("model_config", {})
saved_actions = checkpoint.get("actions", None)

if saved_actions is not None:
    actions = np.array(saved_actions)
    print(f"Actions dimuat dari model: {actions}")
else:
    actions = DEFAULT_ACTIONS
    print(f"Actions default digunakan: {actions}")

num_classes = len(actions)
VIDEO_FRAMES = int(checkpoint.get("video_frames", VIDEO_FRAMES))
model_features = int(checkpoint.get("features", FEATURES))
if model_features != FEATURES:
    raise RuntimeError(
        f"Checkpoint best_model membutuhkan {model_features} fitur; script menghasilkan {FEATURES}."
    )
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

hand_detection_confidence = float(os.getenv("HAND_DETECTION_CONFIDENCE", "0.35"))
hand_presence_confidence = float(os.getenv("HAND_PRESENCE_CONFIDENCE", "0.35"))
hand_tracking_confidence = float(os.getenv("HAND_TRACKING_CONFIDENCE", "0.35"))
for name, value in (
    ("HAND_DETECTION_CONFIDENCE", hand_detection_confidence),
    ("HAND_PRESENCE_CONFIDENCE", hand_presence_confidence),
    ("HAND_TRACKING_CONFIDENCE", hand_tracking_confidence),
):
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} harus antara 0 dan 1.")

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=str(ROOT_DIR / 'handlandmarker' / 'hand_landmarker.task')
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=hand_detection_confidence,
    min_hand_presence_confidence=hand_presence_confidence,
    min_tracking_confidence=hand_tracking_confidence
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
def extract_keypoints(results):
    """Ambil keypoint dengan urutan yang kompatibel dengan data training.

    Default ``legacy`` mempertahankan urutan MediaPipe seperti pipeline lama.
    Mode ``handedness`` menempatkan kiri lalu kanan dan berguna bila urutan
    MediaPipe sering bertukar saat tangan overlap.
    """
    hands = list(results.hand_landmarks[:2]) if results and getattr(results, "hand_landmarks", None) else []
    if os.getenv("REALTIME_HAND_ORDER", "legacy").strip().lower() == "handedness" and hands:
        ordered_hands = [None, None]  # slot 0 = left, slot 1 = right
        handedness_list = getattr(results, "handedness", []) or []
        for index, hand in enumerate(hands):
            label = ""
            if index < len(handedness_list) and handedness_list[index]:
                label = str(handedness_list[index][0].category_name).lower()
            target = 0 if label == "left" else 1 if label == "right" else None
            if target is not None and ordered_hands[target] is None:
                ordered_hands[target] = hand
            elif ordered_hands[0] is None:
                ordered_hands[0] = hand
            elif ordered_hands[1] is None:
                ordered_hands[1] = hand
        hands = [hand for hand in ordered_hands if hand is not None]

    values = []
    for hand in hands:
        for landmark in hand:
            values.extend([landmark.x, landmark.y, landmark.z])
    values.extend([0.0] * (126 - len(values)))
    return normalize_sequence(np.asarray(values[:126], dtype=np.float32))


def has_hands(results):
    return bool(results and getattr(results, "hand_landmarks", None))

# 6. Program Utama Deteksi Real-time
SEQ_LEN = VIDEO_FRAMES
sequence = deque(maxlen=SEQ_LEN)        # Buffer untuk menyimpan satu sequence model
threshold = float(os.getenv("REALTIME_CONFIDENCE_THRESHOLD", "0.35"))
if not 0.0 <= threshold <= 1.0:
    raise ValueError("REALTIME_CONFIDENCE_THRESHOLD harus antara 0 dan 1.")
predicted_action = "-"
predicted_confidence = 0.0
predictions_history = deque(maxlen=3)    # Histori untuk stabilisasi prediksi
recognized_text = ""
last_committed_action = None
empty_hand_frames = 0
frame_counter = 0
prediction_thread = None
pred_lock = threading.Lock()
last_valid_keypoints = None
missing_keypoint_frames = 0
MAX_KEYPOINT_HOLD_FRAMES = int(os.getenv("MAX_KEYPOINT_HOLD_FRAMES", "3"))

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Kamera indeks 0 tidak tersedia, mencoba indeks 1...")
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Kamera indeks 1 tidak tersedia, mencoba indeks 2...")
    cap = cv2.VideoCapture(2)
if not cap.isOpened():
    raise RuntimeError("Kamera tidak dapat dibuka. Periksa kamera atau ubah indeks kamera.")

# Cek spesifikasi kamera
print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

fps_start_time = time.time()
fps_counter = 0

def predict_action(input_data):
    global predicted_action, predicted_confidence

    active_ratio = np.mean(np.abs(input_data[0]).sum(axis=1) > 1e-6)
    if active_ratio < 0.6:
        with pred_lock:
            predictions_history.clear()
            predicted_action = "-"
            predicted_confidence = 0.0
        return

    input_tensor = torch.from_numpy(input_data).float().to(device)
    with torch.inference_mode():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1)
        best_idx = int(probs.argmax(dim=1).item())
        best_conf = float(probs[0, best_idx].item())

    current_pred = actions[best_idx] if best_conf > threshold else "-"

    with pred_lock:
        predictions_history.append((str(current_pred), best_conf))
        counts = {}
        for label, _ in predictions_history:
            if label == "-":
                continue
            counts[label] = counts.get(label, 0) + 1
        if not counts:
            predicted_action = "-"
            predicted_confidence = best_conf
            return

        stable_pred = max(counts, key=counts.get)
        if counts[stable_pred] < 2:
            predicted_action = "-"
            predicted_confidence = best_conf
            return

        stable_scores = [score for label, score in predictions_history if label == stable_pred]
        predicted_action = stable_pred
        predicted_confidence = sum(stable_scores) / len(stable_scores)


def append_label_to_text(label):
    global recognized_text

    label = str(label).strip()
    if not label:
        return
    if len(label) == 1 and label.isalpha():
        recognized_text += label.upper()
    else:
        if recognized_text and not recognized_text.endswith(" "):
            recognized_text += " "
        recognized_text += label + " "


def draw_text_panel(image, text):
    h, w = image.shape[:2]
    panel_top = max(60, h - 105)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, panel_top), (w, h - 24), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, image, 0.35, 0, image)
    cv2.putText(
        image, "Teks:", (10, panel_top + 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA
    )

    display_text = text.rstrip() or "(belum ada teks)"
    line_width = max(16, int((w - 20) / 11))
    lines = textwrap.wrap(display_text, width=line_width) or [display_text]
    for index, line in enumerate(lines[-2:]):
        cv2.putText(
            image, line, (10, panel_top + 49 + index * 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA
        )


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

        # PENTING: Dataset training direkam TANPA flip horizontal
        # (lihat dataset_collection_V2.py — out.write(frame) menulis frame asli).
        # Jadi MediaPipe HARUS memproses frame asli (unflipped) agar koordinat
        # cocok dengan distribusi data training. Flip hanya untuk display.
        h, w = frame.shape[:2]

        # Konversi frame ASLI ke RGB untuk Mediapipe (sesuai pipeline training)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Deteksi landmark (gunakan timestamp untuk mode VIDEO)
        timestamp_ms = int(time.time() * 1000)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Ekstraksi keypoints dari frame asli untuk model.
        # Pertahankan frame valid terakhir sebentar saat tracking putus,
        # karena overlap jari sering membuat MediaPipe kehilangan tangan 1-3 frame.
        keypoints = extract_keypoints(results)
        hand_present = has_hands(results)
        if hand_present:
            last_valid_keypoints = keypoints.copy()
            missing_keypoint_frames = 0
            empty_hand_frames = 0
        else:
            missing_keypoint_frames += 1
            empty_hand_frames += 1
            if last_valid_keypoints is not None and missing_keypoint_frames <= MAX_KEYPOINT_HOLD_FRAMES:
                keypoints = last_valid_keypoints.copy()
            if empty_hand_frames == 8:
                sequence.clear()
                last_valid_keypoints = None
                missing_keypoint_frames = 0
                with pred_lock:
                    predictions_history.clear()
                    predicted_action = "-"
                    predicted_confidence = 0.0
                last_committed_action = None

        # Buat frame display dengan mirror agar terasa natural seperti cermin
        frame = cv2.flip(frame, 1)

        # Gambar landmark di frame display (mirror x supaya tetap menempel di tangan)
        draw_hand_landmarks(frame, results, mirror=True)
        sequence.append(keypoints)
        frame_counter += 1

        # Mulai Prediksi jika sequence sudah terkumpul penuh
        # Optimasi: prediksi setiap 5 frame, dijalankan di thread agar UI tetap lancar
        if len(sequence) == SEQ_LEN and frame_counter % 5 == 0:
            input_data = np.expand_dims(list(sequence), axis=0)
            if prediction_thread is None or not prediction_thread.is_alive():
                prediction_thread = threading.Thread(
                    target=predict_action, args=(input_data,), daemon=True
                )
                prediction_thread.start()

        # ----- UI Overlay -----
        # Bar atas: kata terdeteksi + confidence
        cv2.rectangle(frame, (0, 0), (w, 50), (245, 117, 16), -1)
        with pred_lock:
            cur_action = predicted_action
            cur_conf = predicted_confidence
        cv2.putText(frame, f'Deteksi: {cur_action}  ({cur_conf*100:.1f}%)',
                    (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        if (
            hand_present
            and cur_action != "-"
            and cur_conf > threshold
            and cur_action != last_committed_action
        ):
            append_label_to_text(cur_action)
            last_committed_action = cur_action

        # FPS counter
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            print(f"FPS: {fps:.1f}")
            fps_counter = 0
            fps_start_time = time.time()

        # Progress bar pengisian buffer (kanan atas)
        fill_ratio = len(sequence) / SEQ_LEN
        bar_w = 200
        bar_x = w - bar_w - 10
        cv2.rectangle(frame, (bar_x, 12), (bar_x + bar_w, 38), (255, 255, 255), 1)
        cv2.rectangle(frame, (bar_x, 12),
                      (bar_x + int(bar_w * fill_ratio), 38),
                      (0, 200, 0) if fill_ratio >= 1 else (0, 165, 255), -1)

        draw_text_panel(frame, recognized_text)
        cv2.putText(
            frame,
            "[Q] Keluar  [SPACE] Spasi  [BACKSPACE] Hapus  [C] Reset",
            (10, h - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.imshow('Deteksi BISINDO Realtime', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == 32:
            if recognized_text and not recognized_text.endswith(" "):
                recognized_text += " "
        elif key in (8, 127):
            recognized_text = recognized_text.rstrip()[:-1]
        elif key in (ord('c'), ord('C')):
            recognized_text = ""

cap.release()
cv2.destroyAllWindows()
