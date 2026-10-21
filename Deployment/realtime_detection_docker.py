import asyncio
import json
import os
import time
from collections import deque

import cv2
import mediapipe as mp
import numpy as np
import websockets

# --- KONFIGURASI UTAMA (Disesuaikan dengan Kode Kedua) ---
WS_URL = os.getenv("MODEL_WS_URL", "ws://localhost:8080/ws/predict")
CAMERA_SOURCE = 0
SEQ_LEN = 90                # Diubah dari 45 ke 90 agar cocok dengan input model
PREDICTION_INTERVAL = 5     # Diubah dari 15 ke 5 agar responsif seperti kode kedua
THRESHOLD = 0.7             # Diubah dari 0.25 ke 0.7 sesuai standar kode kedua
FEATURE_SIZE = 126          # 2 tangan x 21 titik x 3 koordinat (x, y, z)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ACTION_LABELS = [
    "A", "Aku", "Anda", "Apa", "B", "Baik", "Berapa", "Bodoh", "C", "D", "Darimana",
    "Dimana", "E", "F", "G", "H", "Hallo", "I", "J", "K", "Kalian", "Kamu", "Kemana",
    "Kenapa", "Kita", "L", "M", "N", "Nunggu", "O", "P", "Pintar", "Q", "R", "S", "Saya",
    "Semua", "Siapa", "Sombong", "Sudah", "T", "Terima Kasih", "U", "V", "W", "X", "Y",
    "Z"
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Ibu jari
    (0, 5), (5, 6), (6, 7), (7, 8),           # Telunjuk
    (5, 9), (9, 10), (10, 11), (11, 12),      # Jari tengah
    (9, 13), (13, 14), (14, 15), (15, 16),    # Jari manis
    (13, 17), (17, 18), (18, 19), (19, 20),   # Kelingking
    (0, 17),                                  # Telapak
]


def get_landmarker_path():
    """Mencari file hand_landmarker.task di folder utama atau subfolder 'handlandmarker'."""
    path1 = os.path.join(BASE_DIR, "hand_landmarker.task")
    path2 = os.path.join(BASE_DIR, "handlandmarker", "hand_landmarker.task")
    if os.path.exists(path1):
        return path1
    elif os.path.exists(path2):
        return path2
    raise FileNotFoundError(
        f"File 'hand_landmarker.task' tidak ditemukan di {path1} maupun {path2}."
    )


def draw_hand_landmarks(image, results, mirror=False):
    if not (results and getattr(results, "hand_landmarks", None)):
        return

    height, width = image.shape[:2]
    for hand in results.hand_landmarks[:2]:
        if mirror:
            points = [
                (int((1.0 - landmark.x) * width), int(landmark.y * height))
                for landmark in hand
            ]
        else:
            points = [
                (int(landmark.x * width), int(landmark.y * height))
                for landmark in hand
            ]

        for start, end in HAND_CONNECTIONS:
            if start < len(points) and end < len(points):
                cv2.line(image, points[start], points[end], (0, 255, 0), 2)
        for point in points:
            cv2.circle(image, point, 4, (0, 0, 255), -1)
            cv2.circle(image, point, 6, (255, 255, 255), 1)


def extract_keypoints(results):
    if results and getattr(results, "hand_landmarks", None):
        landmarks = []
        max_hands = 2 if FEATURE_SIZE == 126 else 1
        for hand in results.hand_landmarks[:max_hands]:
            for landmark in hand:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

        keypoints = np.asarray(landmarks, dtype=np.float32)
        if keypoints.size < FEATURE_SIZE:
            keypoints = np.pad(keypoints, (0, FEATURE_SIZE - keypoints.size))
        return keypoints[:FEATURE_SIZE]

    return np.zeros(FEATURE_SIZE, dtype=np.float32)


def update_prediction(response, predictions_history):
    # Cek jika ada pesan error eksplisit dari server
    if response.get("status") == "error" or "error" in response:
        print(f"[SERVER ERROR]: {response.get('message', response.get('error'))}")
        return "-", 0.0

    label = response.get("label", response.get("action", "-"))
    raw_conf = float(response.get("confidence", 0.0))

    # Normalisasi otomatis: tangani jika server mengirim 0-100 atau 0.0-1.0
    confidence = raw_conf / 100.0 if raw_conf > 1.0 else raw_conf

    print(f"Prediksi Server: {label} ({confidence * 100:.1f}%)")

    current_prediction = label if confidence > THRESHOLD else "-"
    predictions_history.append(current_prediction)

    # Stabilisasi prediksi (ambil suara terbanyak dari 5 histori terakhir)
    if len(predictions_history) == predictions_history.maxlen:
        counts = {}
        for pred in predictions_history:
            counts[pred] = counts.get(pred, 0) + 1
        current_prediction = max(counts, key=counts.get)

    return current_prediction, confidence


async def request_prediction(websocket, sequence_data, predictions_history):
    """Fungsi asynchronous untuk mengirim data ke server tanpa membuat frame kamera macet."""
    try:
        started_at = time.perf_counter()
        payload = json.dumps({"landmarks": sequence_data})
        await websocket.send(payload)
        
        raw_response = await websocket.recv()
        response = json.loads(raw_response)
        
        latency_ms = (time.perf_counter() - started_at) * 1000
        action, conf = update_prediction(response, predictions_history)
        return action, conf, latency_ms
    except Exception as e:
        print(f"[KONEKSI ERROR]: Gagal memproses prediksi dari WebSocket: {e}")
        return "-", 0.0, 0.0


async def run_pc_tester():
    task_path = get_landmarker_path()
    base_options = mp.tasks.BaseOptions(model_asset_path=task_path)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(CAMERA_SOURCE)
    if not cap.isOpened():
        print(f"Kamera indeks {CAMERA_SOURCE} tidak tersedia, mencoba indeks 0...")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Kamera tidak dapat dibuka.")

    sequence = deque(maxlen=SEQ_LEN)
    predictions_history = deque(maxlen=5)
    predicted_action = "-"
    predicted_confidence = 0.0
    frame_counter = 0
    last_timestamp_ms = 0
    prediction_task = None

    print(f"Menghubungkan aplikasi PC ke server: {WS_URL} ...")
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            # ping_interval=None mencegah putus koneksi saat server agak lambat merespons
            async with websockets.connect(WS_URL, ping_interval=None) as websocket:
                print("Terhubung! Tekan 'q' pada jendela video untuk berhenti.")

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    height, width = frame.shape[:2]
                    
                    # PENTING: Proses frame ASLI (unflipped) ke MediaPipe sesuai data training
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb_frame,
                    )
                    
                    # Pastikan timestamp selalu naik (wajib untuk mode VIDEO di MediaPipe)
                    timestamp_ms = int(time.time() * 1000)
                    if timestamp_ms <= last_timestamp_ms:
                        timestamp_ms = last_timestamp_ms + 1
                    last_timestamp_ms = timestamp_ms

                    results = landmarker.detect_for_video(mp_image, timestamp_ms)
                    keypoints = extract_keypoints(results)

                    # Flip frame untuk tampilan layar agar natural seperti cermin
                    frame = cv2.flip(frame, 1)
                    draw_hand_landmarks(frame, results, mirror=True)
                    
                    sequence.append(keypoints)
                    frame_counter += 1

                    # Cek apakah task prediksi sebelumnya sudah selesai
                    if prediction_task is not None and prediction_task.done():
                        predicted_action, predicted_confidence, _ = prediction_task.result()
                        prediction_task = None

                    # Kirim data di background (non-blocking) seperti threading pada kode kedua
                    if (
                        len(sequence) == SEQ_LEN
                        and frame_counter % PREDICTION_INTERVAL == 0
                        and prediction_task is None
                    ):
                        seq_list = np.asarray(sequence, dtype=np.float32).tolist()
                        prediction_task = asyncio.create_task(
                            request_prediction(websocket, seq_list, predictions_history)
                        )

                    # ----- UI OVERLAY -----
                    cv2.rectangle(frame, (0, 0), (width, 50), (245, 117, 16), -1)
                    cv2.putText(
                        frame,
                        f"Deteksi: {predicted_action} ({predicted_confidence * 100:.1f}%)",
                        (10, 33),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                    # Progress bar pengisian sequence buffer
                    fill_ratio = len(sequence) / SEQ_LEN
                    bar_width = 200
                    bar_x = width - bar_width - 10
                    cv2.rectangle(
                        frame,
                        (bar_x, 12),
                        (bar_x + bar_width, 38),
                        (255, 255, 255),
                        1,
                    )
                    cv2.rectangle(
                        frame,
                        (bar_x, 12),
                        (bar_x + int(bar_width * fill_ratio), 38),
                        (0, 200, 0) if fill_ratio >= 1 else (0, 165, 255),
                        -1,
                    )
                    cv2.putText(
                        frame,
                        "[Q] Keluar",
                        (10, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (200, 200, 200),
                        1,
                        cv2.LINE_AA,
                    )
                    
                    cv2.imshow("Deteksi BISINDO Realtime", frame)

                    # Beri waktu 1ms untuk event loop asyncio dan cek tombol 'q'
                    await asyncio.sleep(0.001)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                        
    except ConnectionRefusedError:
        print(f"[ERROR]: Tidak dapat terhubung ke {WS_URL}. Pastikan server WebSocket sudah menyala!")
    finally:
        if prediction_task is not None and not prediction_task.done():
            prediction_task.cancel()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(run_pc_tester())