from __future__ import annotations

import base64
import os
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")


HANDLANDMARKER_URL = os.getenv("HANDLANDMARKER_URL", "http://localhost:8001")
MODEL_URL = os.getenv("MODEL_URL", "http://localhost:8002")
SEQ_LEN = int(os.getenv("SEQ_LEN", "45"))  # Dikurangi dari 90 untuk respons lebih cepat
PREDICT_EVERY_N_FRAMES = int(os.getenv("PREDICT_EVERY_N_FRAMES", "1"))  # Prediksi setiap frame
threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))  # Diturunkan agar lebih responsif

actions = np.array([
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

predicted_action = "-"
predicted_confidence = 0.0
predictions_history = deque(maxlen=3)  # Dikurangi dari 5 agar voting lebih cepat
pred_lock = threading.Lock()
prediction_thread = None
frame_counter = 0

session_id = str(uuid.uuid4())


def put_text_with_bg(img, text, org, font_scale=0.7, color=(255, 255, 255),
                     bg=(0, 0, 0), thickness=2, pad=6):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), bg, -1)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def draw_hand_landmarks(image, keypoints, width, height, mirror=False):
    """Draw landmark points and connections on the image.

    keypoints: list of 126 floats (63 landmarks x,y,z) or list of two hands (63 each).
    mirror: flip x coordinates for display with cv2.flip.
    """
    if not keypoints or len(keypoints) < 126:
        return

    points = []
    for i in range(0, 126, 3):
        x, y, z = keypoints[i], keypoints[i + 1], keypoints[i + 2]
        if mirror:
            x = 1.0 - x
        px, py = int(x * width), int(y * height)
        points.append((px, py))

    for a, b in HAND_CONNECTIONS:
        if a < len(points) and b < len(points):
            cv2.line(image, points[a], points[b], (0, 255, 0), 2)

    for (x, y) in points:
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
        cv2.circle(image, (x, y), 6, (255, 255, 255), 1)


def call_handlandmarker_frame(frame: np.ndarray, timestamp_ms: int) -> list[float]:
    """Send a single frame to the handlandmarker session and get keypoints."""
    _, buffer = cv2.imencode(".jpg", frame)
    frame_b64 = base64.b64encode(buffer).decode("utf-8")

    response = requests.post(
        f"{HANDLANDMARKER_URL.rstrip('/')}/session/frame",
        json={
            "session_id": session_id,
            "frame_data": frame_b64,
            "timestamp_ms": timestamp_ms,
            "width": frame.shape[1],
            "height": frame.shape[0],
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("keypoints", [])


def call_model(keypoints: list[list[float]]) -> tuple[str, float]:
    response = requests.post(
        f"{MODEL_URL.rstrip('/')}/predict",
        json={"instances": [keypoints]},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    prediction = payload["predictions"][0]
    return str(prediction["label"]), float(prediction["confidence"])


def predict_action(keypoints: list[list[float]]):
    global predicted_action, predicted_confidence

    try:
        if not keypoints or len(keypoints) < SEQ_LEN:
            current_pred = "-"
            confidence = 0.0
        else:
            current_pred, confidence = call_model(keypoints)
            if confidence <= threshold:
                current_pred = "-"

        with pred_lock:
            predictions_history.append(current_pred)
            predicted_confidence = confidence

            # Cek cepat: jika kata sama muncul 2x berturut-turut, langsung tampilkan
            if len(predictions_history) >= 2:
                if predictions_history[-1] == predictions_history[-2] and predictions_history[-1] != "-":
                    predicted_action = predictions_history[-1]
                elif len(predictions_history) == predictions_history.maxlen:
                    counts: dict[str, int] = {}
                    for prediction in predictions_history:
                        counts[prediction] = counts.get(prediction, 0) + 1
                    predicted_action = max(counts, key=counts.get)
                else:
                    predicted_action = current_pred
            else:
                predicted_action = current_pred
    except Exception:
        with pred_lock:
            predicted_action = "-"
            predicted_confidence = 0.0


def start_session():
    response = requests.post(
        f"{HANDLANDMARKER_URL.rstrip('/')}/session/start",
        json={"session_id": session_id},
        timeout=30,
    )
    response.raise_for_status()
    print(f"Handlandmarker session started: {session_id}")


def end_session():
    try:
        requests.post(
            f"{HANDLANDMARKER_URL.rstrip('/')}/session/end",
            json={"session_id": session_id},
            timeout=10,
        )
        print("Handlandmarker session closed.")
    except Exception as e:
        print(f"Warning: failed to close session: {e}")


def main():
    global frame_counter, prediction_thread

    print("Memulai koneksi ke Docker services...")
    print(f"  Handlandmarker: {HANDLANDMARKER_URL}")
    print(f"  Model:          {MODEL_URL}")

    start_session()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera indeks 0 tidak tersedia, mencoba indeks 1...")
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise RuntimeError("Kamera tidak bisa dibuka")

    sequence: deque[list[float]] = deque(maxlen=SEQ_LEN)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            timestamp_ms = int(time.time() * 1000)

            try:
                keypoints = call_handlandmarker_frame(frame, timestamp_ms)
            except requests.exceptions.RequestException as e:
                print(f"[WARNING] Gagal mengirim frame ke handlandmarker: {e}")
                keypoints = []

            sequence.append(keypoints)
            frame_counter += 1

            if len(sequence) == SEQ_LEN and frame_counter % PREDICT_EVERY_N_FRAMES == 0:
                if prediction_thread is None or not prediction_thread.is_alive():
                    prediction_thread = threading.Thread(
                        target=predict_action,
                        args=(list(sequence),),
                        daemon=True,
                    )
                    prediction_thread.start()

            # Tampilkan landmark di frame (mirror untuk display natural)
            display_frame = cv2.flip(frame, 1)
            if len(sequence) > 0 and len(sequence[-1]) >= 126:
                draw_hand_landmarks(display_frame, sequence[-1], w, h, mirror=True)

            with pred_lock:
                cur_action = predicted_action
                cur_conf = predicted_confidence

            cv2.rectangle(display_frame, (0, 0), (w, 50), (245, 117, 16), -1)
            cv2.putText(
                display_frame,
                f"Deteksi: {cur_action}  ({cur_conf * 100:.1f}%)",
                (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            fill_ratio = len(sequence) / SEQ_LEN
            bar_w = 200
            bar_x = w - bar_w - 10
            cv2.rectangle(display_frame, (bar_x, 12), (bar_x + bar_w, 38), (255, 255, 255), 1)
            cv2.rectangle(
                display_frame,
                (bar_x, 12),
                (bar_x + int(bar_w * fill_ratio), 38),
                (0, 200, 0) if fill_ratio >= 1 else (0, 165, 255),
                -1,
            )

            cv2.putText(
                display_frame,
                "[Q] Keluar",
                (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Deteksi BISINDO via Docker", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        end_session()


if __name__ == "__main__":
    main()
