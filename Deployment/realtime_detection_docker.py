from __future__ import annotations

import os
import tempfile
import threading
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
SEQ_LEN = int(os.getenv("SEQ_LEN", "90"))
PREDICT_EVERY_N_FRAMES = int(os.getenv("PREDICT_EVERY_N_FRAMES", "5"))
threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.7"))

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
predictions_history = deque(maxlen=5)
pred_lock = threading.Lock()
prediction_thread = None
frame_counter = 0


def call_handlandmarker(video_path: Path) -> list[list[float]]:
    with video_path.open("rb") as video_file:
        response = requests.post(
            f"{HANDLANDMARKER_URL.rstrip('/')}/extract",
            files={"file": (video_path.name, video_file, "video/mp4")},
            timeout=180,
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


def predict_from_clip(video_path: Path):
    global predicted_action, predicted_confidence

    try:
        keypoints = call_handlandmarker(video_path)
        if not keypoints:
            current_pred = "-"
            confidence = 0.0
        else:
            current_pred, confidence = call_model(keypoints)
            if confidence <= threshold:
                current_pred = "-"

        with pred_lock:
            predictions_history.append(current_pred)
            predicted_confidence = confidence

            if len(predictions_history) == predictions_history.maxlen:
                counts: dict[str, int] = {}
                for prediction in predictions_history:
                    counts[prediction] = counts.get(prediction, 0) + 1
                predicted_action = max(counts, key=counts.get)
            else:
                predicted_action = current_pred
    except Exception:
        with pred_lock:
            predicted_action = "-"
            predicted_confidence = 0.0
    finally:
        video_path.unlink(missing_ok=True)


def write_clip(sequence: deque[np.ndarray], frame_size: tuple[int, int], fps: float) -> Path:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp_path = Path(tmp.name)
    tmp.close()

    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_path), codec, fps, frame_size)
    try:
        for frame in sequence:
            writer.write(frame)
    finally:
        writer.release()

    return tmp_path


def main():
    global frame_counter, prediction_thread

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        raise RuntimeError("Kamera tidak bisa dibuka")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    sequence = deque(maxlen=SEQ_LEN)

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            sequence.append(frame.copy())
            frame_counter += 1

            if len(sequence) == SEQ_LEN and frame_counter % PREDICT_EVERY_N_FRAMES == 0:
                frame_size = (frame.shape[1], frame.shape[0])
                clip_path = write_clip(sequence, frame_size, fps)

                if prediction_thread is None or not prediction_thread.is_alive():
                    prediction_thread = threading.Thread(
                        target=lambda path=clip_path: predict_from_clip(path),
                        daemon=True,
                    )
                    prediction_thread.start()
                else:
                    clip_path.unlink(missing_ok=True)

            with pred_lock:
                cur_action = predicted_action
                cur_conf = predicted_confidence

            cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (245, 117, 16), -1)
            cv2.putText(
                frame,
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
            bar_x = frame.shape[1] - bar_w - 10
            cv2.rectangle(frame, (bar_x, 12), (bar_x + bar_w, 38), (255, 255, 255), 1)
            cv2.rectangle(
                frame,
                (bar_x, 12),
                (bar_x + int(bar_w * fill_ratio), 38),
                (0, 200, 0) if fill_ratio >= 1 else (0, 165, 255),
                -1,
            )

            cv2.putText(
                frame,
                "[Q] Keluar",
                (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow("Deteksi BISINDO via Docker Endpoint", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
