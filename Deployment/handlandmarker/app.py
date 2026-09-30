from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="BISINDO Handlandmarker Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = Path(os.getenv("HAND_LANDMARKER_MODEL_PATH", "/app/hand_landmarker.task"))
MAX_HANDS = 2
MAX_FEATURES = 126


def extract_keypoints(results) -> np.ndarray:
    if results and getattr(results, "hand_landmarks", None):
        landmarks_list = []
        for hand in results.hand_landmarks[:MAX_HANDS]:
            for landmark in hand:
                landmarks_list.extend([landmark.x, landmark.y, landmark.z])

        arr = np.array(landmarks_list, dtype=np.float32)
        if arr.shape[0] < MAX_FEATURES:
            arr = np.concatenate([arr, np.zeros(MAX_FEATURES - arr.shape[0], dtype=np.float32)])
        return arr[:MAX_FEATURES]

    return np.zeros(MAX_FEATURES, dtype=np.float32)


def build_landmarker():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"HandLandmarker model not found at {MODEL_PATH}")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
async def extract(file: UploadFile = File(...), max_frames: int = 90):
    if max_frames <= 0:
        raise HTTPException(status_code=400, detail="max_frames must be greater than 0")

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(await file.read())
        temp_path = Path(temp_file.name)

    frames: list[list[float]] = []
    cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        temp_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Cannot open uploaded video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    with build_landmarker() as landmarker:
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int((idx / fps) * 1000)
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            frames.append(extract_keypoints(results).tolist())
            idx += 1

            if len(frames) >= max_frames:
                break

    cap.release()
    temp_path.unlink(missing_ok=True)

    return {
        "frame_count": len(frames),
        "feature_size": MAX_FEATURES,
        "keypoints": frames,
    }
