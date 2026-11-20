from __future__ import annotations

import atexit
import base64
import os
import tempfile
import threading
from pathlib import Path

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

# Default 0.2 = sama dengan resep ekstraksi dataset pada docstring video_to_numpy.py.
# Pastikan nilai ini sama dengan yang dipakai saat membuat file .npy dataset.
HAND_DETECTION_CONFIDENCE = float(os.getenv("HAND_DETECTION_CONFIDENCE", "0.2"))
HAND_PRESENCE_CONFIDENCE = float(os.getenv("HAND_PRESENCE_CONFIDENCE", "0.2"))
HAND_TRACKING_CONFIDENCE = float(os.getenv("HAND_TRACKING_CONFIDENCE", "0.2"))


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
        min_hand_detection_confidence=HAND_DETECTION_CONFIDENCE,
        min_hand_presence_confidence=HAND_PRESENCE_CONFIDENCE,
        min_tracking_confidence=HAND_TRACKING_CONFIDENCE,
    )
    return HandLandmarker.create_from_options(options)


class SessionState:
    """Satu HandLandmarker VIDEO-mode per sesi realtime."""

    __slots__ = ("landmarker", "lock", "last_timestamp_ms")

    def __init__(self) -> None:
        self.landmarker = build_landmarker()
        self.lock = threading.Lock()
        self.last_timestamp_ms = 0

    def close(self) -> None:
        with self.lock:
            try:
                self.landmarker.close()
            except Exception:  # pragma: no cover - best effort
                pass


_sessions: dict[str, SessionState] = {}
_sessions_lock = threading.Lock()


def get_or_create_session(session_id: str) -> SessionState:
    with _sessions_lock:
        state = _sessions.get(session_id)
        if state is None:
            state = SessionState()
            _sessions[session_id] = state
            print(f"[handlandmarker] session dibuat: {session_id}")
        return state


def discard_session(session_id: str) -> bool:
    with _sessions_lock:
        state = _sessions.pop(session_id, None)
    if state is None:
        return False
    state.close()
    print(f"[handlandmarker] session ditutup: {session_id}")
    return True


def close_all_sessions() -> None:
    with _sessions_lock:
        states = list(_sessions.values())
        _sessions.clear()
    for state in states:
        state.close()


atexit.register(close_all_sessions)


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
    active_frames = 0
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
            keypoints = extract_keypoints(results)
            frames.append(keypoints.tolist())
            if getattr(results, "hand_landmarks", None):
                active_frames += 1
            idx += 1

            if len(frames) >= max_frames:
                break

    cap.release()
    temp_path.unlink(missing_ok=True)

    frame_count = len(frames)
    return {
        "frame_count": frame_count,
        "feature_size": MAX_FEATURES,
        "active_frames": active_frames,
        "active_ratio": (active_frames / frame_count) if frame_count else 0.0,
        "keypoints": frames,
    }


class SessionStartRequest(BaseModel):
    session_id: str


class SessionFrameRequest(BaseModel):
    session_id: str
    frame_data: str
    timestamp_ms: int
    width: int | None = None
    height: int | None = None


class SessionEndRequest(BaseModel):
    session_id: str


def decode_frame(frame_data: str) -> np.ndarray:
    if frame_data.lower().startswith("data:"):
        frame_data = frame_data.split(",", 1)[-1]
    try:
        raw = base64.b64decode(frame_data, validate=False)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="frame_data bukan base64 yang valid") from exc

    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="frame_data tidak bisa didekode menjadi gambar")
    return frame


@app.post("/session/start")
def session_start(request: SessionStartRequest):
    get_or_create_session(request.session_id)
    return {"session_id": request.session_id, "status": "started"}


@app.post("/session/frame")
def session_frame(request: SessionFrameRequest):
    frame = decode_frame(request.frame_data)
    state = get_or_create_session(request.session_id)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    with state.lock:
        # detect_for_video() butuh timestamp yang selalu naik.
        timestamp_ms = max(int(request.timestamp_ms), state.last_timestamp_ms + 1)
        state.last_timestamp_ms = timestamp_ms
        results = state.landmarker.detect_for_video(mp_image, timestamp_ms)

    keypoints = extract_keypoints(results)
    hand_count = min(len(results.hand_landmarks), MAX_HANDS) if getattr(results, "hand_landmarks", None) else 0

    return {
        "session_id": request.session_id,
        "timestamp_ms": timestamp_ms,
        "keypoints": keypoints.tolist(),
        "hand_count": hand_count,
        "active": hand_count > 0,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
    }


@app.post("/session/end")
def session_end(request: SessionEndRequest):
    closed = discard_session(request.session_id)
    return {"session_id": request.session_id, "closed": closed}

