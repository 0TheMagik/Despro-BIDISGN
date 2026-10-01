from __future__ import annotations

import base64
import logging
import os
import tempfile
import uuid
from pathlib import Path
from collections import deque

os.environ.setdefault("EGL_PLATFORM", "surfaceless")
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


class FrameRequest(BaseModel):
    session_id: str
    frame_data: str  # base64 encoded frame
    timestamp_ms: int
    width: int
    height: int


class SessionRequest(BaseModel):
    session_id: str


# In-memory session state for frame-by-frame streaming
class SessionState:
    def __init__(self, session_id: str, landmarker):
        self.session_id = session_id
        self.landmarker = landmarker
        self.last_keypoints: list[float] = []


sessions: dict[str, SessionState] = {}


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


@app.post("/session/start")
def session_start(request: SessionRequest):
    """Start a new frame streaming session and keep landmarker alive."""
    session_id = request.session_id
    if session_id in sessions:
        del sessions[session_id]
    landmarker = build_landmarker()
    sessions[session_id] = SessionState(session_id, landmarker)
    logger.info(f"Session {session_id} started.")
    return {"session_id": session_id, "status": "started"}


@app.post("/session/frame")
def session_frame(request: FrameRequest):
    session_id = request.session_id
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Start a session first.")

    try:
        frame_bytes = base64.b64decode(request.frame_data)
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode frame: {e}")

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = sessions[session_id].landmarker.detect_for_video(mp_image, request.timestamp_ms)
    keypoints = extract_keypoints(results).tolist()
    sessions[session_id].last_keypoints = keypoints

    return {
        "session_id": session_id,
        "timestamp_ms": request.timestamp_ms,
        "keypoints": keypoints,
    }


@app.post("/session/end")
def session_end(request: SessionRequest):
    """Close a streaming session and release resources."""
    session_id = request.session_id
    if session_id in sessions:
        sessions[session_id].landmarker.close()
        del sessions[session_id]
        logger.info(f"Session {session_id} closed.")
        return {"session_id": session_id, "status": "closed"}
    raise HTTPException(status_code=404, detail="Session not found.")
