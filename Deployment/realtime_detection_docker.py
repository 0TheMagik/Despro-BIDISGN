"""Realtime BISINDO sign-to-text client.

Keep the input contract aligned with Deployment/model/app.py:
- raw MediaPipe landmarks, 2 x 21 x (x, y, z) = 126 float features/frame
- no mirroring or client-side normalization before POST /predict
- service owns sequence resampling and wrist-relative normalization

Run from repository root:
    python Deployment/realtime_detection_docker.py
    python Deployment/realtime_detection_docker.py --backend docker
    python Deployment/realtime_detection_docker.py --video clip.mp4 --show
"""
from __future__ import annotations

import argparse
import base64
import importlib.util
import os
import textwrap
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # dotenv is optional
    load_dotenv = None

SCRIPT_DIR = Path(__file__).resolve().parent
if load_dotenv is not None:
    load_dotenv(dotenv_path=SCRIPT_DIR / ".env")

FEATURES = 126
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        value = float(os.getenv(name, str(default)))
        return value if np.isfinite(value) else default
    except (TypeError, ValueError):
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_landmark_model_path() -> Path | None:
    candidates: list[Path] = []
    explicit = os.getenv("HAND_LANDMARKER_MODEL_PATH")
    if explicit:
        candidates.append(Path(explicit).expanduser())
    candidates.extend((
        SCRIPT_DIR / "handlandmarker" / "hand_landmarker.task",
        SCRIPT_DIR.parent / "handlandmarker" / "hand_landmarker.task",
        Path("handlandmarker/hand_landmarker.task"),
    ))
    return next((path for path in candidates if path.is_file()), None)


@dataclass(frozen=True)
class Config:
    handlandmarker_url: str
    model_url: str
    landmark_backend: str
    landmark_model_path: Path | None
    window_frames: int
    min_window_frames: int
    stride_frames: int
    prediction_threshold: float
    top1_margin: float
    vote_window: int
    stable_predictions: int
    release_after_empty_frames: int
    min_active_ratio: float
    movement_threshold: float
    gesture_motion_threshold: float
    stability_frames: int
    gesture_timeout: float
    min_hold_seconds: float
    commit_cooldown: float
    require_stable_motion: bool
    require_gesture: bool
    capture_width: int
    capture_height: int
    capture_fps: int
    camera_index: int
    jpeg_quality: int
    text_limit: int
    show_status: bool
    debug: bool


def load_config(args: argparse.Namespace) -> Config:
    window_frames = max(8, min(env_int("WINDOW_FRAMES", env_int("SEQ_LEN", 36)), 180))
    min_window_frames = max(4, min(env_int("MIN_WINDOW_FRAMES", min(24, window_frames)), window_frames))
    landmark_path = resolve_landmark_model_path()
    backend = (args.backend or os.getenv("LANDMARK_BACKEND", "auto")).strip().lower()
    if backend not in {"auto", "local", "docker"}:
        backend = "auto"
    if backend == "auto":
        try:
            backend = "local" if importlib.util.find_spec("mediapipe") and landmark_path else "docker"
        except (ImportError, ValueError):
            backend = "docker"

    return Config(
        handlandmarker_url=os.getenv("HANDLANDMARKER_URL", "http://localhost:8001").rstrip("/"),
        model_url=os.getenv("MODEL_URL", "http://localhost:8002").rstrip("/"),
        landmark_backend=backend,
        landmark_model_path=landmark_path,
        window_frames=window_frames,
        min_window_frames=min_window_frames,
        stride_frames=max(1, env_int("STRIDE_FRAMES", env_int("PREDICT_EVERY_N_FRAMES", 3))),
        prediction_threshold=float(np.clip(env_float("PREDICTION_THRESHOLD", 0.85), 0.0, 0.999)),
        top1_margin=float(np.clip(env_float("TOP1_MARGIN", 0.12), 0.0, 0.95)),
        vote_window=max(1, env_int("VOTE_WINDOW", 3)),
        stable_predictions=max(1, env_int("STABLE_PREDICTIONS", 3)),
        release_after_empty_frames=max(1, env_int("RELEASE_AFTER_EMPTY_FRAMES", 8)),
        min_active_ratio=float(np.clip(env_float("MIN_ACTIVE_RATIO", 0.6), 0.0, 0.95)),
        movement_threshold=max(0.0005, env_float("MOVEMENT_THRESHOLD", 0.012)),
        gesture_motion_threshold=max(0.0005, env_float("GESTURE_MOTION_THRESHOLD", 0.02)),
        stability_frames=max(1, env_int("STABILITY_FRAMES", 3)),
        gesture_timeout=max(0.5, env_float("GESTURE_TIMEOUT_SECONDS", 4.0)),
        min_hold_seconds=max(0.0, env_float("MIN_HOLD_SECONDS", 0.1)),
        commit_cooldown=max(0.0, env_float("COMMIT_COOLDOWN_SECONDS", 0.3)),
        # Motion-based suppression is opt-in: many signs are static and otherwise
        # never produce a prediction. Consensus/confidence still suppress noise.
        require_stable_motion=env_bool("REQUIRE_STABLE_MOTION", False),
        require_gesture=env_bool("REQUIRE_GESTURE", False),
        capture_width=max(160, env_int("CAPTURE_WIDTH", 640)),
        capture_height=max(120, env_int("CAPTURE_HEIGHT", 480)),
        capture_fps=max(5, env_int("CAPTURE_FPS", 30)),
        camera_index=args.camera if args.camera is not None else env_int("CAMERA_INDEX", 0),
        jpeg_quality=int(np.clip(env_int("JPEG_QUALITY", 80), 40, 95)),
        text_limit=max(40, env_int("TEXT_LIMIT", 160)),
        show_status=env_bool("SHOW_STATUS", False),
        debug=bool(args.debug or env_bool("DEBUG", False)),
    )


def print_banner(cfg: Config) -> None:
    print("=" * 64)
    print(" BISINDO realtime -> teks | 97 kelas: A-Z + kata")
    print("=" * 64)
    print(f" Model service : {cfg.model_url}")
    print(f" Landmark      : {cfg.landmark_backend} ({cfg.landmark_model_path or 'service Docker'})")
    print(f" Window        : {cfg.min_window_frames}-{cfg.window_frames} frame, stride {cfg.stride_frames}")
    print(f" Filter        : conf>={cfg.prediction_threshold:.2f}, margin>={cfg.top1_margin:.2f}, "
          f"konsensus={max(cfg.vote_window, cfg.stable_predictions)}")
    print("=" * 64)


def make_http_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class TimestampSource:
    """Monotonic millisecond timestamps required by MediaPipe VIDEO mode."""
    def __init__(self) -> None:
        self._last = -1

    def next(self) -> int:
        value = int(time.perf_counter() * 1000)
        self._last = max(value, self._last + 1)
        return self._last


def extract_keypoints(results: Any) -> tuple[np.ndarray, int]:
    hands = getattr(results, "hand_landmarks", None) or []
    output = np.zeros(FEATURES, dtype=np.float32)
    count = min(len(hands), 2)
    if count == 0:
        return output, 0
    values: list[float] = []
    for hand in hands[:2]:
        for landmark in hand:
            values.extend((landmark.x, landmark.y, landmark.z))
    values_array = np.asarray(values, dtype=np.float32)
    if values_array.size > FEATURES:
        values_array = values_array[:FEATURES]
    output[:values_array.size] = values_array
    if not np.isfinite(output).all():
        return np.zeros(FEATURES, dtype=np.float32), 0
    return output, count


class LandmarkSource:
    backend = "?"

    def process(self, frame: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray, int, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalLandmarks(LandmarkSource):
    """MediaPipe in-process; no JPEG encoding or per-frame HTTP call."""
    backend = "local"

    def __init__(self, model_path: Path | None) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe belum terpasang. Install mediapipe atau gunakan --backend docker.") from exc
        if model_path is None or not model_path.is_file():
            raise RuntimeError("File hand_landmarker.task tidak ditemukan")
        self._mp = mp
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=env_float("HAND_DETECTION_CONFIDENCE", 0.2),
            min_hand_presence_confidence=env_float("HAND_PRESENCE_CONFIDENCE", 0.2),
            min_tracking_confidence=env_float("HAND_TRACKING_CONFIDENCE", 0.2),
        )
        self._landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        self._last_error_at = 0.0

    def process(self, frame: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray, int, Any]:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(image, int(timestamp_ms))
            keypoints, count = extract_keypoints(result)
            return keypoints, count, result
        except Exception as exc:
            now = time.perf_counter()
            if now - self._last_error_at > 2.0:
                print(f"[WARNING] MediaPipe gagal memproses frame: {exc}")
                self._last_error_at = now
            return np.zeros(FEATURES, dtype=np.float32), 0, None

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


class DockerLandmarks(LandmarkSource):
    """Landmarks via the existing handlandmarker /session API."""
    backend = "docker"

    def __init__(self, base_url: str, session_id: str, jpeg_quality: int, timeout: float = 15.0) -> None:
        self.url = base_url.rstrip("/")
        self.session_id = session_id
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self.session = make_http_session()
        self._started = False
        self._last_error_at = 0.0

    def start(self) -> None:
        try:
            response = self.session.post(f"{self.url}/session/start", json={"session_id": self.session_id}, timeout=10)
            response.raise_for_status()
            self._started = True
        except Exception as exc:
            raise RuntimeError(f"Service handlandmarker tidak dapat dihubungi di {self.url}: {exc}\n"
                               "Jalankan: docker compose -f Deployment/docker-compose.yml up --build") from exc

    def process(self, frame: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray, int, Any]:
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return np.zeros(FEATURES, dtype=np.float32), 0, None
        payload = {
            "session_id": self.session_id,
            "frame_data": base64.b64encode(buffer).decode("ascii"),
            "timestamp_ms": int(timestamp_ms),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }
        try:
            response = self.session.post(f"{self.url}/session/frame", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            points = np.asarray(data.get("keypoints", []), dtype=np.float32).reshape(-1)
            if points.size != FEATURES:
                raise ValueError(f"Expected {FEATURES} features, received {points.size}")
            if not np.isfinite(points).all():
                raise ValueError("Handlandmarker returned NaN/Inf keypoints")
            count = int(data.get("hand_count", int(np.abs(points).sum() > 1e-6)))
            return points, max(0, min(count, 2)), None
        except Exception as exc:
            now = time.perf_counter()
            if now - self._last_error_at > 2.0:
                print(f"[WARNING] Gagal memproses frame di handlandmarker: {exc}")
                self._last_error_at = now
            return np.zeros(FEATURES, dtype=np.float32), 0, None

    def close(self) -> None:
        try:
            if self._started:
                self.session.post(f"{self.url}/session/end", json={"session_id": self.session_id}, timeout=5)
        except Exception:
            pass
        finally:
            self.session.close()


def create_landmarker(cfg: Config, session_id: str) -> LandmarkSource:
    if cfg.landmark_backend in {"auto", "local"}:
        try:
            source = LocalLandmarks(cfg.landmark_model_path)
            print(f"[LANDMARK] MediaPipe lokal: {cfg.landmark_model_path}")
            return source
        except Exception as exc:
            if cfg.landmark_backend == "local":
                raise RuntimeError(f"Backend landmark lokal gagal: {exc}") from exc
            print(f"[WARNING] Backend lokal tidak tersedia ({exc}); fallback ke service.")
    source = DockerLandmarks(cfg.handlandmarker_url, session_id, cfg.jpeg_quality)
    source.start()
    print(f"[LANDMARK] Service Docker: {cfg.handlandmarker_url}")
    return source


class ModelClient:
    def __init__(self, base_url: str) -> None:
        self.url = base_url.rstrip("/")
        self.session = make_http_session()
        self.info: dict[str, Any] = {}
        self.labels: list[str] = []
        self.seq_len = 90
        self.features = FEATURES

    def connect(self, attempts: int = 8, delay: float = 1.0) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = self.session.get(f"{self.url}/health", timeout=5)
                response.raise_for_status()
                self.info = response.json()
                if self.info.get("status") not in {None, "ok"}:
                    raise RuntimeError(f"Model belum siap: {self.info.get('status')}")
                labels_response = self.session.get(f"{self.url}/labels", timeout=5)
                labels_response.raise_for_status()
                metadata = labels_response.json()
                self.labels = [str(label) for label in metadata.get("labels", [])]
                self.seq_len = int(metadata.get("seq_len", self.seq_len))
                self.features = int(metadata.get("features", self.features))
                if self.features != FEATURES:
                    raise RuntimeError(f"Model mengharapkan {self.features} fitur, client menghasilkan {FEATURES}")
                if self.seq_len <= 0:
                    raise RuntimeError(f"Model mengembalikan seq_len tidak valid: {self.seq_len}")
                self.info.update(metadata)
                return self.info
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    print(f"[INFO] Menunggu model service siap ({attempt}/{attempts}) ...")
                    time.sleep(delay)
        raise RuntimeError(f"Model service tidak siap di {self.url}. Detail: {last_error}")

    def predict(self, sequence: np.ndarray) -> dict[str, Any]:
        values = np.asarray(sequence, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] != self.features:
            raise ValueError(f"Sequence harus [frames, {self.features}], diterima {values.shape}")
        if not np.isfinite(values).all():
            raise ValueError("Sequence berisi NaN atau Inf")
        response = self.session.post(f"{self.url}/predict", json={"instances": [values.tolist()]}, timeout=20)
        response.raise_for_status()
        predictions = response.json().get("predictions") or []
        if not predictions:
            raise RuntimeError("Response /predict kosong")
        prediction = predictions[0]
        if not isinstance(prediction, dict) or "label" not in prediction:
            raise RuntimeError("Format response /predict tidak valid")
        return prediction

    def close(self) -> None:
        self.session.close()


def print_service_info(client: ModelClient) -> None:
    print(f"[MODEL] {client.url} | backend={client.info.get('backend', '?')} | "
          f"kelas={client.info.get('num_classes', len(client.labels) or '?')} | "
          f"seq_len={client.seq_len} | features={client.features} | normalize={client.info.get('normalize', '?')}")
    if client.labels:
        print(f"[MODEL] label ({len(client.labels)}): {', '.join(client.labels[:12])}"
              f"{', ...' if len(client.labels) > 12 else ''}")
        if client.info.get("num_classes") not in (None, len(client.labels)):
            raise RuntimeError("Metadata model tidak konsisten: num_classes berbeda dari jumlah labels")
        if "A" not in client.labels or len(client.labels) < 26:
            print("[WARNING] Label service tidak memuat seluruh abjad A-Z.")
    else:
        raise RuntimeError("Endpoint /labels tidak mengembalikan labels; urutan kelas tidak dapat diverifikasi")


class SignTextDecoder:
    """Temporal consensus and text assembly, without blocking inference on motion."""
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._lock = threading.RLock()
        self.window: deque[np.ndarray] = deque(maxlen=cfg.window_frames)
        self.required_votes = max(cfg.vote_window, cfg.stable_predictions)
        self.history: deque[tuple[str, float]] = deque(maxlen=self.required_votes)
        self.frames_since_predict = 0
        self.prev_frame: np.ndarray | None = None
        self.motion = 0.0
        self.motion_history: deque[bool] = deque(maxlen=cfg.stability_frames)
        self.is_stable = False
        self.gesture_active = False
        self.stable_since: float | None = None
        self.last_motion_at = 0.0
        self.last_commit_at = 0.0
        self.empty_frames = 0
        self.active_ratio = 0.0
        self.hold_label: str | None = None
        self.raw_label = "-"
        self.raw_confidence = 0.0
        self.raw_margin = 0.0
        self.smoothed_label = "-"
        self.smoothed_confidence = 0.0
        self.top_k: list[tuple[str, float]] = []
        self.latency_ms = 0.0
        self.last_error: str | None = None
        self.status = "menunggu tangan"
        self.text = ""
        self.commit_count = 0
        self.last_commit_label: str | None = None

    def push(self, keypoints: np.ndarray, hand_count: int) -> np.ndarray | None:
        frame = np.asarray(keypoints, dtype=np.float32).reshape(-1)
        if frame.size != FEATURES:
            raise ValueError(f"Expected {FEATURES} keypoints, received {frame.size}")
        if not np.isfinite(frame).all():
            frame = np.zeros(FEATURES, dtype=np.float32)
            hand_count = 0
        frame = frame.copy()
        active = hand_count > 0 and float(np.abs(frame).sum()) > 1e-6
        now = time.monotonic()
        with self._lock:
            previous = self.prev_frame
            self.prev_frame = frame if active else None
            if not active:
                # Treat tracking loss as a segment boundary: never join unrelated
                # samples around a gap and do not infer on an all-empty sequence.
                self.window.clear()
                self.frames_since_predict = 0
                self.active_ratio = 0.0
                self.empty_frames += 1
                self.is_stable = False
                self.gesture_active = False
                self.stable_since = None
                self.motion_history.clear()
                self.history.clear()
                self.status = "tidak ada tangan"
                if self.empty_frames >= self.cfg.release_after_empty_frames:
                    self.hold_label = None
                return None

            self.empty_frames = 0
            self.window.append(frame)
            self.frames_since_predict += 1
            # A first tracked frame follows a no-hand observation. Treat appearing
            # hand landmarks as gesture onset so static held signs can pass the
            # optional REQUIRE_GESTURE gate without needing another movement.
            motion = float(np.abs(frame - previous).mean()) if previous is not None else float(np.abs(frame).mean())
            self.motion = motion
            if motion >= self.cfg.gesture_motion_threshold:
                self.gesture_active = True
                self.last_motion_at = now
                self.history.clear()
                self.motion_history.clear()
                self.stable_since = None
            self.motion_history.append(motion <= self.cfg.movement_threshold)
            self.is_stable = len(self.motion_history) == self.cfg.stability_frames and all(self.motion_history)
            if self.is_stable:
                if self.stable_since is None:
                    self.stable_since = now
            else:
                self.stable_since = None
            self.status = "tangan stabil" if self.is_stable else "mendeteksi gesture"

            if len(self.window) < self.cfg.min_window_frames or self.frames_since_predict < self.cfg.stride_frames:
                return None
            self.active_ratio = 1.0
            # Always infer on a visible sliding window. Requiring motion or stillness
            # here would suppress valid static signs and continuous signs.
            self.frames_since_predict = 0
            return np.stack(tuple(self.window)).astype(np.float32, copy=False)

    def register_prediction(self, prediction: dict[str, Any], latency_ms: float) -> None:
        raw_label = str(prediction.get("label", "-")).strip()
        try:
            confidence = float(prediction.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if not np.isfinite(confidence):
            confidence = 0.0
        confidence = float(np.clip(confidence, 0.0, 1.0))
        top_k: list[tuple[str, float]] = []
        for item in (prediction.get("top_k") or [])[:5]:
            try:
                score = float(item.get("confidence", 0.0) or 0.0)
            except (TypeError, ValueError, AttributeError):
                continue
            if np.isfinite(score):
                top_k.append((str(item.get("label", "?")), float(np.clip(score, 0.0, 1.0))))
        margin = confidence - (top_k[1][1] if len(top_k) > 1 else 0.0)
        now = time.monotonic()
        with self._lock:
            self.raw_label, self.raw_confidence, self.raw_margin = raw_label or "-", confidence, margin
            self.top_k, self.latency_ms, self.last_error = top_k, latency_ms, None
            if prediction.get("empty") or raw_label in {"", "-"}:
                self.history.clear()
                self.smoothed_label, self.smoothed_confidence = "-", 0.0
                self.status = "tidak ada gesture"
                return
            if confidence < self.cfg.prediction_threshold:
                self.history.clear()
                self.smoothed_label, self.smoothed_confidence = "-", 0.0
                self.status = f"confidence rendah ({confidence:.0%})"
                return
            if len(top_k) > 1 and margin < self.cfg.top1_margin:
                self.history.clear()
                self.smoothed_label, self.smoothed_confidence = "-", 0.0
                self.status = f"prediksi ambigu (margin {margin:.0%})"
                return
            self.history.append((raw_label, confidence))
            labels = [label for label, _ in self.history]
            counts = {label: labels.count(label) for label in set(labels)}
            candidate = max(counts, key=counts.get)
            candidate_scores = [score for label, score in self.history if label == candidate]
            self.smoothed_label = candidate
            self.smoothed_confidence = sum(candidate_scores) / len(candidate_scores)
            if counts[candidate] < self.required_votes:
                self.status = f"konsensus {counts[candidate]}/{self.required_votes}"
                return
            if self.smoothed_confidence < self.cfg.prediction_threshold:
                self.status = "confidence konsensus rendah"
                return
            if candidate == self.hold_label:
                self.status = "label sudah dikomit; lepaskan tangan untuk mengulang"
                return
            if now - self.last_commit_at < self.cfg.commit_cooldown:
                self.status = "cooldown"
                return
            if self.cfg.require_gesture:
                if not self.gesture_active:
                    self.status = "menunggu gerakan gesture"
                    return
                if self.cfg.gesture_timeout and now - self.last_motion_at > self.cfg.gesture_timeout:
                    self.gesture_active = False
                    self.history.clear()
                    self.status = "gesture timeout; mulai ulang gerakan"
                    return
            if self.cfg.require_stable_motion and not self.is_stable:
                self.status = "menunggu tangan stabil"
                return
            if self.cfg.min_hold_seconds and self.stable_since is not None and now - self.stable_since < self.cfg.min_hold_seconds:
                self.status = "tahan gesture sebentar"
                return
            self._commit_locked(candidate)
            self.hold_label = candidate
            self.last_commit_at = now
            self.history.clear()
            self.status = f"dikomit: {candidate}"

    def _commit_locked(self, label: str) -> None:
        label = label.strip()
        if not label:
            return
        if len(label) == 1 and label.isalpha():
            text = self.text + label.upper()
        else:
            text = self.text.rstrip() + (" " if self.text.strip() else "") + label + " "
        if len(text) > self.cfg.text_limit:
            # Trim only at a word boundary; if one token itself exceeds the limit,
            # retain its tail rather than exceeding the configured memory bound.
            trimmed = text[-self.cfg.text_limit:]
            first_space = trimmed.find(" ")
            self.text = trimmed[first_space + 1:] if first_space >= 0 else trimmed
        else:
            self.text = text
        self.commit_count += 1
        self.last_commit_label = label
        print(f"[TEKS] +{label} -> {self.text.strip()}")

    def register_error(self, message: str) -> None:
        with self._lock:
            self.last_error = message
            self.status = "request prediksi gagal"

    def add_space(self) -> None:
        with self._lock:
            if self.text and not self.text.endswith(" "):
                self.text += " "
            self.clear_temporal_state()

    def backspace(self) -> None:
        with self._lock:
            self.text = self.text.rstrip()[:-1]
            self.clear_temporal_state()

    def reset_decoder_state(self) -> None:
        """Clear temporal predictions after manual text editing."""
        with self._lock:
            self.history.clear()
            self.smoothed_label, self.smoothed_confidence = "-", 0.0
            self.status = "menunggu tangan"

    def clear_temporal_state(self, clear_hold: bool = False) -> None:
        """Start a fresh inference segment while optionally preserving held-label state."""
        with self._lock:
            self.window.clear()
            self.frames_since_predict = 0
            self.prev_frame = None
            self.motion = 0.0
            self.motion_history.clear()
            self.is_stable = False
            self.gesture_active = False
            self.stable_since = None
            self.last_motion_at = 0.0
            self.empty_frames = 0
            self.active_ratio = 0.0
            self.history.clear()
            self.smoothed_label, self.smoothed_confidence = "-", 0.0
            self.raw_label, self.raw_confidence, self.raw_margin = "-", 0.0, 0.0
            self.top_k = []
            self.last_error = None
            self.status = "menunggu tangan"
            if clear_hold:
                self.hold_label = None

    def clear(self) -> None:
        with self._lock:
            self.text = ""
            self.clear_temporal_state(clear_hold=True)
            self.commit_count = 0
            self.last_commit_label = None

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "raw_label": self.raw_label, "raw_confidence": self.raw_confidence,
                "raw_margin": self.raw_margin, "smoothed_label": self.smoothed_label,
                "smoothed_confidence": self.smoothed_confidence, "top_k": list(self.top_k),
                "latency_ms": self.latency_ms, "text": self.text,
                "window_len": len(self.window), "active_ratio": self.active_ratio,
                "motion": self.motion, "stable": self.is_stable,
                "gesture_active": self.gesture_active, "progress": len(self.history),
                "required": self.required_votes, "status": self.status,
                "error": self.last_error, "commits": self.commit_count,
                "last_commit": self.last_commit_label,
            }


class PredictionWorker(threading.Thread):
    """Single inference worker with a replaceable latest-frame-window queue."""
    def __init__(self, client: ModelClient, decoder: SignTextDecoder, debug: bool = False) -> None:
        super().__init__(daemon=True, name="bisindo-prediction-worker")
        self._client, self._decoder, self._debug = client, decoder, debug
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest: tuple[int, np.ndarray] | None = None
        self._generation = 0
        self._active_generation: int | None = None
        self._shutdown = False  # Do not shadow Thread._stop(), which join() calls.

    def submit(self, sequence: np.ndarray) -> None:
        with self._lock:
            # Keep only the newest window. If an older request is still running,
            # queue this window under the current generation; its completion will
            # wake the worker and the stale result will be discarded.
            generation = self._generation
            self._latest = (generation, sequence.copy())
            if self._active_generation is None:
                self._event.set()

    def invalidate(self, reset_decoder: bool = False, clear_hold: bool = False) -> None:
        """Invalidate stale results without blocking the next gesture segment."""
        with self._lock:
            self._generation += 1
            self._latest = None
            if reset_decoder:
                self._decoder.clear_temporal_state(clear_hold=clear_hold)

    def stop(self) -> None:
        self._shutdown = True
        self._event.set()
        if self.is_alive():
            self.join(timeout=25.0)  # request timeout is 20 s
        if self.is_alive():
            print("[WARNING] Prediction worker masih menunggu service; proses dapat berhenti setelah timeout.")

    def run(self) -> None:
        while True:
            self._event.wait()
            if self._shutdown:
                return
            self._event.clear()
            with self._lock:
                pending, self._latest = self._latest, None
                if pending is not None:
                    self._active_generation = pending[0]
            if pending is None:
                continue
            generation, sequence = pending
            started = time.perf_counter()
            try:
                result = self._client.predict(sequence)
                latency = (time.perf_counter() - started) * 1000.0
                with self._lock:
                    current = generation == self._generation
                    if self._active_generation == generation:
                        self._active_generation = None
                    pending_latest = self._latest
                    if pending_latest is not None:
                        self._event.set()
                    if not current:
                        continue
                    self._decoder.register_prediction(result, latency)
                if self._debug:
                    print(f"[PREDIKSI] {result.get('label')} "
                          f"{float(result.get('confidence', 0.0)):.1%} | {latency:.0f} ms")
            except Exception as exc:
                with self._lock:
                    current = generation == self._generation
                    if self._active_generation == generation:
                        self._active_generation = None
                    pending_latest = self._latest
                    if pending_latest is not None:
                        self._event.set()
                if current:
                    self._decoder.register_error(str(exc))


def _draw_points(image: np.ndarray, points: list[tuple[int, int]]) -> None:
    height, width = image.shape[:2]
    points = [(int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))) for x, y in points]
    for start, end in HAND_CONNECTIONS:
        cv2.line(image, points[start], points[end], (0, 255, 0), 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(image, point, 4, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(image, point, 6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray, mirror: bool = True) -> None:
    if keypoints is None or len(keypoints) < FEATURES:
        return
    height, width = image.shape[:2]
    for offset in (0, 63):
        hand = np.asarray(keypoints[offset:offset + 63], dtype=np.float32).reshape(21, 3)
        if not np.isfinite(hand).all() or float(np.abs(hand).sum()) < 1e-6:
            continue
        points = [(int(((1.0 - x) if mirror else x) * width), int(y * height)) for x, y, _ in hand]
        _draw_points(image, points)


def draw_mediapipe_hands(image: np.ndarray, results: Any, mirror: bool = True) -> None:
    height, width = image.shape[:2]
    for hand in (getattr(results, "hand_landmarks", None) or [])[:2]:
        if len(hand) != 21:
            continue
        if not all(np.isfinite((lm.x, lm.y, lm.z)).all() for lm in hand):
            continue
        points = [(int(((1.0 - lm.x) if mirror else lm.x) * width), int(lm.y * height)) for lm in hand]
        _draw_points(image, points)


def wrap_text(text: str, width: int, max_lines: int) -> list[str]:
    cleaned = text.rstrip()
    if not cleaned:
        return ["(belum ada teks)"]
    lines = textwrap.wrap(cleaned, width=max(10, width))
    return (lines or [cleaned[-max(10, width):]])[-max_lines:]


def draw_hud(image: np.ndarray, snap: dict[str, Any], cfg: Config, fps: float, backend: str) -> None:
    height, width = image.shape[:2]
    cv2.rectangle(image, (0, 0), (width, 66), (245, 117, 16), -1)
    label = snap["smoothed_label"] if snap["smoothed_label"] != "-" else "-"
    cv2.putText(image, f"Kandidat: {label} ({snap['smoothed_confidence']:.0%})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, f"raw {snap['raw_label']} {snap['raw_confidence']:.0%} | margin {snap['raw_margin']:.0%} | "
                f"{snap['latency_ms']:.0f} ms | FPS {fps:.0f} | {backend}", (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    bar_width = min(150, max(60, width // 4))
    bar_x = max(10, width - bar_width - 10)
    cv2.rectangle(image, (bar_x, 14), (bar_x + bar_width, 40), (255, 255, 255), 1)
    required = max(1, int(snap["required"]))
    progress = min(required, int(snap["progress"]))
    cv2.rectangle(image, (bar_x, 14), (bar_x + int(bar_width * progress / required), 40),
                  (0, 200, 0) if progress >= required else (0, 165, 255), -1)
    cv2.putText(image, f"{progress}/{required}", (bar_x + 6, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
    panel_top = max(70, height - 120)
    overlay = image.copy()
    cv2.rectangle(overlay, (0, panel_top), (width, max(panel_top, height - 26)), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.putText(image, "Teks:", (10, panel_top + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    for index, line in enumerate(wrap_text(snap["text"], max(16, int((width - 20) / 11)), 3)):
        y = min(height - 32, panel_top + 42 + index * 24)
        cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(image, "[Q] Keluar  [SPACE] Spasi  [BACKSPACE] Hapus  [C] Reset", (10, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (210, 210, 210), 1, cv2.LINE_AA)
    message = snap["error"] or (snap["status"] if cfg.show_status else None)
    if message:
        cv2.putText(image, str(message)[:70], (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 255) if snap["error"] else (0, 215, 255), 1, cv2.LINE_AA)


def open_camera(cfg: Config) -> cv2.VideoCapture:
    for index in (cfg.camera_index, cfg.camera_index + 1, cfg.camera_index + 2):
        capture = cv2.VideoCapture(index)
        if not capture.isOpened():
            capture.release()
            continue
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.capture_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.capture_height)
        capture.set(cv2.CAP_PROP_FPS, cfg.capture_fps)
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        print(f"[KAMERA] indeks {index} | {int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return capture
    raise RuntimeError("Kamera tidak bisa dibuka. Coba --camera 1 atau periksa izin kamera.")


def _handle_key(key: int, decoder: SignTextDecoder, worker: PredictionWorker | None = None) -> bool:
    if key == ord("q"):
        return False
    if key == 32:
        if worker is not None:
            worker.invalidate(reset_decoder=True)
        else:
            decoder.clear_temporal_state()
        decoder.add_space()
    elif key in (8, 127):
        # On Windows, waitKey(1) & 0xFF returns 255 when no key is pressed.
        # Do not treat 255 as Backspace or the temporal window resets every frame.
        if worker is not None:
            worker.invalidate(reset_decoder=True)
        else:
            decoder.clear_temporal_state()
        decoder.backspace()
    elif key in (ord("c"), ord("C")):
        if worker is not None:
            worker.invalidate(reset_decoder=True, clear_hold=True)
            # invalidate() clears the inference segment, while clear() also
            # removes the text already committed to the output buffer.
            decoder.clear()
        else:
            decoder.clear()
    return True


def run_camera(cfg: Config) -> None:
    client = ModelClient(cfg.model_url)
    landmarker: LandmarkSource | None = None
    capture: cv2.VideoCapture | None = None
    worker: PredictionWorker | None = None
    decoder = SignTextDecoder(cfg)
    try:
        client.connect()
        print_service_info(client)
        landmarker = create_landmarker(cfg, str(uuid.uuid4()))
        worker = PredictionWorker(client, decoder, cfg.debug)
        worker.start()
        capture = open_camera(cfg)
        timestamps = TimestampSource()
        fps, count, started = 0.0, 0, time.perf_counter()
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[WARNING] Frame kamera tidak terbaca, keluar.")
                break
            points, hands, results = landmarker.process(frame, timestamps.next())
            if hands == 0:
                # Invalidate inference results from the previous visible segment,
                # but let decoder.push() count consecutive empty-hand frames.
                worker.invalidate()
            sequence = decoder.push(points, hands)
            if sequence is not None:
                worker.submit(sequence)
            display = cv2.flip(frame, 1)
            if results is not None:
                draw_mediapipe_hands(display, results, mirror=True)
            elif hands:
                draw_keypoints(display, points, mirror=True)
            count += 1
            now = time.perf_counter()
            if now - started >= 0.5:
                fps, count, started = count / (now - started), 0, now
            draw_hud(display, decoder.snapshot(), cfg, fps, landmarker.backend)
            cv2.imshow("Deteksi BISINDO - Teks Realtime", display)
            if not _handle_key(cv2.waitKey(1) & 0xFF, decoder, worker):
                break
    finally:
        if worker is not None:
            worker.stop()
        if capture is not None:
            capture.release()
        if landmarker is not None:
            landmarker.close()
        client.close()
        cv2.destroyAllWindows()
    final = decoder.snapshot()["text"].strip()
    print(f"\nTeks akhir: {final or '(kosong)'}")


def run_video(cfg: Config, video_path: Path, show: bool, max_frames: int | None) -> None:
    client = ModelClient(cfg.model_url)
    landmarker: LandmarkSource | None = None
    capture: cv2.VideoCapture | None = None
    decoder = SignTextDecoder(cfg)
    index = 0
    try:
        client.connect()
        print_service_info(client)
        landmarker = create_landmarker(cfg, str(uuid.uuid4()))
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Video tidak bisa dibuka: {video_path}")
        source_fps = capture.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(source_fps) or source_fps <= 0:
            source_fps = 30.0
        while max_frames is None or index < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = int(index / source_fps * 1000)
            points, hands, results = landmarker.process(frame, timestamp_ms)
            sequence = decoder.push(points, hands)
            if sequence is not None:
                started = time.perf_counter()
                try:
                    prediction = client.predict(sequence)
                    decoder.register_prediction(prediction, (time.perf_counter() - started) * 1000.0)
                    if cfg.debug:
                        print(f"[PREDIKSI] {prediction.get('label')} {float(prediction.get('confidence', 0.0)):.1%}")
                except Exception as exc:
                    decoder.register_error(str(exc))
            if show:
                display = frame.copy()
                if results is not None:
                    draw_mediapipe_hands(display, results, mirror=False)
                elif hands:
                    draw_keypoints(display, points, mirror=False)
                draw_hud(display, decoder.snapshot(), cfg, source_fps, landmarker.backend)
                cv2.imshow("Deteksi BISINDO - Uji Video", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    index += 1
                    break
            index += 1
    finally:
        if capture is not None:
            capture.release()
        if landmarker is not None:
            landmarker.close()
        client.close()
        cv2.destroyAllWindows()
    snap = decoder.snapshot()
    print(f"\nTotal frame diproses: {index} | total commit: {snap['commits']}")
    print(f"Teks akhir: {snap['text'].strip() or '(kosong)'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deteksi BISINDO realtime menjadi teks (model A-Z + kata).")
    parser.add_argument("--camera", type=int, default=None, help="Indeks kamera (default: env CAMERA_INDEX / 0)")
    parser.add_argument("--backend", choices=["auto", "local", "docker"], default=None,
                        help="Backend landmark: local, docker, atau auto")
    parser.add_argument("--video", type=Path, default=None, help="Uji memakai video, bukan kamera")
    parser.add_argument("--show", action="store_true", help="Tampilkan preview video")
    parser.add_argument("--max-frames", type=int, default=None, help="Batas frame dalam mode video")
    parser.add_argument("--debug", action="store_true", help="Cetak prediksi tiap inferensi")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.max_frames is not None and args.max_frames <= 0:
        print("[ERROR] --max-frames harus lebih besar dari 0")
        return 2
    cfg = load_config(args)
    print_banner(cfg)
    try:
        if args.video is not None:
            run_video(cfg, args.video, args.show, args.max_frames)
        else:
            run_camera(cfg)
    except KeyboardInterrupt:
        print("\nDihentikan oleh pengguna.")
    except (RuntimeError, requests.RequestException) as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
