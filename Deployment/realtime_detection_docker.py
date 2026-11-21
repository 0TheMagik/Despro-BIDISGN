"""Deteksi BISINDO realtime -> teks (model terbaru: 97 kelas A-Z + kata).

Alur:
    frame kamera
      -> landmark tangan (MediaPipe lokal ATAU service handlandmarker di Docker)
      -> window keypoint (default 45 frame) -> POST /predict ke model service
      -> voting + stabilisasi -> commit ke teks (huruf digabung, kata diberi spasi)

Contoh:
    python realtime_detection_docker.py                 # backend landmark otomatis
    python realtime_detection_docker.py --backend docker
    python realtime_detection_docker.py --video rekaman.mp4 --show
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
except ImportError:
    load_dotenv = None

SCRIPT_DIR = Path(__file__).resolve().parent

if load_dotenv is not None:
    load_dotenv(dotenv_path=SCRIPT_DIR / ".env")

FEATURES = 126

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
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
        candidates.append(Path(explicit))
    candidates.extend(
        [
            SCRIPT_DIR / "handlandmarker" / "hand_landmarker.task",
            SCRIPT_DIR.parent / "handlandmarker" / "hand_landmarker.task",
            Path("handlandmarker/hand_landmarker.task"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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
    stride_frames = max(1, env_int("STRIDE_FRAMES", env_int("PREDICT_EVERY_N_FRAMES", 3)))
    threshold = min(0.999, max(0.5, env_float("PREDICTION_THRESHOLD", 0.95)))
    top1_margin = min(0.95, max(0.0, env_float("TOP1_MARGIN", 0.20)))
    vote_window = max(1, env_int("VOTE_WINDOW", 5))
    stable_predictions = max(1, env_int("STABLE_PREDICTIONS", 5))
    release_after_empty_frames = max(1, env_int("RELEASE_AFTER_EMPTY_FRAMES", 8))
    min_active_ratio = min(0.95, max(0.0, env_float("MIN_ACTIVE_RATIO", 0.6)))
    movement_threshold = max(0.0005, env_float("MOVEMENT_THRESHOLD", 0.012))
    gesture_motion_threshold = max(movement_threshold, env_float("GESTURE_MOTION_THRESHOLD", 0.02))
    stability_frames = max(1, env_int("STABILITY_FRAMES", 6))
    gesture_timeout = max(0.5, env_float("GESTURE_TIMEOUT_SECONDS", 4.0))
    min_hold_seconds = max(0.0, env_float("MIN_HOLD_SECONDS", 0.3))
    commit_cooldown = max(0.0, env_float("COMMIT_COOLDOWN_SECONDS", 0.6))
    require_stable_motion = env_bool("REQUIRE_STABLE_MOTION", True)
    require_gesture = env_bool("REQUIRE_GESTURE", True)
    show_status = env_bool("SHOW_STATUS", False)

    landmark_model_path = resolve_landmark_model_path()
    backend = (args.backend or os.getenv("LANDMARK_BACKEND", "auto")).strip().lower()
    if backend not in {"auto", "local", "docker"}:
        backend = "auto"
    if backend == "auto":
        has_mediapipe = importlib.util.find_spec("mediapipe") is not None
        backend = "local" if has_mediapipe and landmark_model_path is not None else "docker"

    return Config(
        handlandmarker_url=os.getenv("HANDLANDMARKER_URL", "http://localhost:8001").rstrip("/"),
        model_url=os.getenv("MODEL_URL", "https://despro-bidisgn-model-1031455690132.asia-southeast2.run.app/").rstrip("/"),
        landmark_backend=backend,
        landmark_model_path=landmark_model_path,
        window_frames=window_frames,
        min_window_frames=min_window_frames,
        stride_frames=stride_frames,
        prediction_threshold=threshold,
        top1_margin=top1_margin,
        vote_window=vote_window,
        stable_predictions=stable_predictions,
        release_after_empty_frames=release_after_empty_frames,
        min_active_ratio=min_active_ratio,
        movement_threshold=movement_threshold,
        gesture_motion_threshold=gesture_motion_threshold,
        stability_frames=stability_frames,
        gesture_timeout=gesture_timeout,
        min_hold_seconds=min_hold_seconds,
        commit_cooldown=commit_cooldown,
        require_stable_motion=require_stable_motion,
        require_gesture=require_gesture,
        capture_width=max(160, env_int("CAPTURE_WIDTH", 640)),
        capture_height=max(120, env_int("CAPTURE_HEIGHT", 480)),
        capture_fps=max(5, env_int("CAPTURE_FPS", 30)),
        camera_index=args.camera if args.camera is not None else env_int("CAMERA_INDEX", 0),
        jpeg_quality=min(95, max(40, env_int("JPEG_QUALITY", 80))),
        text_limit=max(40, env_int("TEXT_LIMIT", 160)),
        show_status=show_status,
        debug=bool(args.debug or env_bool("DEBUG", False)),
    )


def print_banner(cfg: Config) -> None:
    print("=" * 64)
    print(" Deteksi BISINDO realtime -> teks (97 kelas: A-Z + kata)")
    print("=" * 64)
    print(f" Model service : {cfg.model_url}")
    print(f" Landmark      : {cfg.landmark_backend} ({cfg.landmark_model_path or 'service Docker'})")
    print(f" Window        : {cfg.min_window_frames}-{cfg.window_frames} frame, stride {cfg.stride_frames}")
    print(
        f" Filter teks   : conf>={cfg.prediction_threshold:.2f} | margin>={cfg.top1_margin:.2f} | "
        f"konsensus {max(cfg.vote_window, cfg.stable_predictions)}x | tahan {cfg.min_hold_seconds:.1f}s"
    )
    print(
        f" Gerak/stabil  : gerak>={cfg.gesture_motion_threshold:.3f} | diam<={cfg.movement_threshold:.3f} "
        f"x{cfg.stability_frames} frame | cooldown {cfg.commit_cooldown:.1f}s"
    )
    print("=" * 64)


def make_http_session() -> requests.Session:
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=8, max_retries=0)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


class TimestampSource:
    """Timestamp monoton naik untuk MediaPipe VIDEO mode."""

    def __init__(self) -> None:
        self._last = 0

    def next(self) -> int:
        timestamp = int(time.perf_counter() * 1000)
        if timestamp <= self._last:
            timestamp = self._last + 1
        self._last = timestamp
        return timestamp


def extract_keypoints(results: Any) -> tuple[np.ndarray, int]:
    hands = getattr(results, "hand_landmarks", None) or []
    if not hands:
        return np.zeros(FEATURES, dtype=np.float32), 0

    values: list[float] = []
    for hand in hands[:2]:
        for landmark in hand:
            values.extend((landmark.x, landmark.y, landmark.z))

    array = np.asarray(values, dtype=np.float32)
    if array.size < FEATURES:
        array = np.concatenate([array, np.zeros(FEATURES - array.size, dtype=np.float32)])
    return array[:FEATURES], min(len(hands), 2)


class LandmarkSource:
    backend = "?"

    def process(self, frame: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray, int, Any]:
        raise NotImplementedError

    def close(self) -> None:
        pass


class LocalLandmarks(LandmarkSource):
    """MediaPipe langsung di proses ini: paling cepat (tanpa HTTP + tanpa encode JPEG)."""

    backend = "local"

    def __init__(self, model_path: Path | None) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe belum terpasang. Jalankan: pip install mediapipe") from exc

        if model_path is None or not model_path.exists():
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
        try:
            results = self._landmarker.detect_for_video(mp_image, int(timestamp_ms))
        except Exception as exc:
            now = time.perf_counter()
            if now - self._last_error_at > 2.0:
                print(f"[WARNING] MediaPipe gagal memproses frame: {exc}")
                self._last_error_at = now
            return np.zeros(FEATURES, dtype=np.float32), 0, None

        keypoints, hand_count = extract_keypoints(results)
        return keypoints, hand_count, results

    def close(self) -> None:
        try:
            self._landmarker.close()
        except Exception:
            pass


class DockerLandmarks(LandmarkSource):
    """Landmark lewat service handlandmarker (Docker/Cloud Run)."""

    backend = "docker"

    def __init__(self, base_url: str, session_id: str, jpeg_quality: int, timeout: float = 15.0) -> None:
        self.url = base_url.rstrip("/")
        self.session_id = session_id
        self.jpeg_quality = jpeg_quality
        self.timeout = timeout
        self.session = make_http_session()
        self._last_error_at = 0.0

    def start(self) -> None:
        try:
            response = self.session.post(
                f"{self.url}/session/start",
                json={"session_id": self.session_id},
                timeout=10,
            )
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"Service handlandmarker tidak dapat dihubungi di {self.url}: {exc}\n"
                "  Jalankan: docker compose -f Deployment/docker-compose.yml up --build"
            ) from exc

    def process(self, frame: np.ndarray, timestamp_ms: int) -> tuple[np.ndarray, int, Any]:
        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
        if not ok:
            return np.zeros(FEATURES, dtype=np.float32), 0, None

        payload = {
            "session_id": self.session_id,
            "frame_data": base64.b64encode(buffer).decode("utf-8"),
            "timestamp_ms": int(timestamp_ms),
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        }

        try:
            response = self.session.post(f"{self.url}/session/frame", json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
        except Exception as exc:
            now = time.perf_counter()
            if now - self._last_error_at > 2.0:
                print(f"[WARNING] Gagal mengirim frame ke handlandmarker: {exc}")
                self._last_error_at = now
            return np.zeros(FEATURES, dtype=np.float32), 0, None

        keypoints = np.asarray(data.get("keypoints") or [], dtype=np.float32)
        if keypoints.size < FEATURES:
            keypoints = np.concatenate([keypoints, np.zeros(FEATURES - keypoints.size, dtype=np.float32)])
        keypoints = keypoints[:FEATURES]
        hand_count = int(data.get("hand_count", 1 if float(np.abs(keypoints).sum()) > 0 else 0))
        return keypoints, hand_count, None

    def close(self) -> None:
        try:
            self.session.post(
                f"{self.url}/session/end",
                json={"session_id": self.session_id},
                timeout=5,
            )
        except Exception:
            pass


def create_landmarker(cfg: Config, session_id: str) -> LandmarkSource:
    if cfg.landmark_backend == "local":
        try:
            source = LocalLandmarks(cfg.landmark_model_path)
            print(f"[LANDMARK] MediaPipe lokal: {cfg.landmark_model_path}")
            return source
        except Exception as exc:
            print(f"[WARNING] Landmark lokal gagal ({exc}); memakai service Docker.")

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
                health = self.session.get(f"{self.url}/health", timeout=5)
                health.raise_for_status()
                self.info = health.json()
                try:
                    labels_response = self.session.get(f"{self.url}/labels", timeout=5)
                    if labels_response.ok:
                        payload = labels_response.json()
                        self.labels = [str(label) for label in payload.get("labels", [])]
                        self.seq_len = int(payload.get("seq_len", self.seq_len))
                        self.features = int(payload.get("features", self.features))
                except Exception:
                    pass
                return self.info
            except Exception as exc:
                last_error = exc
                if attempt < attempts:
                    print(f"[INFO] Menunggu model service siap ({attempt}/{attempts}) ...")
                    time.sleep(delay)

        raise RuntimeError(
            f"Model service tidak dapat dihubungi di {self.url}.\n"
            "  Jalankan: docker compose -f Deployment/docker-compose.yml up --build\n"
            f"  Detail: {last_error}"
        )

    def predict(self, sequence: np.ndarray) -> dict[str, Any]:
        payload = {"instances": [np.round(sequence, 5).tolist()]}
        response = self.session.post(f"{self.url}/predict", json=payload, timeout=15)
        response.raise_for_status()
        predictions = response.json().get("predictions") or []
        if not predictions:
            raise RuntimeError("Response /predict kosong")
        return predictions[0]


def print_service_info(client: ModelClient) -> None:
    info = client.info
    print(
        f"[MODEL] {client.url} | backend={info.get('backend', '?')} | "
        f"kelas={info.get('num_classes', len(client.labels) or '?')} | "
        f"seq_len={info.get('seq_len', client.seq_len)} | "
        f"normalize={info.get('normalize', '?')}"
    )
    if client.labels:
        preview = ", ".join(client.labels[:12])
        suffix = ", ..." if len(client.labels) > 12 else ""
        print(f"[MODEL] label: {preview}{suffix}")

    if not client.labels:
        print(
            "[PERINGATAN] Service model tidak mengirim daftar label (/labels tidak tersedia).\n"
            "            Kemungkinan container masih image lama yang hanya punya 10 kelas,\n"
            "            sehingga huruf A-Z tidak akan pernah terdeteksi.\n"
            "            Jalankan: docker compose -f Deployment/docker-compose.yml up --build -d"
        )
    elif "A" not in client.labels or len(client.labels) < 26:
        print(
            f"[PERINGATAN] Model service hanya punya {len(client.labels)} kelas dan tidak memuat abjad A-Z.\n"
            "            Pastikan image model sudah di-rebuild dengan model terbaru yang memuat abjad A-Z."
        )


class SignTextDecoder:
    """Menyimpan window keypoint, filter gestur, konsensus prediksi, dan menyusun teks.

    Teks hanya dikomit bila SEMUA syarat berikut terpenuhi:
    1. Tangan diam/stabil sesaat (mode wajib gerakan bisa diaktifkan lewat env).
    2. Confidence >= threshold dan margin top-1 vs top-2 cukup besar.
    3. Label konsisten untuk beberapa prediksi berturut-turut (konsensus).
    4. Cooldown antar-komit sudah lewat dan label belum sedang "dipegang".
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._lock = threading.Lock()
        self.window: deque[np.ndarray] = deque(maxlen=cfg.window_frames)
        self.required_votes = max(cfg.vote_window, cfg.stable_predictions)
        self.history: deque[tuple[str, float]] = deque(maxlen=self.required_votes)

        self.frames_since_predict = cfg.stride_frames
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
        self.status = "menunggu gerakan"

        self.text = ""
        self.commit_count = 0
        self.last_commit_label: str | None = None

    def push(self, keypoints: np.ndarray, hand_count: int) -> np.ndarray | None:
        frame = np.asarray(keypoints, dtype=np.float32).reshape(-1)
        if frame.size < FEATURES:
            frame = np.concatenate([frame, np.zeros(FEATURES - frame.size, dtype=np.float32)])
        frame = frame[:FEATURES]

        previous = self.prev_frame
        self.prev_frame = frame.copy()
        timestamp = time.monotonic()

        motion: float | None = None
        if (
            previous is not None
            and hand_count > 0
            and float(np.abs(frame).sum()) > 1e-6
            and float(np.abs(previous).sum()) > 1e-6
        ):
            motion = float(np.abs(frame - previous).mean())

        with self._lock:
            self.window.append(frame)
            self.frames_since_predict += 1
            self.motion = 0.0 if motion is None else motion
            if motion is None:
                self.stable_since = None

            if hand_count <= 0:
                self.empty_frames += 1
                self.is_stable = False
                self.gesture_active = False
                self.stable_since = None
                self.motion_history.clear()
                self.history.clear()
                self.status = "tidak ada tangan"
                if self.empty_frames >= self.cfg.release_after_empty_frames:
                    self._release_hold_locked()
            else:
                self.empty_frames = 0

            if hand_count > 0 and motion is not None:
                if motion >= self.cfg.gesture_motion_threshold:
                    # Gerakan terdeteksi -> mulai sesi gestur baru, buang konsensus lama.
                    self.gesture_active = True
                    self.last_motion_at = timestamp
                    self.is_stable = False
                    self.stable_since = None
                    self.motion_history.clear()
                    self.history.clear()
                    self.status = "gerakan terdeteksi"
                else:
                    self.motion_history.append(motion <= self.cfg.movement_threshold)
                    self.is_stable = (
                        len(self.motion_history) >= self.cfg.stability_frames
                        and all(self.motion_history)
                    )
                    if self.is_stable:
                        if self.stable_since is None:
                            self.stable_since = timestamp
                    else:
                        self.stable_since = None

                    if self.is_stable and self.gesture_active:
                        if (
                            self.cfg.require_gesture
                            and timestamp - self.last_motion_at > self.cfg.gesture_timeout
                        ):
                            self.gesture_active = False
                            self.history.clear()
                            self.status = "gestur kadaluarsa"
                        elif not self.history:
                            self.status = "tahan gestur..."
                    elif not self.is_stable:
                        self.status = "menunggu tangan diam"

            window_len = len(self.window)
            if window_len < self.cfg.min_window_frames:
                return None

            active_frames = sum(1 for item in self.window if float(np.abs(item).sum()) > 1e-6)
            self.active_ratio = active_frames / window_len
            if self.active_ratio < self.cfg.min_active_ratio:
                self.history.clear()
                self.status = "tangan kurang terlihat"
                return None
            if self.frames_since_predict < self.cfg.stride_frames:
                return None
            if self.cfg.require_gesture and not self.gesture_active:
                self.status = "menunggu gerakan"
                return None
            if self.cfg.require_stable_motion and not self.is_stable:
                self.status = "menunggu stabil"
                return None

            self.frames_since_predict = 0
            return np.array(self.window, dtype=np.float32, copy=True)

    def register_prediction(self, prediction: dict[str, Any], latency_ms: float) -> None:
        raw_label = str(prediction.get("label", "-"))
        confidence = float(prediction.get("confidence", 0.0) or 0.0)
        top_k = [
            (str(item.get("label", "?")), float(item.get("confidence", 0.0) or 0.0))
            for item in (prediction.get("top_k") or [])[:5]
        ]
        runner_up = top_k[1][1] if len(top_k) > 1 else 0.0
        margin = confidence - runner_up
        timestamp = time.monotonic()

        with self._lock:
            self.raw_label = raw_label
            self.raw_confidence = confidence
            self.raw_margin = margin
            self.top_k = top_k
            self.latency_ms = latency_ms
            self.last_error = None

            if prediction.get("empty") or raw_label == "-":
                self.history.clear()
                self.status = "tidak ada gestur"
                return
            if confidence < self.cfg.prediction_threshold:
                self.history.clear()
                self.status = f"confidence {confidence * 100:.0f}% < {self.cfg.prediction_threshold * 100:.0f}%"
                return
            if margin < self.cfg.top1_margin:
                self.history.clear()
                self.status = f"ambigu (margin {margin * 100:.0f}%)"
                return
            if self.cfg.require_gesture and not self.gesture_active:
                self.history.clear()
                self.status = "menunggu gerakan"
                return
            if self.cfg.require_stable_motion and not self.is_stable:
                self.history.clear()
                self.status = "menunggu stabil"
                return

            self.history.append((raw_label, confidence))
            self.smoothed_label = raw_label
            self.smoothed_confidence = sum(score for _, score in self.history) / len(self.history)

            labels = {label for label, _ in self.history}
            if len(labels) != 1:
                self.history.clear()
                self.history.append((raw_label, confidence))
                self.status = "label berubah, ulangi"
                return

            if len(self.history) < self.required_votes:
                self.status = f"konsensus {len(self.history)}/{self.required_votes}"
                return

            if self.smoothed_confidence < self.cfg.prediction_threshold:
                self.history.clear()
                self.status = "confidence rata-rata rendah"
                return

            stable_label = raw_label
            if stable_label == self.hold_label:
                self.status = "sudah dikomit"
                return
            if timestamp - self.last_commit_at < self.cfg.commit_cooldown:
                self.status = "cooldown"
                return

            held_seconds = 0.0 if self.stable_since is None else timestamp - self.stable_since
            if held_seconds < self.cfg.min_hold_seconds:
                self.status = f"tahan {held_seconds:.1f}/{self.cfg.min_hold_seconds:.1f}s"
                return
            if self.cfg.require_stable_motion and not self.is_stable:
                self.history.clear()
                self.status = "menunggu stabil"
                return

            self._commit_locked(stable_label)
            self.hold_label = stable_label
            self.last_commit_at = timestamp
            self.history.clear()
            self.gesture_active = False
            self.is_stable = False
            self.motion_history.clear()
            self.status = f"dikomit: {stable_label}"

    def _release_hold_locked(self) -> None:
        self.hold_label = None

    def _commit_locked(self, label: str) -> None:
        label = label.strip()
        if not label:
            return

        if len(label) == 1 and label.isalpha():
            self.text += label.upper()
        else:
            if self.text and not self.text.endswith(" "):
                self.text += " "
            self.text += label + " "

        if len(self.text) > self.cfg.text_limit:
            self.text = self.text[-self.cfg.text_limit :]

        self.commit_count += 1
        self.last_commit_label = label
        print(f"[TEKS] +{label} -> {self.text.strip()}")

    def register_error(self, message: str) -> None:
        with self._lock:
            self.last_error = message

    def add_space(self) -> None:
        with self._lock:
            if self.text and not self.text.endswith(" "):
                self.text += " "

    def backspace(self) -> None:
        with self._lock:
            self.text = self.text.rstrip()
            self.text = self.text[:-1]

    def clear(self) -> None:
        with self._lock:
            self.text = ""
            self.history.clear()
            self.hold_label = None
            self.gesture_active = False
            self.is_stable = False
            self.stable_since = None
            self.motion_history.clear()
            self.smoothed_label = "-"
            self.smoothed_confidence = 0.0
            self.raw_margin = 0.0
            self.status = "menunggu gerakan"

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "raw_label": self.raw_label,
                "raw_confidence": self.raw_confidence,
                "raw_margin": self.raw_margin,
                "smoothed_label": self.smoothed_label,
                "smoothed_confidence": self.smoothed_confidence,
                "top_k": list(self.top_k),
                "latency_ms": self.latency_ms,
                "text": self.text,
                "window_len": len(self.window),
                "active_ratio": self.active_ratio,
                "motion": self.motion,
                "stable": self.is_stable,
                "gesture_active": self.gesture_active,
                "progress": len(self.history),
                "required": self.required_votes,
                "status": self.status,
                "error": self.last_error,
                "commits": self.commit_count,
                "last_commit": self.last_commit_label,
            }


class PredictionWorker(threading.Thread):
    """Inferensi di thread terpisah supaya loop kamera tetap 30 FPS."""

    def __init__(self, client: ModelClient, decoder: SignTextDecoder, debug: bool = False) -> None:
        super().__init__(daemon=True)
        self._client = client
        self._decoder = decoder
        self._debug = debug
        self._lock = threading.Lock()
        self._event = threading.Event()
        self._latest: np.ndarray | None = None
        self._stop = False

    def submit(self, sequence: np.ndarray) -> None:
        with self._lock:
            self._latest = sequence
        self._event.set()

    def stop(self) -> None:
        self._stop = True
        self._event.set()
        self.join(timeout=2.0)

    def run(self) -> None:
        while True:
            self._event.wait(0.5)
            if self._stop:
                return
            self._event.clear()

            with self._lock:
                sequence = self._latest
                self._latest = None
            if sequence is None:
                continue

            started = time.perf_counter()
            try:
                prediction = self._client.predict(sequence)
            except Exception as exc:
                self._decoder.register_error(str(exc))
                continue

            latency_ms = (time.perf_counter() - started) * 1000.0
            self._decoder.register_prediction(prediction, latency_ms)

            if self._debug:
                top = ", ".join(
                    f"{item.get('label')}:{float(item.get('confidence', 0.0)):.2f}"
                    for item in (prediction.get("top_k") or [])[:3]
                )
                print(
                    f"[PREDIKSI] {prediction.get('label')} "
                    f"{float(prediction.get('confidence', 0.0)) * 100:.1f}% | {top} | {latency_ms:.0f} ms"
                )


def draw_keypoints(image: np.ndarray, keypoints: np.ndarray, mirror: bool = True) -> None:
    if keypoints is None or len(keypoints) < FEATURES:
        return
    height, width = image.shape[:2]
    for offset in (0, 63):
        hand = np.asarray(keypoints[offset : offset + 63], dtype=np.float32).reshape(21, 3)
        if float(np.abs(hand).sum()) < 1e-6:
            continue
        points = []
        for x, y, _ in hand:
            px = int(((1.0 - x) if mirror else x) * width)
            py = int(y * height)
            points.append((px, py))
        for start, end in HAND_CONNECTIONS:
            cv2.line(image, points[start], points[end], (0, 255, 0), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(image, point, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(image, point, 6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_mediapipe_hands(image: np.ndarray, results: Any, mirror: bool = True) -> None:
    hands = getattr(results, "hand_landmarks", None) or []
    height, width = image.shape[:2]
    for hand in hands[:2]:
        points = []
        for landmark in hand:
            px = int(((1.0 - landmark.x) if mirror else landmark.x) * width)
            py = int(landmark.y * height)
            points.append((px, py))
        for start, end in HAND_CONNECTIONS:
            cv2.line(image, points[start], points[end], (0, 255, 0), 2, cv2.LINE_AA)
        for point in points:
            cv2.circle(image, point, 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.circle(image, point, 6, (255, 255, 255), 1, cv2.LINE_AA)


def wrap_text(text: str, width: int, max_lines: int) -> list[str]:
    cleaned = text.rstrip()
    if not cleaned:
        return ["(belum ada teks)"]
    lines = textwrap.wrap(cleaned, width=max(10, width))
    if not lines:
        return [cleaned[-max(10, width) :]]
    return lines[-max_lines:]


def draw_hud(image: np.ndarray, snap: dict[str, Any], cfg: Config, fps: float, backend: str) -> None:
    height, width = image.shape[:2]

    cv2.rectangle(image, (0, 0), (width, 66), (245, 117, 16), -1)

    label = snap["smoothed_label"] if snap["smoothed_label"] != "-" else "-"
    confidence = snap["smoothed_confidence"] * 100.0
    cv2.putText(
        image,
        f"Kandidat: {label} ({confidence:.0f}%)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        f"raw {snap['raw_label']} {snap['raw_confidence'] * 100:.0f}% | margin {snap['raw_margin'] * 100:.0f}% | "
        f"{snap['latency_ms']:.0f} ms | FPS {fps:.0f} | {backend}",
        (10, 56),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    bar_width = 150
    bar_x = width - bar_width - 10
    cv2.rectangle(image, (bar_x, 14), (bar_x + bar_width, 40), (255, 255, 255), 1)
    required = max(1, int(snap["required"]))
    progress = min(required, int(snap["progress"]))
    fill_ratio = progress / required
    ready = progress >= required
    cv2.rectangle(
        image,
        (bar_x, 14),
        (bar_x + int(bar_width * fill_ratio), 40),
        (0, 200, 0) if ready else (0, 165, 255),
        -1,
    )
    cv2.putText(
        image,
        f"{progress}/{required}",
        (bar_x + 6, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    panel_top = height - 120
    overlay = image.copy()
    cv2.rectangle(overlay, (0, panel_top), (width, height - 26), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
    cv2.putText(
        image,
        "Teks:",
        (10, panel_top + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    width_chars = max(16, int((width - 20) / 11))
    for index, line in enumerate(wrap_text(snap["text"], width_chars, 3)):
        cv2.putText(
            image,
            line,
            (10, panel_top + 42 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        image,
        "[Q] Keluar  [SPACE] Spasi  [BACKSPACE] Hapus  [C] Reset",
        (10, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )

    if snap["error"]:
        cv2.putText(
            image,
            f"Error: {str(snap['error'])[:70]}",
            (10, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    elif cfg.show_status:
        status = str(snap["status"])
        if snap["required"] > 0 and snap["progress"] > 0:
            status = f"{status}  ({snap['progress']}/{snap['required']})"
        cv2.putText(
            image,
            status[:60],
            (10, 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 215, 255),
            1,
            cv2.LINE_AA,
        )


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

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[KAMERA] indeks {index} | {width}x{height}")
        return capture

    raise RuntimeError("Kamera tidak bisa dibuka. Coba `--camera 1` atau cek izin kamera.")


def run_camera(cfg: Config) -> None:
    client = ModelClient(cfg.model_url)
    client.connect()
    print_service_info(client)

    session_id = str(uuid.uuid4())
    landmarker = create_landmarker(cfg, session_id)
    decoder = SignTextDecoder(cfg)
    worker = PredictionWorker(client, decoder, cfg.debug)
    worker.start()

    capture = open_camera(cfg)
    timestamps = TimestampSource()

    fps_value = 0.0
    fps_counter = 0
    fps_started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("[WARNING] Frame kamera tidak terbaca, keluar.")
                break

            keypoints, hand_count, results = landmarker.process(frame, timestamps.next())
            request_sequence = decoder.push(keypoints, hand_count)
            if request_sequence is not None:
                worker.submit(request_sequence)

            display = cv2.flip(frame, 1)
            if results is not None:
                draw_mediapipe_hands(display, results, mirror=True)
            elif hand_count > 0:
                draw_keypoints(display, keypoints, mirror=True)

            fps_counter += 1
            now = time.perf_counter()
            if now - fps_started >= 0.5:
                fps_value = fps_counter / (now - fps_started)
                fps_counter = 0
                fps_started = now

            draw_hud(display, decoder.snapshot(), cfg, fps_value, landmarker.backend)
            cv2.imshow("Deteksi BISINDO - Teks Realtime", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32:
                decoder.add_space()
            elif key in (8, 127):
                decoder.backspace()
            elif key in (ord("c"), ord("C")):
                decoder.clear()
    finally:
        worker.stop()
        landmarker.close()
        capture.release()
        cv2.destroyAllWindows()

    final_text = decoder.snapshot()["text"].strip()
    print(f"\nTeks akhir: {final_text or '(kosong)'}")


def run_video(cfg: Config, video_path: Path, show: bool, max_frames: int | None) -> None:
    client = ModelClient(cfg.model_url)
    client.connect()
    print_service_info(client)

    session_id = str(uuid.uuid4())
    landmarker = create_landmarker(cfg, session_id)
    decoder = SignTextDecoder(cfg)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        landmarker.close()
        raise RuntimeError(f"Video tidak bisa dibuka: {video_path}")

    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            timestamp_ms = int(index / source_fps * 1000)
            keypoints, hand_count, results = landmarker.process(frame, timestamp_ms)
            request_sequence = decoder.push(keypoints, hand_count)

            if request_sequence is not None:
                started = time.perf_counter()
                try:
                    prediction = client.predict(request_sequence)
                except Exception as exc:
                    decoder.register_error(str(exc))
                else:
                    latency_ms = (time.perf_counter() - started) * 1000.0
                    decoder.register_prediction(prediction, latency_ms)
                    if cfg.debug:
                        top = ", ".join(
                            f"{item.get('label')}:{float(item.get('confidence', 0.0)):.2f}"
                            for item in (prediction.get("top_k") or [])[:3]
                        )
                        print(
                            f"[PREDIKSI] {prediction.get('label')} "
                            f"{float(prediction.get('confidence', 0.0)) * 100:.1f}% | {top}"
                        )

            if show:
                display = frame.copy()
                if results is not None:
                    draw_mediapipe_hands(display, results, mirror=False)
                elif hand_count > 0:
                    draw_keypoints(display, keypoints, mirror=False)
                draw_hud(display, decoder.snapshot(), cfg, source_fps, landmarker.backend)
                cv2.imshow("Deteksi BISINDO - Uji Video", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            index += 1
            if max_frames is not None and index >= max_frames:
                break
    finally:
        capture.release()
        landmarker.close()
        cv2.destroyAllWindows()

    snap = decoder.snapshot()
    print(f"\nTotal frame diproses: {index} | total commit: {snap['commits']}")
    print(f"Teks akhir: {snap['text'].strip() or '(kosong)'}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deteksi BISINDO realtime menjadi teks (model A-Z + kata)."
    )
    parser.add_argument("--camera", type=int, default=None, help="Indeks kamera (default: env CAMERA_INDEX / 0)")
    parser.add_argument(
        "--backend",
        choices=["auto", "local", "docker"],
        default=None,
        help="Sumber landmark tangan (default: env LANDMARK_BACKEND / auto)",
    )
    parser.add_argument("--video", type=Path, default=None, help="Uji memakai file video, bukan kamera")
    parser.add_argument("--show", action="store_true", help="Tampilkan preview saat memakai --video")
    parser.add_argument("--max-frames", type=int, default=None, help="Batasi jumlah frame pada mode --video")
    parser.add_argument("--debug", action="store_true", help="Cetak top-5 prediksi tiap inferensi")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args)
    print_banner(cfg)

    try:
        if args.video is not None:
            run_video(cfg, args.video, args.show, args.max_frames)
        else:
            run_camera(cfg)
    except KeyboardInterrupt:
        print("\nDihentikan oleh pengguna.")
    except RuntimeError as exc:
        print(f"\n[ERROR] {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())