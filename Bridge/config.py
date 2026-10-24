"""Konfigurasi Bridge BISINDO.

Semua nilai bisa dioverride lewat environment variable. File ``Bridge/.env``
dibaca otomatis bila ada (butuh ``python-dotenv``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # pragma: no cover - dependensi opsional
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

BRIDGE_DIR = Path(__file__).resolve().parent

if load_dotenv is not None:
    load_dotenv(dotenv_path=BRIDGE_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    """Nilai konfigurasi aktif untuk satu proses Bridge."""

    handlandmarker_url: str
    model_url: str

    bridge_host: str
    bridge_port: int

    # Panjang jendela rekaman yang dipakai untuk satu siklus prediksi (detik).
    window_seconds: float
    # Jumlah frame keypoint yang diminta ke service model.
    # Model dilatih dengan sequence 90 frame; nilai lain bisa membuat MaxPooling1D gagal.
    target_frames: int
    # Keyakinan minimum agar prediksi dianggap sah.
    prediction_threshold: float
    # Berapa frame minimum sebelum siklus prediksi dijalankan.
    min_frames: int
    # Berapa prediksi terakhir yang dipakai untuk majority vote.
    smoothing_window: int
    # FPS yang ditulis ke file video sementara.
    video_fps: float
    # Fourcc OpenCV untuk video sementara.
    video_fourcc: str
    # Timeout (detik) untuk panggilan ke service upstream.
    request_timeout: float
    # Berapa detik status health service upstream di-cache.
    health_cache_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            handlandmarker_url=_env_str(
                "HANDLANDMARKER_URL",
                "https://handlandmarker-service-1031455690132.asia-southeast2.run.app",
            ).rstrip("/"),
            model_url=_env_str(
                "MODEL_URL",
                "https://model-service-1031455690132.asia-southeast2.run.app",
            ).rstrip("/"),
            bridge_host=_env_str("BRIDGE_HOST", "0.0.0.0"),
            bridge_port=_env_int("BRIDGE_PORT", 8000),
            window_seconds=_env_float("WINDOW_SECONDS", 3.0),
            target_frames=_env_int("TARGET_FRAMES", 90),
            prediction_threshold=_env_float("PREDICTION_THRESHOLD", 0.7),
            min_frames=_env_int("MIN_FRAMES", 8),
            smoothing_window=_env_int("SMOOTHING_WINDOW", 5),
            video_fps=_env_float("VIDEO_FPS", 30.0),
            video_fourcc=_env_str("VIDEO_FOURCC", "mp4v"),
            request_timeout=_env_float("REQUEST_TIMEOUT", 180.0),
            health_cache_seconds=_env_float("HEALTH_CACHE_SECONDS", 10.0),
        )
