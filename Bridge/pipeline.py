"""Orkestrasi pipeline BISINDO: frame JPEG -> video -> keypoint -> prediksi.

Alur satu siklus prediksi:

1. Ambil frame dari buffer yang masuk dalam ``window_seconds`` terakhir.
2. Tulis frame itu menjadi file video sementara (mp4v).
3. Upload video ke ``/extract`` -> keypoint per frame.
4. Resample keypoint menjadi tepat ``target_frames`` baris.
5. Kirim ke ``/predict`` -> label + keyakinan.
6. Terapkan ambang keyakinan dan majority vote.

Catatan penting mengenai urutan waktu: frame **tidak boleh** di-flip horizontal
sebelum landmark diekstrak. Dataset training direkam tanpa flip (lihat
``realtime_detection.py``), sehingga landmark harus dihitung dari frame asli.
Bridge tidak pernah melakukan flip sama sekali.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np

from .config import Settings
from .upstream import Prediction, UpstreamClient, UpstreamError

# Target frame untuk model. Model dilatih dengan sequence 90 frame; mengirimi
# kurang dari 4 frame membuat MaxPooling1D gagal (subtracting 2 from 1).
MIN_USABLE_FRAMES = 4


@dataclass
class FrameSample:
    """Satu frame JPEG yang diterima dari ESP32."""

    data: bytes
    received_at: float
    width: int = 0
    height: int = 0


@dataclass
class PredictionState:
    """Status prediksi terakhir yang dilihat web app."""

    label: str = "-"
    confidence: float = 0.0
    raw_label: str = "-"
    raw_confidence: float = 0.0
    updated_at: float | None = None
    source_frames: int = 0
    latency_ms: int | None = None
    error: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "raw_label": self.raw_label,
            "raw_confidence": round(self.raw_confidence, 4),
            "updated_at": self.updated_at,
            "source_frames": self.source_frames,
            "latency_ms": self.latency_ms,
            "error": self.error,
            "history": self.history[-20:],
        }


def resample_keypoints(
    keypoints: list[list[float]], target_frames: int
) -> list[list[float]]:
    """Ubah jumlah frame keypoint menjadi tepat ``target_frames``.

    Distribusi training memakai video 60 frame yang di-pad ke 90; dan kecepatan
    stream ESP32 tidak akan persis 30fps. Resample linier (mengulang frame
    terdekat) terbukti menjaga label tetap sama pada clip yang diuji.
    """
    array = np.asarray(keypoints, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0:
        raise UpstreamError("Keypoint dari handlandmarker tidak berbentuk matriks.")

    current = array.shape[0]
    if current == target_frames:
        return array.tolist()

    if current < MIN_USABLE_FRAMES:
        raise UpstreamError(
            f"Hanya {current} frame keypoint; minimal {MIN_USABLE_FRAMES} agar model bisa berjalan."
        )

    indices = np.linspace(0, current - 1, target_frames)
    lower = np.floor(indices).astype(np.int64)
    upper = np.ceil(indices).astype(np.int64)
    weight = (indices - lower).astype(np.float32)[:, None]

    blended = array[lower] * (1.0 - weight) + array[upper] * weight
    return blended.astype(np.float32).tolist()


class FrameBuffer:
    """Buffer frame JPEG berbasis waktu dengan lock."""

    def __init__(self, max_age_seconds: float) -> None:
        self._max_age = max_age_seconds
        self._items: deque[FrameSample] = deque()
        self._lock = threading.Lock()
        self._total_received = 0
        self._last_received_at: float | None = None

    def add(self, data: bytes, width: int = 0, height: int = 0) -> None:
        now = time.monotonic()
        with self._lock:
            self._items.append(FrameSample(data, now, width, height))
            self._total_received += 1
            self._last_received_at = now
            self._prune(now)

    def _prune(self, now: float) -> None:
        cutoff = now - self._max_age
        while self._items and self._items[0].received_at < cutoff:
            self._items.popleft()

    def snapshot(self) -> list[FrameSample]:
        """Kembalikan frame yang masuk dalam jendela waktu terakhir."""
        now = time.monotonic()
        with self._lock:
            self._prune(now)
            return list(self._items)

    def latest(self) -> FrameSample | None:
        with self._lock:
            return self._items[-1] if self._items else None

    def stats(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            self._prune(now)
            buffered = len(self._items)
            last = self._last_received_at
            total = self._total_received
            first = self._items[0].received_at if self._items else None

        age = None if last is None else round(now - last, 2)
        fps = None
        if buffered > 1 and first is not None and last is not None:
            span = last - first
            if span > 0:
                fps = round((buffered - 1) / span, 2)
        return {
            "buffered": buffered,
            "total_received": total,
            "seconds_since_last_frame": age,
            "estimated_fps": fps,
            "connected": age is not None and age < 3.0,
        }


def write_video(
    frames: list[FrameSample], path: Path, fps: float, fourcc: str
) -> bool:
    """Tulis frame JPEG menjadi file video.

    Frame di-decode satu per satu (tidak semua sekaligus) supaya pemakaian RAM
    tetap rendah: 90 frame 640x480x3 byte butuh ~83MB kalau sekaligus.
    """
    if not frames:
        return False

    writer: cv2.VideoWriter | None = None
    written = 0
    try:
        for sample in frames:
            buffer = np.frombuffer(sample.data, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if image is None:
                continue

            if writer is None:
                height, width = image.shape[:2]
                writer = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*fourcc),
                    fps,
                    (width, height),
                )
                if not writer.isOpened():
                    return False

            writer.write(image)
            written += 1
    finally:
        if writer is not None:
            writer.release()

    return written > 0


class PredictionPipeline:
    """Menjalankan siklus prediksi di thread terpisah."""

    def __init__(self, settings: Settings, client: UpstreamClient) -> None:
        self.settings = settings
        self.client = client
        self.buffer = FrameBuffer(max_age_seconds=settings.window_seconds * 3)
        self.state = PredictionState()
        self._history: deque[str] = deque(maxlen=settings.smoothing_window)
        self._state_lock = threading.Lock()
        self._in_flight = False
        self._busy_lock = threading.Lock()
        self._health_cache: dict[str, Any] = {}
        self._health_checked_at = 0.0

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    def health(self) -> dict[str, Any]:
        """Status service upstream, di-cache agar tidak membebani Cloud Run."""
        now = time.monotonic()
        with self._busy_lock:
            cached = self._health_cache
            checked_at = self._health_checked_at
            if cached and (now - checked_at) < self.settings.health_cache_seconds:
                return cached

        result = self.client.check_health()
        with self._busy_lock:
            self._health_cache = result
            self._health_checked_at = time.monotonic()
        return result

    # ------------------------------------------------------------------ #
    # Prediksi
    # ------------------------------------------------------------------ #
    def predict_once(self) -> dict[str, Any]:
        """Jalankan satu siklus prediksi penuh."""
        started = time.monotonic()
        frames = self.buffer.snapshot()
        if len(frames) < self.settings.min_frames:
            raise UpstreamError(
                f"Frame belum cukup ({len(frames)}/{self.settings.min_frames})."
            )

        video_dir = TemporaryDirectory(prefix="bridge_clip_")
        video_path = Path(video_dir.name) / "clip.mp4"
        try:
            if not write_video(
                frames, video_path, self.settings.video_fps, self.settings.video_fourcc
            ):
                raise UpstreamError("Gagal menulis video sementara dari frame JPEG.")

            keypoints = self.client.extract_keypoints(video_path)
            resampled = resample_keypoints(keypoints, self.settings.target_frames)
            prediction = self.client.predict(resampled)
        finally:
            video_dir.cleanup()

        smoothed = self._smooth(prediction)
        latency_ms = int((time.monotonic() - started) * 1000)

        with self._state_lock:
            self.state.raw_label = prediction.label
            self.state.raw_confidence = prediction.confidence
            self.state.label = smoothed.label
            self.state.confidence = smoothed.confidence
            self.state.updated_at = time.time()
            self.state.source_frames = len(frames)
            self.state.latency_ms = latency_ms
            self.state.error = None
            if smoothed.label and smoothed.label != "-":
                self.state.history.insert(
                    0,
                    {
                        "label": smoothed.label,
                        "confidence": round(smoothed.confidence, 4),
                        "at": self.state.updated_at,
                    },
                )
                del self.state.history[20:]

        return {
            "label": smoothed.label,
            "confidence": smoothed.confidence,
            "raw_label": prediction.label,
            "raw_confidence": prediction.confidence,
            "source_frames": len(frames),
            "latency_ms": latency_ms,
        }

    def _smooth(self, prediction: Prediction) -> Prediction:
        """Ambang keyakinan + majority vote atas beberapa prediksi terakhir.

        Meniru ``realtime_detection.py`` yang sudah dipakai proyek: prediksi di
        bawah ambang dibuang (label ``"-"``), lalu label terbanyak dari riwayat
        terakhir yang dipakai sebagai hasil akhir.
        """
        if prediction.confidence < self.settings.prediction_threshold:
            candidate = "-"
        else:
            candidate = prediction.label

        with self._state_lock:
            self._history.append(candidate)
            if len(self._history) == self._history.maxlen:
                counts: dict[str, int] = {}
                for item in self._history:
                    counts[item] = counts.get(item, 0) + 1
                winner = max(counts, key=counts.get)
            else:
                winner = candidate

        if winner == "-":
            return Prediction(label="-", confidence=0.0)
        return Prediction(label=winner, confidence=prediction.confidence)

    def try_acquire(self) -> bool:
        """Ambil lock siklus; ``False`` bila siklus lain masih berjalan."""
        with self._busy_lock:
            if self._in_flight:
                return False
            self._in_flight = True
            return True

    def release(self) -> None:
        with self._busy_lock:
            self._in_flight = False

    def set_error(self, message: str) -> None:
        with self._state_lock:
            self.state.error = message

    def snapshot_state(self) -> dict[str, Any]:
        """Salinan status prediksi terakhir (aman dibaca dari thread lain)."""
        with self._state_lock:
            return self.state.to_dict()
