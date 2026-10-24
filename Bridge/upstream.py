"""Klien HTTP ke service handlandmarker dan model.

Dua hal penting yang ditangani modul ini:

1. **Bentuk payload ``/predict`` berbeda antar versi service.** Kode service yang
   ada (``Deployment/model/app.py``) mengharapkan ``{"landmarks": [...]}`` dan
   membalas ``{"label": ..., "confidence": ...}``. Versi lama (yang masih
   dipakai ``Deployment/realtime_detection_docker.py``) mengharapkan
   ``{"instances": [keypoints]}`` dan membalas ``{"predictions": [{...}]}``.
   Karena tidak diketahui service mana yang aktif, modul ini mencoba bentuk
   ``landmarks`` lebih dulu lalu jatuh ke ``instances`` bila ditolak.

2. **Skala confidence tidak seragam.** Service yang aktif mengembalikan persen
   (``99.19``), sedangkan logika lama memperlakukannya sebagai ``0..1``.
   Semua nilai dinormalisasi ke ``0..1`` sebelum keluar dari modul ini.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


class UpstreamError(RuntimeError):
    """Panggilan ke service upstream gagal."""


@dataclass(frozen=True)
class Prediction:
    """Hasil prediksi yang sudah dinormalisasi."""

    label: str
    confidence: float  # selalu 0..1


def _normalize_confidence(value: Any) -> float:
    """Ubah confidence apa pun menjadi rentang 0..1.

    Service yang aktif mengembalikan persen (``round(conf * 100, 2)``), sedangkan
    service lama mengembalikan pecahan ``0..1``. Ambang keyakinan di
    ``realtime_detection.py`` memakai satuan ``0..1``, jadi normalisasi di sini
    mencegah perbandingan yang diam-diam selalu benar.
    """
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0

    if confidence > 1.0:
        # Persen (mis. 99.19) -> pecahan.
        confidence /= 100.0

    if confidence < 0.0:
        return 0.0
    if confidence > 1.0:
        return 1.0
    return confidence


def _parse_prediction(payload: Any) -> Prediction:
    """Baca label & confidence dari berbagai bentuk response."""
    if not isinstance(payload, dict):
        raise UpstreamError("Respons /predict bukan objek JSON.")

    # Bentuk service aktif: {"status", "label", "confidence", "input_shape"}
    if "label" in payload:
        return Prediction(
            label=str(payload.get("label", "-")),
            confidence=_normalize_confidence(payload.get("confidence", 0.0)),
        )

    # Bentuk service lama: {"predictions": [{"label", "confidence"}, ...]}
    predictions = payload.get("predictions")
    if isinstance(predictions, list) and predictions:
        first = predictions[0]
        if isinstance(first, dict):
            return Prediction(
                label=str(first.get("label", "-")),
                confidence=_normalize_confidence(first.get("confidence", 0.0)),
            )

    raise UpstreamError(
        "Tidak menemukan 'label' maupun 'predictions' pada respons /predict: "
        f"{sorted(payload)[:8]}"
    )


class UpstreamClient:
    """Menjalankan siklus ekstraksi keypoint + prediksi label."""

    def __init__(
        self,
        handlandmarker_url: str,
        model_url: str,
        timeout: float = 180.0,
        session: requests.Session | None = None,
    ) -> None:
        self.handlandmarker_url = handlandmarker_url.rstrip("/")
        self.model_url = model_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()

    # ------------------------------------------------------------------ #
    # Handlandmarker
    # ------------------------------------------------------------------ #
    def extract_keypoints(self, video_path: Path) -> list[list[float]]:
        """Kirim file video ke ``/extract`` dan kembalikan keypoint per frame.

        Field multipart harus bernama ``file`` (lihat
        ``Deployment/handlandmarker/app.py``).
        """
        url = f"{self.handlandmarker_url}/extract"
        try:
            with video_path.open("rb") as handle:
                response = self.session.post(
                    url,
                    files={"file": (video_path.name, handle, "video/mp4")},
                    timeout=self.timeout,
                )
        except requests.RequestException as exc:
            raise UpstreamError(f"Gagal menghubungi handlandmarker: {exc}") from exc

        if response.status_code != 200:
            raise UpstreamError(
                f"Handlandmarker /extract menjawab HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise UpstreamError("Respons handlandmarker bukan JSON valid.") from exc

        keypoints = payload.get("keypoints")
        if not isinstance(keypoints, list) or not keypoints:
            raise UpstreamError("Handlandmarker tidak mengembalikan keypoint.")

        return keypoints

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    def predict(self, keypoints: list[list[float]]) -> Prediction:
        """Kirim keypoint ke ``/predict`` dan kembalikan prediksi ternormalisasi.

        Bentuk ``{"landmarks": ...}`` dicoba lebih dulu; bila service menolaknya
        dengan HTTP 422 (Pydantic), bentuk ``{"instances": [keypoints]}`` dicoba
        sebagai cadangan.
        """
        attempts = (
            {"landmarks": keypoints},
            {"instances": [keypoints]},
        )

        last_error: str | None = None
        for payload in attempts:
            url = f"{self.model_url}/predict"
            try:
                response = self.session.post(url, json=payload, timeout=self.timeout)
            except requests.RequestException as exc:
                raise UpstreamError(f"Gagal menghubungi model service: {exc}") from exc

            if response.status_code == 200:
                try:
                    return _parse_prediction(response.json())
                except UpstreamError as exc:
                    last_error = str(exc)
                    continue

            if response.status_code == 422:
                # Bentuk payload ditolak -> coba bentuk berikutnya.
                last_error = f"HTTP 422: {response.text[:200]}"
                continue

            raise UpstreamError(
                f"Model /predict menjawab HTTP {response.status_code}: {response.text[:200]}"
            )

        raise UpstreamError(
            "Semua bentuk payload /predict ditolak. "
            f"Error terakhir: {last_error or 'tidak diketahui'}"
        )

    # ------------------------------------------------------------------ #
    # Health
    # ------------------------------------------------------------------ #
    def check_health(self) -> dict[str, dict[str, Any]]:
        """Periksa apakah kedua service hidup. Tidak melempar exception."""
        return {
            "handlandmarker": self._check_one(
                f"{self.handlandmarker_url}/health", "GET"
            ),
            "model": self._check_one(f"{self.model_url}/", "GET"),
        }

    def _check_one(self, url: str, method: str) -> dict[str, Any]:
        try:
            response = self.session.request(method, url, timeout=10)
        except requests.RequestException as exc:
            return {"url": url, "ok": False, "status": None, "detail": str(exc)[:200]}

        ok = response.status_code == 200
        detail = "ok" if ok else f"HTTP {response.status_code}"
        if not ok:
            # Cloud Run menjawab 404 dengan halaman HTML; ringkas saja.
            detail = f"HTTP {response.status_code} (service tidak aktif atau URL salah)"
        return {"url": url, "ok": ok, "status": response.status_code, "detail": detail}
