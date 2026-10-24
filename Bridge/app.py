"""Aplikasi FastAPI Bridge BISINDO.

Endpoint:

- ``POST /ingest``      : terima satu frame JPEG dari ESP32-CAM (body mentah).
- ``GET  /stream``      : pratinjau MJPEG untuk web app.
- ``GET  /api/state``   : status prediksi terakhir (JSON).
- ``GET  /api/health``  : status service upstream + buffer.
- ``GET  /``            : web app (file ``Bridge/ui/index.html``).
- ``POST /api/predict`` : paksa satu siklus prediksi (tombol "tebak sekarang").
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .config import BRIDGE_DIR, Settings
from .pipeline import PredictionPipeline
from .upstream import UpstreamClient, UpstreamError

UI_PATH = BRIDGE_DIR / "ui" / "index.html"

# Batas ukuran frame JPEG yang diterima (VGA ~40KB, dilebihkan untuk keamanan).
MAX_FRAME_BYTES = 2 * 1024 * 1024

settings = Settings.from_env()
client = UpstreamClient(
    handlandmarker_url=settings.handlandmarker_url,
    model_url=settings.model_url,
    timeout=settings.request_timeout,
)
pipeline = PredictionPipeline(settings=settings, client=client)

app = FastAPI(title="Bridge BISINDO", version="1.0.0")

# Event yang di-set setiap ada frame baru, dipakai oleh MJPEG streamer.
_frame_event = threading.Event()

# Thread background yang menjalankan siklus prediksi terus-menerus.
_worker_stop = threading.Event()


def _prediction_loop() -> None:
    """Jalankan siklus prediksi berkala di thread terpisah.

    Satu inferensi berjalan pada satu waktu (gaya ``realtime_detection.py``),
    jadi request ke service model tidak menumpuk.
    """
    while not _worker_stop.is_set():
        try:
            if not pipeline.try_acquire():
                time.sleep(0.3)
                continue

            try:
                pipeline.predict_once()
            except UpstreamError as exc:
                pipeline.set_error(str(exc))
            except Exception as exc:  # noqa: BLE001 - worker tidak boleh mati
                pipeline.set_error(f"Kesalahan tak terduga: {exc}")
            finally:
                pipeline.release()
        except Exception:  # noqa: BLE001
            pass

        # Jeda pendek; siklus yang butuh ~6 detik membuat jeda ini tidak berarti.
        _worker_stop.wait(0.5)


@app.on_event("startup")
def _startup() -> None:
    thread = threading.Thread(target=_prediction_loop, name="prediction-worker", daemon=True)
    thread.start()


@app.on_event("shutdown")
def _shutdown() -> None:
    _worker_stop.set()


# --------------------------------------------------------------------------- #
# Ingest
# --------------------------------------------------------------------------- #
@app.post("/ingest")
async def ingest(request: Request) -> Response:
    """Terima satu frame JPEG dari ESP32-CAM.

    Dua bentuk yang didukung:

    - body mentah dengan header ``Content-Type: image/jpeg`` (dipakai firmware).
    - multipart/form-data dengan field ``file`` (dipakai skrip pengujian).
    """
    content_type = request.headers.get("content-type", "")
    data = b""

    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file") or form.get("frame")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse(
                {"status": "error", "detail": "Field multipart 'file' tidak ditemukan."},
                status_code=400,
            )
        data = await upload.read()
    else:
        data = await request.body()

    if not data:
        return JSONResponse(
            {"status": "error", "detail": "Body kosong."}, status_code=400
        )

    if len(data) > MAX_FRAME_BYTES:
        return JSONResponse(
            {
                "status": "error",
                "detail": f"Frame terlalu besar ({len(data)} byte).",
            },
            status_code=413,
        )

    if not data.startswith(b"\xff\xd8"):
        return JSONResponse(
            {"status": "error", "detail": "Body bukan JPEG (tanda SOI tidak ditemukan)."},
            status_code=415,
        )

    pipeline.buffer.add(data)
    _frame_event.set()

    return {
        "status": "ok",
        "bytes": len(data),
        "buffered": len(pipeline.buffer.snapshot()),
    }


# --------------------------------------------------------------------------- #
# Stream
# --------------------------------------------------------------------------- #
def _mjpeg_stream():
    """Hasilkan frame JPEG berurutan sebagai multipart/x-mixed-replace."""
    boundary = "frame"
    while True:
        sample = pipeline.buffer.latest()
        if sample is None:
            time.sleep(0.1)
            continue

        yield (
            b"--" + boundary.encode() + b"\r\n"
            b"Content-Type: image/jpeg\r\n"
            b"Content-Length: " + str(len(sample.data)).encode() + b"\r\n\r\n"
            + sample.data + b"\r\n"
        )
        _frame_event.wait(0.08)
        _frame_event.clear()


@app.get("/stream")
async def stream() -> StreamingResponse:
    return StreamingResponse(
        _mjpeg_stream(),
        media_type=f"multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store", "Pragma": "no-cache"},
    )


# --------------------------------------------------------------------------- #
# State & health
# --------------------------------------------------------------------------- #
@app.get("/api/state")
async def state() -> dict[str, Any]:
    prediction = pipeline.snapshot_state()
    prediction["buffer"] = pipeline.buffer.stats()
    prediction["target_frames"] = settings.target_frames
    prediction["threshold"] = settings.prediction_threshold
    return prediction


@app.get("/api/health")
async def health() -> dict[str, Any]:
    upstream = await asyncio.to_thread(pipeline.health)
    return {
        "bridge": {
            "ok": True,
            "handlandmarker_url": settings.handlandmarker_url,
            "model_url": settings.model_url,
            "window_seconds": settings.window_seconds,
            "target_frames": settings.target_frames,
        },
        "upstream": upstream,
        "buffer": pipeline.buffer.stats(),
    }


@app.post("/api/predict")
async def force_predict() -> Response:
    """Paksa satu siklus prediksi di luar worker berkala."""
    if not pipeline.try_acquire():
        return JSONResponse(
            {"status": "busy", "detail": "Siklus prediksi sedang berjalan."},
            status_code=409,
        )
    try:
        result = await asyncio.to_thread(pipeline.predict_once)
        return {"status": "ok", "result": result}
    except UpstreamError as exc:
        pipeline.set_error(str(exc))
        return JSONResponse({"status": "error", "detail": str(exc)}, status_code=502)
    finally:
        pipeline.release()


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
@app.get("/")
async def index() -> Response:
    if not UI_PATH.exists():
        return JSONResponse(
            {"status": "error", "detail": f"UI belum ada di {UI_PATH}"},
            status_code=500,
        )
    return FileResponse(UI_PATH, media_type="text/html")


def main() -> None:
    """Jalankan server development."""
    import uvicorn

    uvicorn.run(
        "Bridge.app:app",
        host=settings.bridge_host,
        port=settings.bridge_port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
