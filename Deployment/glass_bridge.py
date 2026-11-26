#!/usr/bin/env python3
"""Jembatan T-Glass <-> pipeline BISINDO dan speech-to-text.

Dua arah data:

1. Isyarat -> teks
   ``Deployment/realtime_detection_docker.py`` mengirim teks hasil translasi ke
   ``POST /api/sign``. T-Glass mengambil teks terbaru lewat ``GET /api/sign``.

2. Suara -> teks
   T-Glass mengirim rekaman PCM 16 kHz / 16-bit / mono ke ``POST /api/speech``.
   Server menjalankan SpeechRecognition (Google, default ``id-ID``) dan
   mengembalikan teks biasa di body response.

Contoh:
    python Deployment/glass_bridge.py
    python Deployment/glass_bridge.py --port 8765 --language id-ID

Catatan: server ini tanpa autentikasi, jadi jalankan hanya di jaringan lokal
yang tepercaya. Rekaman suara dikirim ke layanan Google Speech Recognition.
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import socket
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAX_AUDIO_BYTES = 2 * 1024 * 1024
_recognition_lock = threading.Lock()


class SpeechError(Exception):
    """Kesalahan yang aman ditampilkan ke T-Glass."""

    def __init__(self, message: str, status: int = 500) -> None:
        super().__init__(message)
        self.status = status


class TextStore:
    """Menyimpan teks isyarat dan hasil speech terakhir."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sign_text = ""
        self._sign_updated_at = 0.0
        self._muted_text: str | None = None
        self._speech_text = ""
        self._speech_updated_at = 0.0

    def set_sign(self, text: str) -> bool:
        with self._lock:
            if self._muted_text is not None:
                if text == self._muted_text:
                    # Deteksi mengirim ulang teks yang baru saja dihapus; abaikan
                    # sampai ada teks baru dari gestur berikutnya.
                    return False
                self._muted_text = None
            changed = text != self._sign_text
            self._sign_text = text
            self._sign_updated_at = time.time()
            return changed

    def clear_sign(self) -> None:
        with self._lock:
            self._muted_text = self._sign_text
            self._sign_text = ""
            self._sign_updated_at = time.time()

    def sign_text(self) -> str:
        with self._lock:
            return self._sign_text

    def set_speech(self, text: str) -> None:
        with self._lock:
            self._speech_text = text
            self._speech_updated_at = time.time()

    def speech_text(self) -> str:
        with self._lock:
            return self._speech_text

    def snapshot(self) -> tuple[str, float, str, float]:
        with self._lock:
            return (
                self._sign_text,
                self._sign_updated_at,
                self._speech_text,
                self._speech_updated_at,
            )


def transcribe_pcm(pcm: bytes, rate: int, width: int, language: str) -> str:
    """Ubah PCM mentah menjadi teks memakai speech_to_text.py."""
    try:
        import speech_recognition as sr
    except ImportError as exc:
        raise SpeechError(
            "SpeechRecognition belum terpasang (pip install -r requirements-speech.txt)",
            503,
        ) from exc

    try:
        from speech_to_text import recognize_speech
    except ImportError as exc:
        raise SpeechError(f"speech_to_text.py tidak ditemukan: {exc}", 500) from exc

    recognizer = sr.Recognizer()
    audio = sr.AudioData(pcm, rate, width)
    try:
        with _recognition_lock:
            return recognize_speech(recognizer, audio, language)
    except sr.UnknownValueError as exc:
        raise SpeechError("Suara tidak jelas", 422) from exc
    except sr.RequestError as exc:
        raise SpeechError(f"Layanan speech error: {exc}", 502) from exc


def render_page(store: TextStore) -> bytes:
    sign_text, sign_updated, speech_text, speech_updated = store.snapshot()

    def stamp(value: float) -> str:
        return time.strftime("%H:%M:%S", time.localtime(value)) if value else "-"

    page = f"""<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="1">
<title>T-Glass Bridge</title>
<style>
body {{ background:#101418; color:#e8eaed; font-family:system-ui,sans-serif; margin:24px; }}
h1 {{ font-size:18px; font-weight:600; }}
section {{ margin-bottom:20px; }}
small {{ color:#9aa0a6; }}
pre {{ background:#1b2026; border:1px solid #2b323a; border-radius:8px;
       padding:12px; min-height:48px; white-space:pre-wrap; font-size:16px; }}
</style>
</head>
<body>
<h1>T-Glass Bridge</h1>
<section>
  <h2>Isyarat &rarr; teks</h2>
  <p><small>diperbarui {stamp(sign_updated)}</small></p>
  <pre>{html.escape(sign_text) or "(kosong)"}</pre>
</section>
<section>
  <h2>Suara &rarr; teks</h2>
  <p><small>diperbarui {stamp(speech_updated)}</small></p>
  <pre>{html.escape(speech_text) or "(kosong)"}</pre>
</section>
</body>
</html>
"""
    return page.encode("utf-8")


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "TGlassBridge/1.0"

    @property
    def store(self) -> TextStore:
        return self.server.store  # type: ignore[attr-defined]

    @property
    def language(self) -> str:
        return self.server.language  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[bridge] {self.address_string()} {fmt % args}", flush=True)

    def log_request(self, code: Any = "-", size: Any = "-") -> None:
        if self.command == "GET" and self.path.startswith("/api/sign"):
            return
        super().log_request(code, size)

    def _send(self, status: int, body: bytes, content_type: str = "text/plain; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _send_text(self, status: int, text: str) -> None:
        self._send(status, text.encode("utf-8"))

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.close_connection = True
            length = 0
        if length <= 0:
            return b""
        if length > MAX_AUDIO_BYTES:
            self.close_connection = True
            raise SpeechError("Body terlalu besar", 413)
        return self.rfile.read(length)

    def do_GET(self) -> None:  # noqa: N802 (nama method HTTP)
        parsed = urlparse(self.path)
        if parsed.path == "/api/sign":
            self._send_text(200, self.store.sign_text())
            return
        if parsed.path == "/api/health":
            self._send_text(200, "ok")
            return
        if parsed.path in ("/", "/index.html"):
            self._send(200, render_page(self.store), "text/html; charset=utf-8")
            return
        self._send_text(404, "not found")

    def do_POST(self) -> None:  # noqa: N802 (nama method HTTP)
        parsed = urlparse(self.path)
        try:
            body = self._read_body()
            if parsed.path == "/api/sign":
                self._handle_sign(body)
            elif parsed.path == "/api/sign/clear":
                self.store.clear_sign()
                print("[SIGN] (dihapus)", flush=True)
                self._send_text(200, "ok")
            elif parsed.path == "/api/speech":
                self._handle_speech(parsed.query, body)
            else:
                self._send_text(404, "not found")
        except SpeechError as exc:
            self._send_text(exc.status, str(exc))
        except Exception:
            traceback.print_exc()
            self._send_text(500, "Gagal memproses permintaan")

    def _handle_sign(self, raw: bytes) -> None:
        text = ""
        if raw:
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                text = str(payload.get("text", ""))
            elif payload is None:
                text = raw.decode("utf-8", "replace")
            else:
                text = str(payload)
        text = text.strip()
        if self.store.set_sign(text):
            print(f"[SIGN] {text or '(kosong)'}", flush=True)
        self._send_text(200, "ok")

    def _handle_speech(self, query: str, pcm: bytes) -> None:
        params = parse_qs(query)
        try:
            rate = int(params.get("rate", ["16000"])[0])
            width = int(params.get("width", ["2"])[0])
        except (TypeError, ValueError):
            raise SpeechError("Parameter audio tidak valid", 400)
        if not 8000 <= rate <= 48000 or width not in (1, 2, 3, 4):
            raise SpeechError("Parameter audio di luar batas", 400)

        language = params.get("lang", [self.language])[0] or self.language
        if not pcm:
            raise SpeechError("Audio kosong", 400)
        if len(pcm) % width != 0:
            raise SpeechError("Panjang audio tidak sesuai lebar sample", 400)

        started = time.perf_counter()
        text = transcribe_pcm(pcm, rate, width, language)
        elapsed = (time.perf_counter() - started) * 1000.0
        text = " ".join(text.split())

        if text:
            self.store.set_speech(text)
            print(f"[SPEECH] {elapsed:.0f} ms, {len(pcm) // max(width, 1)} sample: {text}", flush=True)
        else:
            print(f"[SPEECH] {elapsed:.0f} ms: (kosong)", flush=True)
        self._send_text(200, text)


class BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], store: TextStore, language: str) -> None:
        super().__init__(address, BridgeHandler)
        self.store = store
        self.language = language


def lan_ipv4() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Jembatan T-Glass untuk BISINDO dan speech-to-text.")
    parser.add_argument("--host", default="0.0.0.0", help="Alamat bind (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8765, help="Port HTTP (default: 8765)")
    parser.add_argument("--language", default="id-ID", help="Bahasa speech recognition (default: id-ID)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = TextStore()
    server = BridgeServer((args.host, args.port), store, args.language)

    ip = lan_ipv4()
    print("=" * 62)
    print(" T-Glass bridge aktif")
    print(f" URL T-Glass : http://{ip}:{args.port}")
    print(f" Halaman      : http://{ip}:{args.port}/")
    print(f" Bahasa       : {args.language}")
    print(" Pastikan PC dan T-Glass berada di jaringan Wi-Fi yang sama,")
    print(" dan Windows Firewall mengizinkan Python masuk ke port ini.")
    print("=" * 62)
    if importlib.util.find_spec("speech_recognition") is None:
        print("[bridge] PERINGATAN: SpeechRecognition belum terpasang.")
        print("[bridge]   pip install -r requirements-speech.txt")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[bridge] dihentikan")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
