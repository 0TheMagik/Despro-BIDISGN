#!/usr/bin/env python3
"""Prototype speech-to-text program

Menggunakan pustaka SpeechRecognition untuk mengubah ucapan menjadi teks. 
Program ini mendengarkan ucapan dari mikrofon dan menampilkan hasil transkripsi di layar. 
Opsi tambahan memungkinkan pengguna untuk menyimpan transkrip ke file teks.

masih online menggunakan google service, sehingga memerlukan koneksi internet untuk berfungsi.

untuk versi offline, bisa menggunakan model vosk atau whisper, namun perlu diunduh terlebih dahulu.
Examples:
    python speech_to_text.py
    python speech_to_text.py --language en-US --output transcript.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

def recognize_speech(recognizer, audio, language: str) -> str:
    """Convert one recorded phrase to text using the configured language."""
    return recognizer.recognize_google(audio, language=language).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Listen to microphone speech and convert it to text."
    )
    parser.add_argument(
        "--language",
        default="id-ID",
        help="Speech recognition language, for example id-ID or en-US (default: id-ID)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional UTF-8 file where recognized phrases are appended",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        help="Microphone device index; omit this to use the system default",
    )
    parser.add_argument(
        "--energy-threshold",
        type=int,
        default=300,
        help="Starting microphone energy threshold (default: 300)",
    )
    parser.add_argument(
        "--pause-threshold",
        type=float,
        default=0.8,
        help="Seconds of silence that end a phrase (default: 0.8)",
    )
    return parser


def write_transcript(output_path: Path | None, text: str) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as transcript:
        transcript.write(text + "\n")


def run(args: argparse.Namespace) -> int:
    try:
        import speech_recognition as sr
    except ImportError as exc:
        print(
            "SpeechRecognition is not installed. Run: "
            "python -m pip install -r requirements-speech.txt",
            file=sys.stderr,
        )
        return 1

    recognizer = sr.Recognizer()
    recognizer.energy_threshold = args.energy_threshold
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = args.pause_threshold

    try:
        microphone = sr.Microphone(device_index=args.device_index)
    except (OSError, AttributeError) as exc:
        print(
            "Could not open a microphone. Check that it is connected and that "
            "PyAudio is installed.",
            file=sys.stderr,
        )
        print(f"Details: {exc}", file=sys.stderr)
        return 1

    print(f"Speech-to-text prototype ready ({args.language}).")
    print("Calibrating microphone for ambient noise...")
    print("Speak normally. Press Ctrl+C to stop.\n")

    try:
        with microphone as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)
            while True:
                print("Listening...", flush=True)
                try:
                    audio = recognizer.listen(source, timeout=5, phrase_time_limit=12)
                except sr.WaitTimeoutError:
                    print("No speech detected; listening again.")
                    continue

                try:
                    text = recognize_speech(recognizer, audio, args.language)
                except sr.UnknownValueError:
                    print("Speech was not clear enough to recognize.")
                    continue
                except sr.RequestError as exc:
                    print(f"Speech recognition service error: {exc}", file=sys.stderr)
                    return 1

                if text:
                    print(f"Text: {text}")
                    write_transcript(args.output, text)
    except KeyboardInterrupt:
        print("\nSpeech-to-text stopped.")

    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())