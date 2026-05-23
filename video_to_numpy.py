#!/usr/bin/env python3
"""
Convert video files into NumPy keypoint arrays using MediaPipe HandLandmarker.

Each frame becomes a 126-value vector:
2 hands x 21 landmarks x 3 coordinates (x, y, z).

Usage examples:
    python video_to_numpy.py --input recording/video1.mp4 --output-dir data/numpy
    python video_to_numpy.py --input-dir recording --output-dir data/numpy --max-frames 60

This script depends on:
  pip install opencv-python numpy mediapipe
"""
from pathlib import Path
import argparse
import os
import sys

import cv2
import mediapipe as mp
import numpy as np


os.environ["EGL_PLATFORM"] = "surfaceless"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def extract_keypoints(results):
    if results and getattr(results, "hand_landmarks", None):
        landmarks_list = []
        for hand in results.hand_landmarks[:2]:
            for landmark in hand:
                landmarks_list.extend([landmark.x, landmark.y, landmark.z])

        arr = np.array(landmarks_list, dtype=np.float32)
        if arr.shape[0] < 126:
            arr = np.concatenate([arr, np.zeros(126 - arr.shape[0], dtype=np.float32)])
        return arr[:126]

    return np.zeros(126, dtype=np.float32)


def convert_video_to_numpy(video_path: Path, out_path: Path, *, frame_step=1, max_frames=60, normalize=False):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(
            model_asset_path='handlandmarker/hand_landmarker.task',
            delegate=BaseOptions.Delegate.GPU,
        ),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 90.0

    frames = []
    idx = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if idx % frame_step != 0:
                idx += 1
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            frame_timestamp_ms = int((idx / fps) * 1000)
            results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)
            keypoints = extract_keypoints(results)

            frames.append(keypoints)
            idx += 1

            if max_frames is not None and len(frames) >= max_frames:
                break

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")

    arr = np.stack(frames).astype(np.float32)
    if normalize:
        arr = arr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path.with_suffix('.npy')), arr)
    return out_path.with_suffix('.npy')


def find_videos_in_dir(input_dir: Path):
    exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    for p in sorted(input_dir.rglob('*')):
        if p.suffix.lower() in exts:
            yield p


def main(argv=sys.argv[1:]):
    parser = argparse.ArgumentParser(description='Convert video files to NumPy keypoint arrays using MediaPipe HandLandmarker')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=Path, help='Path to a single video file')
    group.add_argument('--input-dir', type=Path, help='Directory to scan for video files')
    parser.add_argument('--output-dir', '-o', type=Path, required=True, help='Directory to write NumPy outputs')
    parser.add_argument('--frame-step', type=int, default=1, help='Take one frame every N frames (default 1 = all)')
    parser.add_argument('--max-frames', type=int, default=60, help='Stop after saving this many frames (default: 60)')
    parser.add_argument('--normalize', action='store_true', help='Kept for compatibility; keypoints are already coordinates')
    parser.add_argument('--prefix', type=str, default='', help='Optional prefix for output filenames')

    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = [args.input] if args.input else list(find_videos_in_dir(args.input_dir))

    if not videos:
        print('No video files found.', file=sys.stderr)
        return 1

    for v in videos:
        try:
            name = args.prefix + v.stem
            out_file = output_dir / name
            saved = convert_video_to_numpy(v, out_file, frame_step=args.frame_step, max_frames=args.max_frames, normalize=args.normalize)
            print(f'Saved {saved} (from {v})')
        except Exception as e:
            print(f'Error processing {v}: {e}', file=sys.stderr)


if __name__ == '__main__':
    raise SystemExit(main())
