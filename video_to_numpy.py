#!/usr/bin/env python3
"""
Convert video files (e.g., camera recordings) into NumPy arrays and save them.

Usage examples:
    python video_to_numpy.py --input recordings/video1.mp4 --output-dir data/numpy
    python video_to_numpy.py --input-dir recordings --output-dir data/numpy --resize 224 224 --grayscale

This script depends on OpenCV and NumPy:
  pip install opencv-python numpy
"""
from pathlib import Path
import argparse
import cv2
import numpy as np
import sys


def convert_video_to_numpy(video_path: Path, out_path: Path, *, resize=None, grayscale=False, frame_step=1, max_frames=60, normalize=False, compress=True):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []
    idx = 0
    # Baca frame satu per satu dari video.
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Kalau ingin sampling, lewati frame tertentu sesuai frame_step.
        if idx % frame_step != 0:
            idx += 1
            continue

        # Ubah warna frame ke grayscale atau RGB.
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Kalau perlu, samakan ukuran frame sebelum disimpan.
        if resize:
            frame = cv2.resize(frame, (resize[0], resize[1]), interpolation=cv2.INTER_AREA)

        frames.append(frame)
        idx += 1

        # Jika max_frames diisi, hentikan setelah jumlah frame yang diminta tercapai.
        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from {video_path}")

    arr = np.stack(frames)
    if normalize:
        arr = arr.astype(np.float32) / 255.0

    # Simpan hasil akhir ke folder output.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        np.savez_compressed(str(out_path.with_suffix('.npz')), frames=arr)
        return out_path.with_suffix('.npz')

    np.save(str(out_path.with_suffix('.npy')), arr)
    return out_path.with_suffix('.npy')


def find_videos_in_dir(input_dir: Path):
    exts = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}
    for p in sorted(input_dir.rglob('*')):
        if p.suffix.lower() in exts:
            yield p


def main(argv=sys.argv[1:]):
    p = argparse.ArgumentParser(description='Convert video files to NumPy arrays')
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--input', '-i', type=Path, help='Path to a single video file')
    group.add_argument('--input-dir', type=Path, help='Directory to scan for video files')
    p.add_argument('--output-dir', '-o', type=Path, required=True, help='Directory to write NumPy outputs')
    p.add_argument('--resize', nargs=2, type=int, metavar=('W', 'H'), help='Resize frames to W H')
    p.add_argument('--grayscale', action='store_true', help='Convert frames to grayscale')
    p.add_argument('--frame-step', type=int, default=1, help='Take one frame every N frames (default 1 = all)')
    p.add_argument('--max-frames', type=int, default=60, help='Stop after saving this many frames (default: 60)')
    p.add_argument('--compress', action='store_true', help='Save as compressed .npz instead of .npy')
    p.add_argument('--normalize', action='store_true', help='Convert uint8 frames to float32 in [0,1]')
    p.add_argument('--prefix', type=str, default='', help='Optional prefix for output filenames')

    args = p.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pilih satu video atau semua video di folder input.
    videos = []
    if args.input:
        videos = [args.input]
    else:
        videos = list(find_videos_in_dir(args.input_dir))

    if not videos:
        print('No video files found.', file=sys.stderr)
        return 1

    for v in videos:
        try:
            # Nama file output mengikuti nama video input.
            name = args.prefix + v.stem
            out_file = output_dir / name
            saved = convert_video_to_numpy(v, out_file, resize=(args.resize[0], args.resize[1]) if args.resize else None,
                                           grayscale=args.grayscale, frame_step=args.frame_step, max_frames=args.max_frames,
                                           normalize=args.normalize, compress=args.compress)
            print(f'Saved {saved} (from {v})')
        except Exception as e:
            print(f'Error processing {v}: {e}', file=sys.stderr)


if __name__ == '__main__':
    raise SystemExit(main())
