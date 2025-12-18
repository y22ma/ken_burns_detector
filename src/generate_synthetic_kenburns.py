#!/usr/bin/env python3
"""
Generate synthetic positive Ken Burns clips from still frames.

Approach:
  - Pick a random source video from dataset/
  - Grab a single random frame (treat as a still image)
  - Render a new clip by applying a smooth similarity transform over time:
      uniform scale + translation (x,y) + in-plane rotation (z-axis)
  - Save as MP4 + write labels CSV (all True)

This is useful to increase positive examples (Ken Burns) for training.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np


def ease_in_out_cos(t: float) -> float:
    """Smooth 0->1 easing."""
    return 0.5 - 0.5 * math.cos(math.pi * t)


def sample_random_frame(video_path: Path, rng: random.Random) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            n = 1
        idx = rng.randrange(0, n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            # Fallback: try first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read a frame from {video_path}")
        return frame
    finally:
        cap.release()


def resize_to_width(frame_bgr: np.ndarray, target_width: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if w == target_width:
        return frame_bgr
    scale = target_width / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (target_width, new_h), interpolation=cv2.INTER_AREA)


def render_kenburns_clip(
    still_bgr: np.ndarray,
    num_frames: int,
    scale_start: float,
    scale_end: float,
    rot_start_deg: float,
    rot_end_deg: float,
    tx_start: float,
    tx_end: float,
    ty_start: float,
    ty_end: float,
    border_mode: int = cv2.BORDER_REFLECT101,
) -> Tuple[np.ndarray, Dict]:
    """
    Render a Ken Burns clip from a still.

    tx/ty are in pixels in the output image coordinate system.
    """
    # Render on a padded canvas and then center-crop back to (w, h).
    # This avoids black patches when the transform maps out-of-bounds.
    h, w = still_bgr.shape[:2]
    max_trans = max(abs(tx_start), abs(tx_end), abs(ty_start), abs(ty_end))
    # Heuristic padding: enough for translation + some slack for rotation/scale.
    pad = int(max(8.0, 0.30 * max(h, w) + max_trans))

    padded = cv2.copyMakeBorder(
        still_bgr,
        pad,
        pad,
        pad,
        pad,
        borderType=border_mode,
    )
    hp, wp = padded.shape[:2]
    center = (wp / 2.0, hp / 2.0)

    frames = []
    for i in range(num_frames):
        t = 0.0 if num_frames <= 1 else i / (num_frames - 1)
        e = ease_in_out_cos(t)
        s = scale_start + e * (scale_end - scale_start)
        ang = rot_start_deg + e * (rot_end_deg - rot_start_deg)
        tx = tx_start + e * (tx_end - tx_start)
        ty = ty_start + e * (ty_end - ty_start)

        M = cv2.getRotationMatrix2D(center, ang, s)  # 2x3
        M[0, 2] += tx
        M[1, 2] += ty

        out_padded = cv2.warpAffine(
            padded,
            M,
            (wp, hp),
            flags=cv2.INTER_LINEAR,
            borderMode=border_mode,
        )
        # Center crop back to original size
        out = out_padded[pad : pad + h, pad : pad + w].copy()
        frames.append(out)

    meta = {
        "w": w,
        "h": h,
        "num_frames": num_frames,
        "pad": pad,
        "scale_start": scale_start,
        "scale_end": scale_end,
        "rot_start_deg": rot_start_deg,
        "rot_end_deg": rot_end_deg,
        "tx_start": tx_start,
        "tx_end": tx_end,
        "ty_start": ty_start,
        "ty_end": ty_end,
    }
    return np.stack(frames, axis=0), meta


def next_index(out_dir: Path, prefix: str) -> int:
    existing = sorted(out_dir.glob(f"{prefix}_*.mp4"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        stem = p.stem
        parts = stem.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            nums.append(int(parts[-1]))
    return (max(nums) + 1) if nums else 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_dir", type=str, default="dataset")
    ap.add_argument("--out_dir", type=str, default="synthetic_dataset")
    ap.add_argument("--labels_out", type=str, default="synthetic_label.csv")
    ap.add_argument("--num_clips", type=int, default=100)
    ap.add_argument("--num_frames", type=int, default=150)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--target_width", type=int, default=640)
    ap.add_argument("--seed", type=int, default=42)

    # Parameter ranges (sane defaults)
    ap.add_argument("--scale_min", type=float, default=0.90)
    ap.add_argument("--scale_max", type=float, default=1.15)
    ap.add_argument("--rot_deg_max", type=float, default=3.0)
    ap.add_argument("--trans_frac_max", type=float, default=0.12, help="Max translation over clip as fraction of width/height")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    dataset_dir = Path(args.dataset_dir)
    out_dir = Path(args.out_dir)
    labels_out = Path(args.labels_out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(dataset_dir.glob("*.mp4"))
    if not videos:
        raise RuntimeError(f"No .mp4 files found in {dataset_dir}")

    # Prepare labels file (append if exists)
    write_header = not labels_out.exists()
    f = labels_out.open("a", encoding="utf-8")
    try:
        if write_header:
            f.write("file_name, is_ken_burn, reasoning\n")

        start_idx = next_index(out_dir, "kb_syn")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        for j in range(args.num_clips):
            vid = rng.choice(videos)
            still = sample_random_frame(vid, rng)
            still = resize_to_width(still, args.target_width)
            h, w = still.shape[:2]

            # Sample end params (start is identity-ish)
            scale_start = 1.0
            scale_end = rng.uniform(args.scale_min, args.scale_max)
            rot_start = 0.0
            rot_end = rng.uniform(-args.rot_deg_max, args.rot_deg_max)

            max_tx = args.trans_frac_max * w
            max_ty = args.trans_frac_max * h
            tx_start = 0.0
            ty_start = 0.0
            tx_end = rng.uniform(-max_tx, max_tx)
            ty_end = rng.uniform(-max_ty, max_ty)

            clip, meta = render_kenburns_clip(
                still,
                num_frames=args.num_frames,
                scale_start=scale_start,
                scale_end=scale_end,
                rot_start_deg=rot_start,
                rot_end_deg=rot_end,
                tx_start=tx_start,
                tx_end=tx_end,
                ty_start=ty_start,
                ty_end=ty_end,
            )

            idx = start_idx + j
            name = f"kb_syn_{idx:04d}"
            out_path = out_dir / f"{name}.mp4"

            vw = cv2.VideoWriter(str(out_path), fourcc, float(args.fps), (w, h))
            if not vw.isOpened():
                raise RuntimeError(f"Could not open VideoWriter for {out_path}")
            try:
                for k in range(clip.shape[0]):
                    vw.write(clip[k])
            finally:
                vw.release()

            reasoning = {
                "source_video": vid.name,
                **meta,
            }
            f.write(f"{name}, True, {json.dumps(reasoning)}\n")

            if (j + 1) % 10 == 0 or (j + 1) == args.num_clips:
                print(f"Generated {j+1}/{args.num_clips}: {out_path}")

    finally:
        f.close()


if __name__ == "__main__":
    main()


