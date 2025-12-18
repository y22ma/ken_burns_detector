#!/usr/bin/env python3
"""
Visualize inlier vs outlier correspondences for a homography between two frames.

Outputs:
  - matches.png: side-by-side match visualization (inliers=green, outliers=red)
  - overlay.png: frame2 overlaid with warped frame1 (optional)

Also prints reprojection error stats for ALL matches and for INLIERS only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from utils import load_video_frames, compute_homography_with_inlier_mask


def _draw_colored_matches(
    img1: np.ndarray,
    kp1: List[cv2.KeyPoint],
    img2: np.ndarray,
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    inlier_mask: np.ndarray,
) -> np.ndarray:
    """Draw matches with inliers green and outliers red."""
    if len(matches) == 0:
        return np.zeros((10, 10, 3), dtype=np.uint8)

    # Draw ALL matches (no max_draw truncation)
    inlier_matches = [m for i, m in enumerate(matches) if inlier_mask is not None and inlier_mask[i] == 1]
    outlier_matches = [m for i, m in enumerate(matches) if inlier_mask is None or inlier_mask[i] == 0]

    flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    canvas = cv2.drawMatches(img1, kp1, img2, kp2, [], None, flags=flags)

    if len(outlier_matches) > 0:
        canvas = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            outlier_matches,
            canvas,
            matchColor=(0, 0, 255),  # red (BGR)
            singlePointColor=None,
            flags=flags,
        )

    if len(inlier_matches) > 0:
        canvas = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            inlier_matches,
            canvas,
            matchColor=(0, 255, 0),  # green (BGR)
            singlePointColor=None,
            flags=flags,
        )

    return canvas


def _warp_overlay(frame1: np.ndarray, frame2: np.ndarray, H: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Warp frame1 into frame2 coords and alpha-blend with frame2."""
    h2, w2 = frame2.shape[:2]
    warped = cv2.warpPerspective(frame1, H, (w2, h2))
    overlay = cv2.addWeighted(frame2, 1.0 - alpha, warped, alpha, 0.0)
    return overlay


def _error_heatmap_overlay(
    frame1: np.ndarray,
    frame2: np.ndarray,
    H: np.ndarray,
    alpha: float = 0.5,
    clip_percentile: float = 99.0,
) -> np.ndarray:
    """
    Warp frame1 into frame2 coords, compute per-pixel error, and overlay a heatmap on frame2.

    Error is computed on grayscale as absolute difference.
    Values are clipped at clip_percentile for readability.
    """
    h2, w2 = frame2.shape[:2]
    warped = cv2.warpPerspective(frame1, H, (w2, h2))

    g2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gw = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    err = np.abs(g2 - gw)  # 0..255

    # Mask out regions where warp produced nothing (black). This isn't perfect but helps a lot.
    valid = (cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0).astype(np.uint8)
    err_masked = err[valid.astype(bool)]
    if err_masked.size == 0:
        # Fallback: no valid region
        err_norm = np.zeros_like(err, dtype=np.uint8)
    else:
        hi = float(np.percentile(err_masked, clip_percentile))
        hi = max(1.0, hi)
        err_clip = np.clip(err, 0.0, hi)
        err_norm = (err_clip / hi * 255.0).astype(np.uint8)

    heat = cv2.applyColorMap(err_norm, cv2.COLORMAP_TURBO)
    out = cv2.addWeighted(frame2, 1.0 - alpha, heat, alpha, 0.0)

    # Dim invalid region to avoid confusion
    inv = (valid == 0)
    out[inv] = (out[inv] * 0.35).astype(out.dtype)
    return out


def _print_error_stats(per_err: np.ndarray, inlier_mask: np.ndarray | None) -> None:
    per_err = np.asarray(per_err, dtype=np.float64)
    finite = np.isfinite(per_err)
    if finite.sum() == 0:
        print("No finite reprojection errors to report.")
        return

    def stats(x: np.ndarray) -> Tuple[float, float, float, float]:
        return float(np.mean(x)), float(np.std(x)), float(np.min(x)), float(np.max(x))

    all_err = per_err[finite]
    m, s, mn, mx = stats(all_err)
    print(f"Reprojection error (ALL matches): mean={m:.4f}  std={s:.4f}  min={mn:.4f}  max={mx:.4f}  n={len(all_err)}")

    if inlier_mask is None:
        return

    inlier_mask = np.asarray(inlier_mask).astype(bool)
    inl = inlier_mask & finite
    outl = (~inlier_mask) & finite
    if inl.sum() > 0:
        m, s, mn, mx = stats(per_err[inl])
        print(f"Reprojection error (INLIERS only): mean={m:.4f}  std={s:.4f}  min={mn:.4f}  max={mx:.4f}  n={int(inl.sum())}")
    if outl.sum() > 0:
        m, s, mn, mx = stats(per_err[outl])
        print(f"Reprojection error (OUTLIERS only): mean={m:.4f}  std={s:.4f}  min={mn:.4f}  max={mx:.4f}  n={int(outl.sum())}")


def _draw_inlier_outlier_points_on_frame2(
    frame2: np.ndarray,
    kp1: List[cv2.KeyPoint],
    kp2: List[cv2.KeyPoint],
    matches: List[cv2.DMatch],
    inlier_mask: np.ndarray | None,
    radius: int = 3,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw keypoint locations for matched points on frame2.

    - Inliers: green circles
    - Outliers: red circles
    """
    out = frame2.copy()
    if len(matches) == 0:
        return out

    if inlier_mask is None:
        inlier_mask = np.zeros((len(matches),), dtype=np.uint8)
    else:
        inlier_mask = np.asarray(inlier_mask).ravel().astype(np.uint8)

    for i, m in enumerate(matches):
        pt2 = tuple(int(x) for x in kp2[m.trainIdx].pt)
        if inlier_mask[i] == 1:
            color = (0, 255, 0)  # green
        else:
            color = (0, 0, 255)  # red
        cv2.circle(out, pt2, radius, color, thickness)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=str, required=True, help="Path to .mp4 (e.g., dataset/32.mp4)")
    ap.add_argument("--frame_idx", type=int, default=0, help="Index of first frame")
    ap.add_argument("--step", type=int, default=1, help="Compare frame_idx vs frame_idx+step")
    ap.add_argument("--method", type=str, default="ORB", choices=["ORB", "SIFT"])
    ap.add_argument("--ransac_threshold", type=float, default=2.0)
    ap.add_argument("--target_width", type=int, default=640)
    ap.add_argument("--max_frames", type=int, default=0, help="0 means no limit")
    ap.add_argument("--out_dir", type=str, default="viz_out")
    ap.add_argument("--overlay", action="store_true", help="Also save warped overlay image")
    ap.add_argument("--error_overlay", action="store_true", help="Save reprojection error heatmap overlay")
    ap.add_argument("--error_alpha", type=float, default=0.55, help="Alpha for error heatmap overlay")
    ap.add_argument("--error_clip_percentile", type=float, default=99.0, help="Clip error heatmap at this percentile")
    ap.add_argument("--points", action="store_true", help="Save frame2 with inlier/outlier points (inlier=green, outlier=red)")
    ap.add_argument("--points_on_error", action="store_true", help="If --error_overlay, also draw inlier/outlier points on the error overlay")
    ap.add_argument("--point_radius", type=int, default=3)
    ap.add_argument("--point_thickness", type=int, default=2)
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(str(video_path))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    max_frames = None if args.max_frames == 0 else int(args.max_frames)
    frames = load_video_frames(
        str(video_path),
        frame_skip=1,
        max_frames=max_frames,
        target_width=args.target_width,
    )
    if args.frame_idx < 0 or args.frame_idx + args.step >= len(frames):
        raise ValueError(f"Invalid frame indices: frame_idx={args.frame_idx}, step={args.step}, num_frames={len(frames)}")

    f1 = frames[args.frame_idx]
    f2 = frames[args.frame_idx + args.step]

    H, kp1, kp2, matches, mask, per_err = compute_homography_with_inlier_mask(
        f1, f2, method=args.method, ransac_threshold=args.ransac_threshold
    )

    print(f"video={video_path.name}  frames={len(frames)}  pair=({args.frame_idx},{args.frame_idx+args.step})")
    print(f"method={args.method}  ransac_threshold={args.ransac_threshold}")
    print(f"matches={len(matches)}  inliers={(int(mask.sum()) if mask is not None else 0)}  outliers={(len(matches)-int(mask.sum()) if mask is not None else len(matches))}")

    if H is None or mask is None:
        print("Homography failed (H is None).")
    else:
        _print_error_stats(per_err, mask)

    canvas = _draw_colored_matches(f1, kp1, f2, kp2, matches, mask)
    matches_path = out_dir / f"{video_path.stem}_f{args.frame_idx}_to_f{args.frame_idx+args.step}_matches.png"
    cv2.imwrite(str(matches_path), canvas)
    print(f"Saved: {matches_path}")

    if args.overlay and H is not None:
        overlay = _warp_overlay(f1, f2, H, alpha=0.5)
        overlay_path = out_dir / f"{video_path.stem}_f{args.frame_idx}_to_f{args.frame_idx+args.step}_overlay.png"
        cv2.imwrite(str(overlay_path), overlay)
        print(f"Saved: {overlay_path}")

    if args.error_overlay and H is not None:
        err_overlay = _error_heatmap_overlay(
            f1,
            f2,
            H,
            alpha=float(args.error_alpha),
            clip_percentile=float(args.error_clip_percentile),
        )
        err_path = out_dir / f"{video_path.stem}_f{args.frame_idx}_to_f{args.frame_idx+args.step}_error_overlay.png"
        cv2.imwrite(str(err_path), err_overlay)
        print(f"Saved: {err_path}")

        if args.points_on_error:
            err_with_pts = _draw_inlier_outlier_points_on_frame2(
                err_overlay,
                kp1,
                kp2,
                matches,
                mask,
                radius=int(args.point_radius),
                thickness=int(args.point_thickness),
            )
            err_pts_path = out_dir / f"{video_path.stem}_f{args.frame_idx}_to_f{args.frame_idx+args.step}_error_overlay_points.png"
            cv2.imwrite(str(err_pts_path), err_with_pts)
            print(f"Saved: {err_pts_path}")

    if args.points:
        pts_img = _draw_inlier_outlier_points_on_frame2(
            f2,
            kp1,
            kp2,
            matches,
            mask,
            radius=int(args.point_radius),
            thickness=int(args.point_thickness),
        )
        pts_path = out_dir / f"{video_path.stem}_f{args.frame_idx}_to_f{args.frame_idx+args.step}_points.png"
        cv2.imwrite(str(pts_path), pts_img)
        print(f"Saved: {pts_path}")


if __name__ == "__main__":
    main()


