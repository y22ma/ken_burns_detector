import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict
import os
import pandas as pd
from pathlib import Path
import time


def load_video_frames(video_path: str, 
                     frame_skip: int = 1,
                     max_frames: Optional[int] = None,
                     target_width: Optional[int] = None,
                     target_height: Optional[int] = None,
                     scale_factor: Optional[float] = None) -> List[np.ndarray]:
    """
    Load an MP4 video file and extract frames as a list of numpy arrays.
    Supports temporal and spatial downsampling.
    
    Args:
        video_path: Path to the MP4 video file
        frame_skip: Take every Nth frame (1 = all frames, 2 = every other frame, etc.)
        max_frames: Maximum number of frames to load (None = all frames)
        target_width: Target width for resizing (None = keep original)
        target_height: Target height for resizing (None = keep original)
        scale_factor: Scale factor for resizing (e.g., 0.5 = half size). 
                     If specified, overrides target_width/target_height.
        
    Returns:
        List of numpy arrays, where each array represents a video frame in BGR format.
        Each frame has shape (height, width, 3).
        
    Raises:
        FileNotFoundError: If the video file doesn't exist
        ValueError: If the video file cannot be opened or is invalid
        
    Example:
        >>> frames = load_video_frames("video.mp4", frame_skip=2, scale_factor=0.5)
        >>> print(f"Loaded {len(frames)} frames")
        >>> print(f"Frame shape: {frames[0].shape}")
    """
    # Open video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frames = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                # End of video or error reading frame
                break
            
            # Temporal downsampling: skip frames
            if frame_count % frame_skip != 0:
                frame_count += 1
                continue
            
            # Spatial downsampling
            if scale_factor is not None and scale_factor != 1.0:
                height, width = frame.shape[:2]
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            elif target_width is not None or target_height is not None:
                if target_width is not None and target_height is not None:
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                elif target_width is not None:
                    height, width = frame.shape[:2]
                    aspect_ratio = height / width
                    target_height = int(target_width * aspect_ratio)
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
                elif target_height is not None:
                    height, width = frame.shape[:2]
                    aspect_ratio = width / height
                    target_width = int(target_height * aspect_ratio)
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            
            frames.append(frame)
            frame_count += 1
            
            # Limit number of frames
            if max_frames is not None and len(frames) >= max_frames:
                break
    
    finally:
        # Always release the video capture object
        cap.release()
    
    if len(frames) == 0:
        raise ValueError(f"No frames could be read from video: {video_path}")
    
    return frames


def load_video_frames_rgb(video_path: str,
                         frame_skip: int = 1,
                         max_frames: Optional[int] = None,
                         target_width: Optional[int] = None,
                         target_height: Optional[int] = None,
                         scale_factor: Optional[float] = None) -> List[np.ndarray]:
    """
    Load an MP4 video file and extract frames as a list of numpy arrays in RGB format.
    Supports temporal and spatial downsampling.
    
    OpenCV reads videos in BGR format by default. This function converts frames to RGB.
    
    Args:
        video_path: Path to the MP4 video file
        frame_skip: Take every Nth frame (1 = all frames, 2 = every other frame, etc.)
        max_frames: Maximum number of frames to load (None = all frames)
        target_width: Target width for resizing (None = keep original)
        target_height: Target height for resizing (None = keep original)
        scale_factor: Scale factor for resizing (e.g., 0.5 = half size).
                     If specified, overrides target_width/target_height.
        
    Returns:
        List of numpy arrays, where each array represents a video frame in RGB format.
        Each frame has shape (height, width, 3).
        
    Raises:
        FileNotFoundError: If the video file doesn't exist
        ValueError: If the video file cannot be opened or is invalid
    """
    frames_bgr = load_video_frames(video_path, 
                                  frame_skip=frame_skip,
                                  max_frames=max_frames,
                                  target_width=target_width,
                                  target_height=target_height,
                                  scale_factor=scale_factor)
    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    return frames_rgb


def compute_homography(frame1: np.ndarray, frame2: np.ndarray, 
                      method: str = 'SIFT', 
                      ransac_threshold: float = 5.0) -> Tuple[Optional[np.ndarray], float, int, int, float]:
    """
    Compute homography between two frames using feature matching.
    
    Args:
        frame1: First frame (BGR format)
        frame2: Second frame (BGR format)
        method: Feature detector method ('SIFT' or 'ORB')
        ransac_threshold: RANSAC reprojection threshold in pixels
        
    Returns:
        Tuple of (homography_matrix, mean_reprojection_error, num_inliers, num_matches, inlier_ratio)
        If homography cannot be computed, returns (None, float('inf'), 0, 0, 0.0)
    """
    # Convert to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Initialize feature detector
    if method == 'SIFT':
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher(crossCheck=False)
    elif method == 'ORB':
        detector = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'SIFT' or 'ORB'")
    
    # Detect keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return None, float('inf'), 0, 0, 0.0
    
    # Match features
    if method == 'SIFT':
        matches = matcher.knnMatch(des1, des2, k=2)
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
    else:  # ORB
        matches = matcher.knnMatch(des1, des2, k=2)
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
    
    num_matches = len(good_matches)
    if num_matches < 4:
        return None, float('inf'), 0, num_matches, 0.0
    
    # Extract matched points
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Compute homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, 
                                  cv2.RANSAC, 
                                  ransacReprojThreshold=ransac_threshold,
                                  maxIters=2000,
                                  confidence=0.99)
    
    if H is None:
        return None, float('inf'), 0, num_matches, 0.0
    
    # Compute reprojection error for inliers
    inlier_mask = mask.ravel() == 1
    num_inliers = np.sum(inlier_mask)
    inlier_ratio = num_inliers / num_matches if num_matches > 0 else 0.0
    
    if num_inliers == 0:
        return H, float('inf'), num_inliers, num_matches, inlier_ratio
    
    # Transform source points using homography
    src_pts_inliers = src_pts[inlier_mask]
    dst_pts_inliers = dst_pts[inlier_mask]
    
    # Convert to homogeneous coordinates
    src_pts_homogeneous = np.hstack([src_pts_inliers.reshape(-1, 2), np.ones((num_inliers, 1))])
    projected_pts = (H @ src_pts_homogeneous.T).T
    projected_pts = projected_pts[:, :2] / projected_pts[:, 2:3]
    
    # Compute reprojection error
    errors = np.linalg.norm(projected_pts - dst_pts_inliers.reshape(-1, 2), axis=1)
    mean_error = np.mean(errors)
    
    return H, mean_error, num_inliers, num_matches, inlier_ratio


def compute_homography_with_inlier_mask(
    frame1: np.ndarray,
    frame2: np.ndarray,
    method: str = "SIFT",
    ransac_threshold: float = 5.0,
) -> Tuple[
    Optional[np.ndarray],
    List[cv2.KeyPoint],
    List[cv2.KeyPoint],
    List[cv2.DMatch],
    Optional[np.ndarray],
    Optional[np.ndarray],
]:
    """
    Compute homography AND return the raw matching artifacts for visualization.

    Returns:
        (H, kp1, kp2, matches, inlier_mask, per_match_errors)

        - H: 3x3 homography or None
        - kp1/kp2: keypoints for frame1/frame2
        - matches: list of good matches after ratio test
        - inlier_mask: (N,) uint8 mask from cv2.findHomography for matches (1=inlier, 0=outlier) or None
        - per_match_errors: (N,) float reprojection error (pixels) for each match under H (NaN if H is None)
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    if method == "SIFT":
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher(crossCheck=False)
        ratio = 0.7
        norm = None
    elif method == "ORB":
        detector = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        ratio = 0.75
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError(f"Unknown method: {method}. Use 'SIFT' or 'ORB'")

    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return None, kp1, kp2, [], None, None

    matches_knn = matcher.knnMatch(des1, des2, k=2)
    good_matches: List[cv2.DMatch] = []
    for mp in matches_knn:
        if len(mp) != 2:
            continue
        m, n = mp
        if m.distance < ratio * n.distance:
            good_matches.append(m)

    if len(good_matches) < 4:
        return None, kp1, kp2, good_matches, None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(
        src_pts,
        dst_pts,
        cv2.RANSAC,
        ransacReprojThreshold=ransac_threshold,
        maxIters=2000,
        confidence=0.99,
    )

    if H is None or mask is None:
        per_err = np.full((len(good_matches),), np.nan, dtype=np.float32)
        return None, kp1, kp2, good_matches, None, per_err

    mask = mask.ravel().astype(np.uint8)

    # Per-match reprojection error under H (for all matches)
    src_xy = src_pts.reshape(-1, 2)
    dst_xy = dst_pts.reshape(-1, 2)
    src_h = np.hstack([src_xy, np.ones((len(src_xy), 1), dtype=np.float32)])
    proj = (H @ src_h.T).T
    proj_xy = proj[:, :2] / proj[:, 2:3]
    per_err = np.linalg.norm(proj_xy - dst_xy, axis=1).astype(np.float32)

    return H, kp1, kp2, good_matches, mask, per_err


def compute_reprojection_error_bidirectional(frame1: np.ndarray, frame2: np.ndarray,
                                           H: np.ndarray) -> Tuple[float, float]:
    """
    Compute reprojection error in both directions (frame1->frame2 and frame2->frame1).
    
    Args:
        frame1: First frame
        frame2: Second frame
        H: Homography matrix from frame1 to frame2
        
    Returns:
        Tuple of (error_forward, error_backward) where error_forward is frame1->frame2
        and error_backward is frame2->frame1
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Use dense sampling or feature points for error computation
    detector = cv2.SIFT_create()
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return float('inf'), float('inf')
    
    # Match features
    matcher = cv2.BFMatcher(crossCheck=False)
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
    
    if len(good_matches) < 4:
        return float('inf'), float('inf')
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Forward error: frame1 -> frame2
    src_pts_homogeneous = np.hstack([src_pts, np.ones((len(src_pts), 1))])
    projected_pts = (H @ src_pts_homogeneous.T).T
    projected_pts = projected_pts[:, :2] / projected_pts[:, 2:3]
    error_forward = np.mean(np.linalg.norm(projected_pts - dst_pts, axis=1))
    
    # Backward error: frame2 -> frame1
    H_inv = np.linalg.inv(H)
    dst_pts_homogeneous = np.hstack([dst_pts, np.ones((len(dst_pts), 1))])
    projected_pts_back = (H_inv @ dst_pts_homogeneous.T).T
    projected_pts_back = projected_pts_back[:, :2] / projected_pts_back[:, 2:3]
    error_backward = np.mean(np.linalg.norm(projected_pts_back - src_pts, axis=1))
    
    return error_forward, error_backward


def decompose_homography(H: np.ndarray) -> Tuple[float, float, float, float, float]:
    """
    Decompose a 2D homography matrix into rotation, scale, and translation.
    Assumes similarity transform (uniform scale, rotation, translation).
    
    Args:
        H: 3x3 homography matrix
        
    Returns:
        Tuple of (rotation_angle_rad, rotation_angle_deg, scale, tx, ty)
        Rotation angle in both radians and degrees, scale factor, and translation
    """
    if H is None:
        return 0.0, 0.0, 1.0, 0.0, 0.0
    
    # Extract rotation and scale from the upper-left 2x2 submatrix
    # For similarity transform: H = [s*cos(θ) -s*sin(θ) tx]
    #                              [s*sin(θ)  s*cos(θ) ty]
    #                              [0          0       1]
    
    h00, h01, h02 = H[0, 0], H[0, 1], H[0, 2]
    h10, h11, h12 = H[1, 0], H[1, 1], H[1, 2]
    
    # Compute scale from the rotation matrix part
    # Scale can be computed as: s = sqrt(h00^2 + h10^2) or sqrt(h01^2 + h11^2)
    scale1 = np.sqrt(h00**2 + h10**2)
    scale2 = np.sqrt(h01**2 + h11**2)
    scale = (scale1 + scale2) / 2.0  # Average for robustness
    
    # Compute rotation angle
    # θ = atan2(h10, h00) or atan2(-h01, h11)
    angle1 = np.arctan2(h10, h00)
    angle2 = np.arctan2(-h01, h11)
    angle_rad = (angle1 + angle2) / 2.0  # Average for robustness
    angle_deg = np.degrees(angle_rad)
    
    # Translation
    tx = h02
    ty = h12
    
    return angle_rad, angle_deg, scale, tx, ty


def compute_smoothness_metrics(parameters: List[float]) -> Dict[str, float]:
    """
    Compute smoothness metrics for a sequence of parameters.
    
    Args:
        parameters: List of parameter values over time
        
    Returns:
        Dictionary with smoothness metrics:
        - 'variance': variance of the parameter values
        - 'std': standard deviation
        - 'first_derivative_variance': variance of first differences
        - 'second_derivative_variance': variance of second differences
        - 'max_first_derivative': maximum absolute first difference
        - 'max_second_derivative': maximum absolute second difference
    """
    if len(parameters) == 0:
        return {
            'variance': 0.0,
            'std': 0.0,
            'first_derivative_variance': 0.0,
            'second_derivative_variance': 0.0,
            'max_first_derivative': 0.0,
            'max_second_derivative': 0.0
        }
    
    params_array = np.array(parameters)
    
    # Basic statistics
    variance = np.var(params_array)
    std = np.std(params_array)
    
    # First derivative (differences between consecutive values)
    if len(parameters) > 1:
        first_diff = np.diff(params_array)
        first_derivative_variance = np.var(first_diff)
        max_first_derivative = np.max(np.abs(first_diff))
    else:
        first_derivative_variance = 0.0
        max_first_derivative = 0.0
    
    # Second derivative (rate of change of first derivative)
    if len(parameters) > 2:
        second_diff = np.diff(first_diff)
        second_derivative_variance = np.var(second_diff)
        max_second_derivative = np.max(np.abs(second_diff))
    else:
        second_derivative_variance = 0.0
        max_second_derivative = 0.0
    
    return {
        'variance': variance,
        'std': std,
        'first_derivative_variance': first_derivative_variance,
        'second_derivative_variance': second_derivative_variance,
        'max_first_derivative': max_first_derivative,
        'max_second_derivative': max_second_derivative
    }


def process_video_homography(video_path: str, 
                           method: str = 'SIFT',
                           ransac_threshold: float = 5.0,
                           frame_skip: int = 1,
                           max_frames: Optional[int] = None,
                           scale_factor: Optional[float] = None,
                           target_width: Optional[int] = None) -> Dict:
    """
    Process a video and compute homography statistics between consecutive frame pairs.
    
    Args:
        video_path: Path to video file
        method: Feature detection method ('SIFT' or 'ORB')
        ransac_threshold: RANSAC threshold for homography estimation
        frame_skip: Take every Nth frame (1 = all frames, 2 = every other frame, etc.)
        max_frames: Maximum number of frames to process (None = all frames)
        scale_factor: Scale factor for spatial downsampling (e.g., 0.5 = half size)
        target_width: Target width for resizing (None = keep original)
        
    Returns:
        Dictionary containing:
        - 'file_name': video file name
        - 'num_frames': number of frames
        - 'num_pairs': number of frame pairs processed
        - 'homographies': list of homography matrices
        - 'errors_forward': list of forward reprojection errors
        - 'errors_backward': list of backward reprojection errors
        - 'num_inliers': list of inlier counts
        - 'mean_error': mean reprojection error across all pairs
        - 'std_error': standard deviation of errors
        - 'max_error': maximum error
        - 'min_error': minimum error
        - 'mean_inliers': mean number of inliers
    """
    load_start = time.time()
    frames = load_video_frames(video_path, 
                              frame_skip=frame_skip,
                              max_frames=max_frames,
                              scale_factor=scale_factor,
                              target_width=target_width)
    load_time = time.time() - load_start
    
    if len(frames) < 2:
        return {
            'file_name': os.path.basename(video_path),
            'num_frames': len(frames),
            'num_pairs': 0,
            'mean_error': float('inf'),
            'std_error': 0,
            'max_error': float('inf'),
            'min_error': float('inf'),
            'mean_inliers': 0
        }
    
    print(f"    ⏱  Loaded {len(frames)} frames in {load_time:.2f}s")
    
    homographies = []
    errors_forward = []
    errors_backward = []
    num_inliers_list = []
    num_matches_list = []
    inlier_ratios = []
    
    # Process consecutive frame pairs
    pair_times = []
    for i in range(len(frames) - 1):
        pair_start = time.time()
        H, error, n_inliers, n_matches, inlier_ratio = compute_homography(frames[i], frames[i+1], 
                                                 method=method,
                                                 ransac_threshold=ransac_threshold)
        
        if H is not None:
            # Compute bidirectional error
            err_fwd, err_bwd = compute_reprojection_error_bidirectional(
                frames[i], frames[i+1], H)
            
            homographies.append(H)
            errors_forward.append(err_fwd)
            errors_backward.append(err_bwd)
            num_inliers_list.append(n_inliers)
            num_matches_list.append(n_matches)
            inlier_ratios.append(inlier_ratio)
        else:
            # Failed to compute homography
            homographies.append(None)
            errors_forward.append(float('inf'))
            errors_backward.append(float('inf'))
            num_inliers_list.append(0)
            num_matches_list.append(n_matches if n_matches > 0 else 0)
            inlier_ratios.append(0.0)
        
        pair_time = time.time() - pair_start
        pair_times.append(pair_time)
        
        # Print progress every 10 pairs
        if (i + 1) % 10 == 0:
            avg_pair_time = np.mean(pair_times[-10:])
            print(f"    ⏱  Processed {i+1}/{len(frames)-1} pairs "
                  f"(avg {avg_pair_time:.3f}s/pair, {avg_pair_time*(len(frames)-1-i):.1f}s remaining)")
    
    # Filter out infinite errors for statistics
    valid_errors = [e for e in errors_forward if e != float('inf')]
    
    # Extract homography parameters (rotation, scale, translation)
    rotations_deg = []
    scales = []
    translations_x = []
    translations_y = []
    
    for H in homographies:
        if H is not None:
            angle_rad, angle_deg, scale, tx, ty = decompose_homography(H)
            rotations_deg.append(angle_deg)
            scales.append(scale)
            translations_x.append(tx)
            translations_y.append(ty)
        else:
            rotations_deg.append(None)
            scales.append(None)
            translations_x.append(None)
            translations_y.append(None)
    
    # Compute smoothness metrics for valid parameters
    valid_rotations = [r for r in rotations_deg if r is not None]
    valid_scales = [s for s in scales if s is not None]
    valid_tx = [tx for tx in translations_x if tx is not None]
    valid_ty = [ty for ty in translations_y if ty is not None]
    
    rotation_smoothness = compute_smoothness_metrics(valid_rotations) if len(valid_rotations) > 0 else {}
    scale_smoothness = compute_smoothness_metrics(valid_scales) if len(valid_scales) > 0 else {}
    tx_smoothness = compute_smoothness_metrics(valid_tx) if len(valid_tx) > 0 else {}
    ty_smoothness = compute_smoothness_metrics(valid_ty) if len(valid_ty) > 0 else {}
    
    result = {
        'file_name': os.path.basename(video_path),
        'num_frames': len(frames),
        'num_pairs': len(homographies),
        'homographies': homographies,
        'errors_forward': errors_forward,
        'errors_backward': errors_backward,
        'num_inliers': num_inliers_list,
        'num_matches': num_matches_list,
        'inlier_ratios': inlier_ratios,
        'rotations_deg': rotations_deg,
        'scales': scales,
        'translations_x': translations_x,
        'translations_y': translations_y,
    }
    
    if len(valid_errors) > 0:
        result['mean_error'] = np.mean(valid_errors)
        result['std_error'] = np.std(valid_errors)
        result['max_error'] = np.max(valid_errors)
        result['min_error'] = np.min(valid_errors)
        result['mean_inliers'] = np.mean([n for n in num_inliers_list if n > 0])
        # Compute inlier ratio statistics
        valid_ratios = [r for r in inlier_ratios if r > 0]
        if len(valid_ratios) > 0:
            result['mean_inlier_ratio'] = np.mean(valid_ratios)
            result['std_inlier_ratio'] = np.std(valid_ratios)
            result['min_inlier_ratio'] = np.min(valid_ratios)
            result['max_inlier_ratio'] = np.max(valid_ratios)
        else:
            result['mean_inlier_ratio'] = 0.0
            result['std_inlier_ratio'] = 0.0
            result['min_inlier_ratio'] = 0.0
            result['max_inlier_ratio'] = 0.0
        
        # Compute outlier statistics
        num_outliers_list = [matches - inliers for matches, inliers in zip(num_matches_list, num_inliers_list)]
        valid_outliers = [o for o in num_outliers_list if o >= 0]
        if len(valid_outliers) > 0:
            result['mean_outliers'] = np.mean(valid_outliers)
            result['std_outliers'] = np.std(valid_outliers)
        else:
            result['mean_outliers'] = 0.0
            result['std_outliers'] = 0.0
        
        # Compute inlier/outlier ratio
        inlier_outlier_ratios = []
        for inliers, outliers in zip(num_inliers_list, num_outliers_list):
            if outliers > 0:
                inlier_outlier_ratios.append(inliers / outliers)
            elif inliers > 0:
                inlier_outlier_ratios.append(float('inf'))  # All matches are inliers
            else:
                inlier_outlier_ratios.append(0.0)
        
        valid_io_ratios = [r for r in inlier_outlier_ratios if r != float('inf') and r > 0]
        if len(valid_io_ratios) > 0:
            result['mean_inlier_outlier_ratio'] = np.mean(valid_io_ratios)
            result['std_inlier_outlier_ratio'] = np.std(valid_io_ratios)
            result['min_inlier_outlier_ratio'] = np.min(valid_io_ratios)
            result['max_inlier_outlier_ratio'] = np.max(valid_io_ratios)
        else:
            result['mean_inlier_outlier_ratio'] = 0.0
            result['std_inlier_outlier_ratio'] = 0.0
            result['min_inlier_outlier_ratio'] = 0.0
            result['max_inlier_outlier_ratio'] = 0.0
        
        result['num_outliers'] = num_outliers_list
        result['inlier_outlier_ratios'] = inlier_outlier_ratios
        
        # Add smoothness metrics
        if len(rotation_smoothness) > 0:
            result['rotation_variance'] = rotation_smoothness['variance']
            result['rotation_first_deriv_variance'] = rotation_smoothness['first_derivative_variance']
            result['rotation_second_deriv_variance'] = rotation_smoothness['second_derivative_variance']
            result['rotation_max_first_deriv'] = rotation_smoothness['max_first_derivative']
        else:
            result['rotation_variance'] = float('inf')
            result['rotation_first_deriv_variance'] = float('inf')
            result['rotation_second_deriv_variance'] = float('inf')
            result['rotation_max_first_deriv'] = float('inf')
        
        if len(scale_smoothness) > 0:
            result['scale_variance'] = scale_smoothness['variance']
            result['scale_first_deriv_variance'] = scale_smoothness['first_derivative_variance']
            result['scale_second_deriv_variance'] = scale_smoothness['second_derivative_variance']
            result['scale_max_first_deriv'] = scale_smoothness['max_first_derivative']
        else:
            result['scale_variance'] = float('inf')
            result['scale_first_deriv_variance'] = float('inf')
            result['scale_second_deriv_variance'] = float('inf')
            result['scale_max_first_deriv'] = float('inf')
        
        if len(tx_smoothness) > 0:
            result['tx_variance'] = tx_smoothness['variance']
            result['tx_first_deriv_variance'] = tx_smoothness['first_derivative_variance']
            result['tx_second_deriv_variance'] = tx_smoothness['second_derivative_variance']
            result['tx_max_first_deriv'] = tx_smoothness['max_first_derivative']
        else:
            result['tx_variance'] = float('inf')
            result['tx_first_deriv_variance'] = float('inf')
            result['tx_second_deriv_variance'] = float('inf')
            result['tx_max_first_deriv'] = float('inf')
        
        if len(ty_smoothness) > 0:
            result['ty_variance'] = ty_smoothness['variance']
            result['ty_first_deriv_variance'] = ty_smoothness['first_derivative_variance']
            result['ty_second_deriv_variance'] = ty_smoothness['second_derivative_variance']
            result['ty_max_first_deriv'] = ty_smoothness['max_first_derivative']
        else:
            result['ty_variance'] = float('inf')
            result['ty_first_deriv_variance'] = float('inf')
            result['ty_second_deriv_variance'] = float('inf')
            result['ty_max_first_deriv'] = float('inf')
    else:
        result['mean_error'] = float('inf')
        result['std_error'] = 0
        result['max_error'] = float('inf')
        result['min_error'] = float('inf')
        result['mean_inliers'] = 0
        result['mean_inlier_ratio'] = 0.0
        result['std_inlier_ratio'] = 0.0
        result['min_inlier_ratio'] = 0.0
        result['max_inlier_ratio'] = 0.0
        result['mean_outliers'] = 0.0
        result['std_outliers'] = 0.0
        result['mean_inlier_outlier_ratio'] = 0.0
        result['std_inlier_outlier_ratio'] = 0.0
        result['min_inlier_outlier_ratio'] = 0.0
        result['max_inlier_outlier_ratio'] = 0.0
        result['num_outliers'] = []
        result['inlier_outlier_ratios'] = []
        # Set smoothness metrics to infinity for failed cases
        result['rotation_variance'] = float('inf')
        result['rotation_first_deriv_variance'] = float('inf')
        result['rotation_second_deriv_variance'] = float('inf')
        result['rotation_max_first_deriv'] = float('inf')
        result['scale_variance'] = float('inf')
        result['scale_first_deriv_variance'] = float('inf')
        result['scale_second_deriv_variance'] = float('inf')
        result['scale_max_first_deriv'] = float('inf')
        result['tx_variance'] = float('inf')
        result['tx_first_deriv_variance'] = float('inf')
        result['tx_second_deriv_variance'] = float('inf')
        result['tx_max_first_deriv'] = float('inf')
        result['ty_variance'] = float('inf')
        result['ty_first_deriv_variance'] = float('inf')
        result['ty_second_deriv_variance'] = float('inf')
        result['ty_max_first_deriv'] = float('inf')
    
    return result


def parse_labels(label_file: str) -> Dict[str, bool]:
    """
    Parse the label.txt file to extract Ken Burns labels.
    
    Args:
        label_file: Path to label.txt file
        
    Returns:
        Dictionary mapping file names (without extension) to boolean labels
    """
    labels = {}
    with open(label_file, 'r') as f:
        lines = f.readlines()
    
    # Skip header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        
        # Handle the typo in line 5 (4. True -> 4, True)
        line = line.replace('4. True', '4, True')
        
        parts = line.split(',', 2)
        if len(parts) >= 2:
            file_name = parts[0].strip()
            is_ken_burn = parts[1].strip().lower() == 'true'
            labels[file_name] = is_ken_burn
    
    return labels

