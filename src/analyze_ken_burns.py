#!/usr/bin/env python3
"""
Analyze videos to detect Ken Burns effects using homography-based reprojection error.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict
import time
import pickle
import hashlib
import json
from utils import process_video_homography, parse_labels
from tqdm import tqdm


def print_metrics(result: Dict, processing_time: float, total_time: float, from_cache: bool = False):
    """
    Print all computed metrics in a nicely formatted way.
    
    Args:
        result: Dictionary containing all computed metrics
        processing_time: Time taken to process the video
        total_time: Total time including loading
        from_cache: Whether this result was loaded from cache
    """
    print("\n" + "=" * 70)
    print(f"METRICS SUMMARY: {result.get('file_name', 'Unknown')}")
    if from_cache:
        print("  (Loaded from cache)")
    print("=" * 70)
    
    # Basic video info
    print("\n📹 Video Information:")
    print(f"  Frames: {result.get('num_frames', 0)}")
    print(f"  Frame pairs processed: {result.get('num_pairs', 0)}")
    if result.get('label_available'):
        kb_status = "✓ Ken Burns" if result.get('is_ken_burn') else "✗ Not Ken Burns"
        print(f"  Label: {kb_status}")
    
    # Reprojection error metrics
    print("\n📊 Reprojection Error Metrics:")
    mean_err = result.get('mean_error', float('inf'))
    if mean_err != float('inf'):
        print(f"  Mean error:        {mean_err:.4f} px")
        print(f"  Std error:         {result.get('std_error', 0):.4f} px")
        print(f"  Min error:         {result.get('min_error', float('inf')):.4f} px")
        print(f"  Max error:         {result.get('max_error', float('inf')):.4f} px")
    else:
        print("  ⚠ Failed to compute homography")
    
    # Inlier/Outlier metrics
    print("\n🎯 Inlier/Outlier Metrics:")
    print(f"  Mean inliers:      {result.get('mean_inliers', 0):.1f}")
    mean_outliers = result.get('mean_outliers', 0)
    if mean_outliers > 0:
        print(f"  Mean outliers:     {mean_outliers:.1f}")
        print(f"  Std outliers:      {result.get('std_outliers', 0):.1f}")
    
    mean_io_ratio = result.get('mean_inlier_outlier_ratio', 0)
    if mean_io_ratio > 0 and mean_io_ratio != float('inf'):
        print(f"  Inlier/Outlier ratio: {mean_io_ratio:.4f}")
        print(f"    (Std: {result.get('std_inlier_outlier_ratio', 0):.4f})")
        print(f"    (Range: [{result.get('min_inlier_outlier_ratio', 0):.4f}, "
              f"{result.get('max_inlier_outlier_ratio', 0):.4f}])")
    
    # Inlier ratio metrics
    mean_inlier_ratio = result.get('mean_inlier_ratio', 0)
    if mean_inlier_ratio > 0:
        print(f"  Mean inlier ratio:  {mean_inlier_ratio:.4f}")
        print(f"    (Std: {result.get('std_inlier_ratio', 0):.4f})")
        print(f"    (Range: [{result.get('min_inlier_ratio', 0):.4f}, "
              f"{result.get('max_inlier_ratio', 0):.4f}])")
    
    # Homography parameter smoothness - Rotation
    print("\n🔄 Rotation Smoothness:")
    rot_var = result.get('rotation_variance', float('inf'))
    if rot_var != float('inf'):
        print(f"  Variance:                    {rot_var:.6f} deg²")
        print(f"  1st derivative variance:    {result.get('rotation_first_deriv_variance', 0):.6f}")
        print(f"  2nd derivative variance:    {result.get('rotation_second_deriv_variance', 0):.6f}")
        print(f"  Max 1st derivative:         {result.get('rotation_max_first_deriv', 0):.6f} deg")
    else:
        print("  ⚠ No valid rotation data")
    
    # Homography parameter smoothness - Scale
    print("\n📏 Scale Smoothness:")
    scale_var = result.get('scale_variance', float('inf'))
    if scale_var != float('inf'):
        print(f"  Variance:                    {scale_var:.6f}")
        print(f"  1st derivative variance:    {result.get('scale_first_deriv_variance', 0):.6f}")
        print(f"  2nd derivative variance:    {result.get('scale_second_deriv_variance', 0):.6f}")
        print(f"  Max 1st derivative:         {result.get('scale_max_first_deriv', 0):.6f}")
    else:
        print("  ⚠ No valid scale data")
    
    # Homography parameter smoothness - Translation X
    print("\n↔️  Translation X Smoothness:")
    tx_var = result.get('tx_variance', float('inf'))
    if tx_var != float('inf'):
        print(f"  Variance:                    {tx_var:.4f} px²")
        print(f"  1st derivative variance:    {result.get('tx_first_deriv_variance', 0):.4f}")
        print(f"  2nd derivative variance:    {result.get('tx_second_deriv_variance', 0):.4f}")
        print(f"  Max 1st derivative:         {result.get('tx_max_first_deriv', 0):.4f} px")
    else:
        print("  ⚠ No valid translation X data")
    
    # Homography parameter smoothness - Translation Y
    print("\n↕️  Translation Y Smoothness:")
    ty_var = result.get('ty_variance', float('inf'))
    if ty_var != float('inf'):
        print(f"  Variance:                    {ty_var:.4f} px²")
        print(f"  1st derivative variance:    {result.get('ty_first_deriv_variance', 0):.4f}")
        print(f"  2nd derivative variance:    {result.get('ty_second_deriv_variance', 0):.4f}")
        print(f"  Max 1st derivative:         {result.get('ty_max_first_deriv', 0):.4f} px")
    else:
        print("  ⚠ No valid translation Y data")
    
    # Timing information
    if processing_time > 0:
        num_pairs = result.get('num_pairs', 1)
        print("\n⏱️  Performance:")
        print(f"  Processing time:   {processing_time:.2f}s")
        if num_pairs > 0:
            print(f"  Time per pair:    {processing_time/num_pairs:.3f}s")
        print(f"  Total time:        {total_time:.2f}s")
    
    print("=" * 70 + "\n")


def get_cache_key(video_path: str, method: str, ransac_threshold: float,
                 frame_skip: int, max_frames: Optional[int],
                 scale_factor: Optional[float], target_width: Optional[int]) -> str:
    """
    Generate a cache key based on video path and processing parameters.
    
    Args:
        video_path: Path to video file
        method: Feature detection method
        ransac_threshold: RANSAC threshold
        frame_skip: Frame skip parameter
        max_frames: Max frames parameter
        scale_factor: Scale factor parameter
        target_width: Target width parameter
        
    Returns:
        Cache key string
    """
    # Get video file modification time and size for cache invalidation
    try:
        stat = os.stat(video_path)
        video_info = f"{stat.st_mtime}_{stat.st_size}"
    except:
        video_info = "unknown"
    
    params = {
        'method': method,
        'ransac_threshold': ransac_threshold,
        'frame_skip': frame_skip,
        'max_frames': max_frames,
        'scale_factor': scale_factor,
        'target_width': target_width,
        'video_info': video_info
    }
    
    param_str = json.dumps(params, sort_keys=True)
    cache_key = hashlib.md5(param_str.encode()).hexdigest()
    return cache_key


def load_cache(cache_file: str = 'homography_cache.pkl') -> Dict:
    """
    Load cached results from file.
    
    Args:
        cache_file: Path to cache file
        
    Returns:
        Dictionary mapping cache keys to results
    """
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Warning: Could not load cache file {cache_file}: {e}")
            return {}
    return {}


def save_cache(cache: Dict, cache_file: str = 'homography_cache.pkl'):
    """
    Save results to cache file.
    
    Args:
        cache: Dictionary mapping cache keys to results
        cache_file: Path to cache file
    """
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(cache, f)
    except Exception as e:
        print(f"Warning: Could not save cache file {cache_file}: {e}")


def analyze_all_videos(dataset_dir: str, label_file: str, 
                      method: str = 'SIFT',
                      ransac_threshold: float = 5.0,
                      frame_skip: int = 1,
                      max_frames: Optional[int] = None,
                      scale_factor: Optional[float] = None,
                      target_width: Optional[int] = None,
                      max_videos: Optional[int] = None,
                      cache_file: str = 'homography_cache.pkl',
                      force_reprocess: bool = False) -> pd.DataFrame:
    """
    Process all videos in the dataset and compute homography statistics.
    
    Args:
        dataset_dir: Directory containing video files
        label_file: Path to label.txt file
        method: Feature detection method
        ransac_threshold: RANSAC threshold
        frame_skip: Take every Nth frame (1 = all frames, 2 = every other frame, etc.)
        max_frames: Maximum number of frames to process per video (None = all frames)
        scale_factor: Scale factor for spatial downsampling (e.g., 0.5 = half size)
        target_width: Target width for resizing (None = keep original)
        max_videos: Maximum number of NEGATIVE cases to process (None = all negative cases).
                    All positive cases (Ken Burns=True) are always included regardless of this limit.
        
    Returns:
        DataFrame with results for each video
    """
    labels = parse_labels(label_file)
    results = []
    
    # Load cache
    cache = {} if force_reprocess else load_cache(cache_file)
    cache_updated = False
    
    # Get all video files
    all_video_files = sorted(Path(dataset_dir).glob("*.mp4"))
    
    # Separate videos into positive (Ken Burns) and negative (not Ken Burns) cases
    positive_videos = []
    negative_videos = []
    
    for video_path in all_video_files:
        file_name = video_path.stem
        if file_name in labels:
            if labels[file_name]:  # True = Ken Burns (positive)
                positive_videos.append(video_path)
            else:  # False = Not Ken Burns (negative)
                negative_videos.append(video_path)
        else:
            # Unlabeled videos go to negative category
            negative_videos.append(video_path)
    
    # Include all positive cases, limit negative cases by max_videos
    if max_videos is not None:
        # Limit negative videos, but keep all positive
        negative_videos = negative_videos[:max_videos]
    
    # Combine: all positives first, then limited negatives
    video_files = sorted(positive_videos) + sorted(negative_videos)
    
    num_positive = len(positive_videos)
    num_negative = len(negative_videos)
    
    print(f"Processing {len(video_files)} videos...")
    print(f"  Positive cases (Ken Burns): {num_positive} (all included)")
    if max_videos is not None:
        total_negative = len([v for v in all_video_files if v.stem not in labels or not labels.get(v.stem, False)])
        print(f"  Negative cases (Not Ken Burns): {num_negative} (limited from {total_negative})")
    else:
        print(f"  Negative cases (Not Ken Burns): {num_negative} (all included)")
    if frame_skip > 1:
        print(f"  Temporal downsampling: taking every {frame_skip} frames")
    if scale_factor is not None:
        print(f"  Spatial downsampling: scale factor = {scale_factor}")
    if target_width is not None:
        print(f"  Spatial downsampling: target width = {target_width} pixels")
    if max_frames is not None:
        print(f"  Max frames per video: {max_frames}")
    if not force_reprocess:
        print(f"  Using cache: {cache_file}")
    print()
    
    for idx, video_path in enumerate(video_files, 1):
        file_name = video_path.stem  # filename without extension
        video_start_time = time.time()
        
        # Check cache
        cache_key = get_cache_key(str(video_path), method, ransac_threshold,
                                 frame_skip, max_frames, scale_factor, target_width)
        
        if cache_key in cache and not force_reprocess:
            print(f"[{idx}/{len(video_files)}] Loading {file_name} from cache...")
            result = cache[cache_key].copy()
            processing_time = 0.0  # No processing time for cached results
        else:
            print(f"[{idx}/{len(video_files)}] Processing {file_name}...")
            
            try:
                load_start = time.time()
                result = process_video_homography(str(video_path), 
                                                method=method,
                                                ransac_threshold=ransac_threshold,
                                                frame_skip=frame_skip,
                                                max_frames=max_frames,
                                                scale_factor=scale_factor,
                                                target_width=target_width)
                processing_time = time.time() - load_start
                
                # Save to cache
                cache[cache_key] = result.copy()
                cache_updated = True
            
                # Add label
                result['is_ken_burn'] = labels.get(file_name, None)
                result['label_available'] = file_name in labels
                
                total_time = time.time() - video_start_time
                
                # Print comprehensive metrics
                print_metrics(result, processing_time, total_time)
                
            except Exception as e:
                total_time = time.time() - video_start_time
                print(f"  ✗ Error processing {file_name}: {e}")
                print(f"  ⏱  Time before error: {total_time:.2f}s")
                print()
                result = {
                    'file_name': file_name,
                    'num_frames': 0,
                    'num_pairs': 0,
                    'mean_error': float('inf'),
                    'std_error': 0,
                    'max_error': float('inf'),
                    'min_error': float('inf'),
                    'mean_inliers': 0,
                    'mean_inlier_ratio': 0.0,
                    'std_inlier_ratio': 0.0,
                    'min_inlier_ratio': 0.0,
                    'max_inlier_ratio': 0.0,
                    'rotation_variance': float('inf'),
                    'rotation_first_deriv_variance': float('inf'),
                    'rotation_second_deriv_variance': float('inf'),
                    'rotation_max_first_deriv': float('inf'),
                    'scale_variance': float('inf'),
                    'scale_first_deriv_variance': float('inf'),
                    'scale_second_deriv_variance': float('inf'),
                    'scale_max_first_deriv': float('inf'),
                    'tx_variance': float('inf'),
                    'tx_first_deriv_variance': float('inf'),
                    'tx_second_deriv_variance': float('inf'),
                    'tx_max_first_deriv': float('inf'),
                    'ty_variance': float('inf'),
                    'ty_first_deriv_variance': float('inf'),
                    'ty_second_deriv_variance': float('inf'),
                    'ty_max_first_deriv': float('inf'),
                    'mean_outliers': 0.0,
                    'std_outliers': 0.0,
                    'mean_inlier_outlier_ratio': 0.0,
                    'std_inlier_outlier_ratio': 0.0,
                    'min_inlier_outlier_ratio': 0.0,
                    'max_inlier_outlier_ratio': 0.0,
                    'is_ken_burn': labels.get(file_name, None),
                    'label_available': file_name in labels
                }
        
        # Add label (in case it wasn't added above)
        if 'is_ken_burn' not in result:
            result['is_ken_burn'] = labels.get(file_name, None)
            result['label_available'] = file_name in labels
        
        # Print metrics for cached results too
        if cache_key in cache and not force_reprocess:
            print_metrics(result, 0.0, time.time() - video_start_time, from_cache=True)
        
        results.append(result)
    
    # Save cache if updated
    if cache_updated:
        save_cache(cache, cache_file)
        print(f"Cache updated: {cache_file}")
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    return df


def find_optimal_threshold(df: pd.DataFrame, metric: str = 'mean_error', 
                          higher_is_ken_burns: bool = False) -> dict:
    """
    Find optimal threshold for separating Ken Burns from non-Ken Burns videos.
    
    Args:
        df: DataFrame with results
        metric: Metric to use for thresholding ('mean_error', 'max_error', 'std_error', etc.)
        higher_is_ken_burns: If True, higher values indicate Ken Burns (e.g., inlier_ratio).
                            If False, lower values indicate Ken Burns (e.g., error).
        
    Returns:
        Dictionary with threshold information
    """
    # Check if metric exists in dataframe
    if metric not in df.columns:
        return {'threshold': None, 'accuracy': 0, 'error': f'Metric {metric} not found in dataframe'}
    
    # Filter to only labeled videos with valid values
    if metric in ['mean_inlier_ratio', 'mean_inlier_outlier_ratio']:
        labeled_df = df[df['label_available'] & (df[metric] != 0.0) & (df[metric].notna()) & (df[metric] != float('inf'))].copy()
    else:
        labeled_df = df[df['label_available'] & (df[metric] != float('inf')) & (df[metric].notna())].copy()
    
    if len(labeled_df) == 0:
        return {'threshold': None, 'accuracy': 0, 'error': 'No valid data'}
    
    ken_burns = labeled_df[labeled_df['is_ken_burn'] == True][metric].values
    non_ken_burns = labeled_df[labeled_df['is_ken_burn'] == False][metric].values
    
    if len(ken_burns) == 0 or len(non_ken_burns) == 0:
        return {'threshold': None, 'accuracy': 0, 'error': 'Missing class data'}
    
    # Try different thresholds
    min_val = labeled_df[metric].min()
    max_val = labeled_df[metric].max()
    
    thresholds = np.linspace(min_val, max_val, 1000)
    best_threshold = None
    best_accuracy = 0
    best_stats = None
    
    for threshold in thresholds:
        # Predict based on direction
        if higher_is_ken_burns:
            # Higher values = Ken Burns (True)
            predictions = labeled_df[metric] >= threshold
        else:
            # Lower values = Ken Burns (True)
            predictions = labeled_df[metric] <= threshold
        accuracy = (predictions == labeled_df['is_ken_burn']).mean()
        
        tp = ((predictions == True) & (labeled_df['is_ken_burn'] == True)).sum()
        tn = ((predictions == False) & (labeled_df['is_ken_burn'] == False)).sum()
        fp = ((predictions == True) & (labeled_df['is_ken_burn'] == False)).sum()
        fn = ((predictions == False) & (labeled_df['is_ken_burn'] == True)).sum()
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
            best_stats = {
                'accuracy': accuracy,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn,
                'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            }
    
    return {
        'threshold': best_threshold,
        'metric': metric,
        **best_stats
    }


def filter_outliers(data: pd.Series, lower_percentile: float = 5, upper_percentile: float = 95):
    """
    Filter extreme outliers using percentile-based limits.
    
    Args:
        data: Series of data values
        lower_percentile: Lower percentile to use as limit
        upper_percentile: Upper percentile to use as limit
        
    Returns:
        Filtered data and limits
    """
    valid_data = data[data != float('inf')]
    if len(valid_data) == 0:
        return data, None, None
    
    lower_limit = np.percentile(valid_data, lower_percentile)
    upper_limit = np.percentile(valid_data, upper_percentile)
    
    # Filter data within limits
    filtered = data[(data >= lower_limit) & (data <= upper_limit) & (data != float('inf'))]
    
    return filtered, lower_limit, upper_limit


def visualize_results(df: pd.DataFrame, output_dir: str = '.'):
    """
    Create visualizations of the analysis results.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    labeled_df = df[df['label_available'] & (df['mean_error'] != float('inf'))].copy()
    
    if len(labeled_df) == 0:
        print("No valid data to visualize")
        return
    
    # Create figure with subplots (3x2 layout for more plots)
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    axes = axes.flatten()
    
    # 1. Mean error distribution (with outlier filtering)
    ax = axes[0]
    ken_burns_raw = labeled_df[labeled_df['is_ken_burn'] == True]['mean_error']
    non_ken_burns_raw = labeled_df[labeled_df['is_ken_burn'] == False]['mean_error']
    
    ken_burns, _, _ = filter_outliers(ken_burns_raw)
    non_ken_burns, _, _ = filter_outliers(non_ken_burns_raw)
    
    ax.hist(non_ken_burns, bins=30, alpha=0.7, label='Not Ken Burns', color='red')
    ax.hist(ken_burns, bins=30, alpha=0.7, label='Ken Burns', color='blue')
    ax.set_xlabel('Mean Reprojection Error (pixels)')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Mean Reprojection Error (outliers filtered)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box plot comparison (with outlier filtering)
    ax = axes[1]
    ken_burns_filtered, kb_lower, kb_upper = filter_outliers(ken_burns_raw)
    non_kb_filtered, nkb_lower, nkb_upper = filter_outliers(non_ken_burns_raw)
    
    data_to_plot = [non_kb_filtered, ken_burns_filtered]
    bp = ax.boxplot(data_to_plot, labels=['Not Ken Burns', 'Ken Burns'], showfliers=False)
    ax.set_ylabel('Mean Reprojection Error (pixels)')
    ax.set_title('Error Comparison (outliers filtered)')
    ax.grid(True, alpha=0.3)
    
    # 3. Scatter plot: mean error vs mean inlier ratio (with outlier filtering)
    ax = axes[2]
    for is_kb, color, label in [(False, 'red', 'Not Ken Burns'), (True, 'blue', 'Ken Burns')]:
        subset = labeled_df[labeled_df['is_ken_burn'] == is_kb]
        error_filtered, _, _ = filter_outliers(subset['mean_error'])
        ratio_filtered, _, _ = filter_outliers(subset['mean_inlier_ratio'])
        
        # Align indices
        common_idx = error_filtered.index.intersection(ratio_filtered.index)
        ax.scatter(error_filtered[common_idx], ratio_filtered[common_idx], 
                  alpha=0.6, color=color, label=label, s=50)
    ax.set_xlabel('Mean Error (pixels)')
    ax.set_ylabel('Mean Inlier Ratio')
    ax.set_title('Mean Error vs Mean Inlier Ratio (outliers filtered)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Scatter plot: mean inlier ratio vs mean inlier/outlier ratio
    ax = axes[3]
    for is_kb, color, label in [(False, 'red', 'Not Ken Burns'), (True, 'blue', 'Ken Burns')]:
        subset = labeled_df[labeled_df['is_ken_burn'] == is_kb]
        ratio_filtered, _, _ = filter_outliers(subset['mean_inlier_ratio'])
        io_ratio_filtered, _, _ = filter_outliers(subset['mean_inlier_outlier_ratio'])
        
        # Align indices
        common_idx = ratio_filtered.index.intersection(io_ratio_filtered.index)
        ax.scatter(ratio_filtered[common_idx], io_ratio_filtered[common_idx], 
                  alpha=0.6, color=color, label=label, s=50)
    ax.set_xlabel('Mean Inlier Ratio')
    ax.set_ylabel('Mean Inlier/Outlier Ratio')
    ax.set_title('Inlier Ratio vs Inlier/Outlier Ratio (outliers filtered)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Mean outliers comparison
    ax = axes[4]
    ken_burns_outliers = labeled_df[labeled_df['is_ken_burn'] == True]['mean_outliers']
    non_ken_burns_outliers = labeled_df[labeled_df['is_ken_burn'] == False]['mean_outliers']
    
    ken_burns_outliers_filt, _, _ = filter_outliers(ken_burns_outliers)
    non_ken_burns_outliers_filt, _, _ = filter_outliers(non_ken_burns_outliers)
    
    data_to_plot = [non_ken_burns_outliers_filt, ken_burns_outliers_filt]
    bp = ax.boxplot(data_to_plot, labels=['Not Ken Burns', 'Ken Burns'], showfliers=False)
    ax.set_ylabel('Mean Outliers')
    ax.set_title('Mean Outliers Comparison (outliers filtered)')
    ax.grid(True, alpha=0.3)
    
    # 6. Mean inlier/outlier ratio comparison
    ax = axes[5]
    ken_burns_io_ratio = labeled_df[labeled_df['is_ken_burn'] == True]['mean_inlier_outlier_ratio']
    non_ken_burns_io_ratio = labeled_df[labeled_df['is_ken_burn'] == False]['mean_inlier_outlier_ratio']
    
    ken_burns_io_filt, _, _ = filter_outliers(ken_burns_io_ratio)
    non_ken_burns_io_filt, _, _ = filter_outliers(non_ken_burns_io_ratio)
    
    data_to_plot = [non_ken_burns_io_filt, ken_burns_io_filt]
    bp = ax.boxplot(data_to_plot, labels=['Not Ken Burns', 'Ken Burns'], showfliers=False)
    ax.set_ylabel('Mean Inlier/Outlier Ratio')
    ax.set_title('Inlier/Outlier Ratio Comparison (outliers filtered)')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save as PNG
    output_path = os.path.join(output_dir, 'ken_burns_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    
    # Save as pickle for interactive viewing
    pickle_path = os.path.join(output_dir, 'ken_burns_analysis.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(fig, f)
    print(f"Saved interactive figure to {pickle_path} (load with: pickle.load(open('{pickle_path}', 'rb')))")
    
    plt.close()


def plot_homography_parameters(df: pd.DataFrame, output_dir: str = '.'):
    """
    Plot homography parameters (rotation, scale, translation) over time for each video.
    
    Args:
        df: DataFrame with results
        output_dir: Directory to save plots
    """
    # Filter to videos with valid homography data
    valid_df = df[df['rotations_deg'].notna()].copy()
    
    if len(valid_df) == 0:
        print("No valid homography parameter data to plot")
        return
    
    # Create a plot for each video (or a subset)
    max_plots = min(12, len(valid_df))  # Plot up to 12 videos
    n_cols = 3
    n_rows = (max_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(valid_df.head(max_plots).iterrows()):
        ax = axes[idx]
        
        rotations = [r for r in row['rotations_deg'] if r is not None]
        scales = [s for s in row['scales'] if s is not None]
        tx = [x for x in row['translations_x'] if x is not None]
        ty = [y for y in row['translations_y'] if y is not None]
        
        if len(rotations) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{row['file_name']}\n({'Ken Burns' if row['is_ken_burn'] else 'Not Ken Burns'})")
            continue
        
        frame_indices = np.arange(len(rotations))
        
        # Plot rotation
        ax2 = ax.twinx()
        ax.plot(frame_indices, rotations, 'b-', label='Rotation (deg)', linewidth=1.5, alpha=0.7)
        ax2.plot(frame_indices, scales, 'r-', label='Scale', linewidth=1.5, alpha=0.7)
        
        # Plot translation on a separate y-axis or normalize
        tx_normalized = np.array(tx) / (np.max(np.abs(tx)) + 1e-6) if len(tx) > 0 and np.max(np.abs(tx)) > 0 else []
        ty_normalized = np.array(ty) / (np.max(np.abs(ty)) + 1e-6) if len(ty) > 0 and np.max(np.abs(ty)) > 0 else []
        
        if len(tx_normalized) > 0:
            ax.plot(frame_indices, tx_normalized, 'g--', label='Tx (norm)', linewidth=1, alpha=0.5)
        if len(ty_normalized) > 0:
            ax.plot(frame_indices, ty_normalized, 'm--', label='Ty (norm)', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Frame Pair Index')
        ax.set_ylabel('Rotation (deg)', color='b')
        ax2.set_ylabel('Scale', color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        kb_label = 'Ken Burns' if row['is_ken_burn'] else 'Not Ken Burns'
        ax.set_title(f"{row['file_name']}\n({kb_label})", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=7)
        ax2.legend(loc='upper right', fontsize=7)
    
    # Hide unused subplots
    for idx in range(max_plots, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    # Save as PNG
    output_path = os.path.join(output_dir, 'homography_parameters.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved homography parameters plot to {output_path}")
    
    # Save as pickle for interactive viewing
    pickle_path = os.path.join(output_dir, 'homography_parameters.pkl')
    with open(pickle_path, 'wb') as f:
        pickle.dump(fig, f)
    print(f"Saved interactive figure to {pickle_path}")
    
    plt.close()


def main():
    """Main analysis pipeline."""
    dataset_dir = 'dataset'
    label_file = 'label_combined.csv'
    
    # Downsampling parameters
    frame_skip = 5 
    target_width = 640  # Resize to 640 pixels width
    max_frames = 100  # No limit on number of frames
    max_videos = None  # Process only 3 videos for testing
    method = 'SIFT'
    ransac_threshold = 0.5
    
    print("=" * 60)
    print("Ken Burns Effect Detection via Homography Analysis")
    print("=" * 60)
    
    # Process all videos (with caching)
    # Set force_reprocess=True to bypass cache and reprocess all videos
    force_reprocess = True  # Change to True to ignore cache
    
    df = analyze_all_videos(dataset_dir, label_file, 
                           method=method, 
                           ransac_threshold=ransac_threshold,
                           frame_skip=frame_skip,
                           max_frames=max_frames,
                           target_width=target_width,
                           max_videos=max_videos,
                           cache_file='homography_cache.pkl',
                           force_reprocess=force_reprocess)
    
    # Save results
    output_csv = 'homography_results.csv'
    df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")
    
    # Find optimal threshold
    print("\n" + "=" * 60)
    print("Finding Optimal Threshold")
    print("=" * 60)
    
    # Metrics where lower values indicate Ken Burns (smoothness metrics)
    smoothness_metrics = [
        'rotation_first_deriv_variance', 'rotation_second_deriv_variance',
        'scale_first_deriv_variance', 'scale_second_deriv_variance',
        'tx_first_deriv_variance', 'ty_first_deriv_variance'
    ]
    
    # Standard metrics
    # mean_inlier_outlier_ratio: higher is better (more inliers relative to outliers = Ken Burns)
    metrics = ['mean_error', 'max_error', 'std_error', 'min_error', 'mean_inlier_ratio', 
               'mean_inlier_outlier_ratio', 'mean_outliers'] + smoothness_metrics
    best_overall = None
    best_accuracy = 0
    
    for metric in metrics:
        # Inlier ratio and inlier/outlier ratio: higher is better (indicates Ken Burns)
        # Mean outliers: lower is better (fewer outliers = Ken Burns)
        # Smoothness metrics: lower is better (smoother = Ken Burns)
        higher_is_ken_burns = (metric in ['mean_inlier_ratio', 'mean_inlier_outlier_ratio'])
        threshold_info = find_optimal_threshold(df, metric=metric, 
                                               higher_is_ken_burns=higher_is_ken_burns)
        if threshold_info.get('threshold') is not None:
            print(f"\n{metric.upper()}:")
            print(f"  Optimal threshold: {threshold_info['threshold']:.4f}")
            print(f"  Accuracy: {threshold_info['accuracy']:.4f} ({threshold_info['accuracy']*100:.2f}%)")
            print(f"  Precision: {threshold_info['precision']:.4f}")
            print(f"  Recall: {threshold_info['recall']:.4f}")
            print(f"  TP: {threshold_info['true_positives']}, TN: {threshold_info['true_negatives']}")
            print(f"  FP: {threshold_info['false_positives']}, FN: {threshold_info['false_negatives']}")
            
            if threshold_info['accuracy'] > best_accuracy:
                best_accuracy = threshold_info['accuracy']
                best_overall = (metric, threshold_info)
    
    if best_overall:
        print(f"\n{'='*60}")
        print(f"BEST METRIC: {best_overall[0].upper()}")
        print(f"Threshold: {best_overall[1]['threshold']:.4f}")
        print(f"Accuracy: {best_overall[1]['accuracy']*100:.2f}%")
        print(f"{'='*60}")
    
    # Visualize results
    print("\nGenerating visualizations...")
    visualize_results(df)
    plot_homography_parameters(df)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    labeled_df = df[df['label_available'] & (df['mean_error'] != float('inf'))]
    
    if len(labeled_df) > 0:
        ken_burns = labeled_df[labeled_df['is_ken_burn'] == True]
        non_ken_burns = labeled_df[labeled_df['is_ken_burn'] == False]
        
        print(f"\nKen Burns videos (n={len(ken_burns)}):")
        print(f"  Mean error: {ken_burns['mean_error'].mean():.4f} ± {ken_burns['mean_error'].std():.4f}")
        print(f"  Median error: {ken_burns['mean_error'].median():.4f}")
        print(f"  Range: [{ken_burns['mean_error'].min():.4f}, {ken_burns['mean_error'].max():.4f}]")
        
        print(f"\nNon-Ken Burns videos (n={len(non_ken_burns)}):")
        print(f"  Mean error: {non_ken_burns['mean_error'].mean():.4f} ± {non_ken_burns['mean_error'].std():.4f}")
        print(f"  Median error: {non_ken_burns['mean_error'].median():.4f}")
        print(f"  Range: [{non_ken_burns['mean_error'].min():.4f}, {non_ken_burns['mean_error'].max():.4f}]")


if __name__ == '__main__':
    main()

