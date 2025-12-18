# Ken Burns Effect Detection via Homography Analysis

## Overview

This project implements a computer vision-based approach to detect Ken Burns effects in video sequences. Ken Burns effects are characterized by smooth camera movements (zooms, pans, translations) applied to static images, creating a cinematic effect. The detection method uses homography estimation between consecutive frames to analyze motion characteristics and distinguish Ken Burns effects from dynamic scenes with moving objects.

## Core Hypothesis

Ken Burns effects exhibit:
1. **Low reprojection error**: Feature correspondences between frames should fit well to a planar homography model
2. **High inlier ratio**: Most feature matches should be consistent with the estimated homography
3. **Smooth parameter evolution**: Homography parameters (rotation, scale, translation) should change smoothly over time

In contrast, dynamic scenes with moving objects show:
1. **Higher reprojection error**: Moving objects violate the planar homography assumption
2. **Lower inlier ratio**: Many feature matches are outliers due to object motion
3. **Erratic parameter changes**: Homography parameters jump around due to inconsistent feature matches

## Methodology

### 1. Video Preprocessing

- **Spatial downsampling**: Videos are resized to 640 pixels width (maintaining aspect ratio) to reduce computational cost
- **Temporal sampling**: All frames are processed (no frame skipping by default)
- **Format**: Frames are converted to grayscale for feature detection

### 2. Feature Detection and Matching

- **Feature detector**: ORB (Oriented FAST and Rotated BRIEF) or SIFT
- **Matching**: Brute-force matcher with ratio test (Lowe's ratio test for SIFT, 0.75 threshold for ORB)
- **Filtering**: Only high-quality matches are retained for homography estimation

### 3. Homography Estimation

For each consecutive frame pair:
- Extract feature correspondences
- Estimate homography matrix using RANSAC (Random Sample Consensus)
- **RANSAC parameters**:
  - Reprojection threshold: 2.0 pixels (configurable)
  - Max iterations: 2000
  - Confidence: 0.99

### 4. Metrics Computed

#### 4.1 Reprojection Error Metrics
- **Mean reprojection error**: Average pixel distance between projected and actual feature points
- **Standard deviation**: Consistency of errors across frame pairs
- **Min/Max error**: Range of error values

#### 4.2 Inlier/Outlier Analysis
- **Number of inliers**: Feature matches consistent with homography
- **Number of outliers**: Feature matches rejected by RANSAC
- **Inlier ratio**: Proportion of matches that are inliers (inliers / total matches)
- **Inlier/Outlier ratio**: Ratio of inliers to outliers (higher = better for Ken Burns)

#### 4.3 Homography Parameter Smoothness

The homography matrix is decomposed into:
- **Rotation** (θ): Angle of rotation in degrees
- **Scale** (s): Uniform scaling factor
- **Translation** (tx, ty): Horizontal and vertical translation in pixels

For each parameter, smoothness is quantified by:
- **Variance**: Overall variability of the parameter
- **First derivative variance**: Smoothness of parameter changes (rate of change)
- **Second derivative variance**: Smoothness of acceleration (rate of change of rate of change)
- **Max first derivative**: Maximum instantaneous change

### 5. Classification Strategy

Multiple metrics are evaluated to find optimal thresholds:

**Metrics where lower values indicate Ken Burns:**
- Mean reprojection error
- Max/Std/Min reprojection error
- Smoothness metrics (rotation, scale, translation variance and derivatives)
- Mean outliers

**Metrics where higher values indicate Ken Burns:**
- Mean inlier ratio
- Mean inlier/outlier ratio

The system tests all metrics and selects the one with highest classification accuracy.

## Implementation Details

### Caching System

- Results are cached based on video file path, modification time, size, and processing parameters
- Prevents reprocessing of unchanged videos
- Cache key includes: method, ransac_threshold, frame_skip, max_frames, scale_factor, target_width

### SVM Classifier (Cache-Only Training/Eval)

In addition to single-metric thresholding, we support training and evaluating an SVM classifier using the **cached per-video metrics**.

- **Script**: `svm_from_cache.py`
- **Inputs**:
  - `homography_cache.pkl` (produced by running `analyze_ken_burns.py`)
  - a label file (e.g. `label.csv` or `label_combined.csv`)
- **Features used**: see `DEFAULT_FEATURE_COLS` in `svm_from_cache.py` (error stats, inlier/outlier stats, and smoothness stats).
- **Baseline comparison**: the script also picks the best single-metric threshold on the training split and compares SVM vs baseline on the test split.
- **Note on imbalance**: positives are rare in the original dataset, so single train/test splits can be unstable. Consider multiple seeds / repeated splits when interpreting results.

### Data Processing

- **Positive case handling**: All Ken Burns videos (positive cases) are always included
- **Negative case limiting**: The `max_videos` parameter only limits negative cases to ensure balanced datasets
- **Error handling**: Failed homography computations are marked with infinite error values

### Visualization

- **Static plots**: PNG files showing distributions and comparisons
- **Interactive plots**: Pickle files for zoom/pan exploration
- **Parameter evolution**: Time-series plots of homography parameters

### Debug Visualization (Inliers/Outliers + Error Heatmap)

To understand failure cases (e.g., 3D camera motion that still “looks planar” for many matches), we provide a visualization tool:

- **Script**: `visualize_inliers_outliers.py`
- **Outputs**:
  - match visualization: inliers (green), outliers (red)
  - error heatmap overlay (warped frame vs target frame) with optional inlier/outlier point overlay

### When to Generate Synthetic Ken Burns Data

Synthetic Ken Burns generation (`generate_synthetic_kenburns.py`) is used as **data augmentation** to get a better sense of what “ideal” Ken Burns looks like under our assumptions (smooth similarity motion: scale + translation + in-plane rotation on a still frame).

It’s especially helpful when the original dataset has **few positive examples**, since adding more positives makes it easier to train and debug classifiers and to validate that the homography/inlier/smoothness metrics behave as expected.

## Expected Results

### Ken Burns Effects Should Show:
- ✅ Low mean reprojection error (< 5 pixels typically)
- ✅ High inlier ratio (> 0.7 typically)
- ✅ High inlier/outlier ratio (> 2.0 typically)
- ✅ Low smoothness variance (gradual parameter changes)
- ✅ Low first/second derivative variance (smooth transitions)

### Dynamic Scenes Should Show:
- ❌ Higher reprojection error (> 10 pixels typically)
- ❌ Lower inlier ratio (< 0.5 typically)
- ❌ Lower inlier/outlier ratio (< 1.0 typically)
- ❌ Higher smoothness variance (erratic parameter changes)
- ❌ Higher derivative variance (abrupt transitions)

## Files

- `utils.py`: Core homography computation and video processing functions
- `analyze_ken_burns.py`: Main analysis pipeline with caching and visualization
- `load_plot.py`: Helper script to load and view interactive plots
- `svm_from_cache.py`: Train/evaluate an SVM using cached metrics and compare vs best single-metric baseline
- `visualize_inliers_outliers.py`: Visualize inlier/outlier matches and error heatmaps for a chosen frame pair
- `generate_synthetic_kenburns.py`: Generate synthetic positive Ken Burns clips from still frames using smooth similarity transforms
- `homography_cache.pkl`: Cached processing results
- `homography_results.csv`: Exported results in CSV format
- `ken_burns_analysis.png`: Static visualization of results
- `homography_parameters.png`: Parameter evolution plots
- `synthetic_dataset/`: Output directory for generated synthetic positive clips (`kb_syn_*.mp4`)
- `synthetic_label.csv`: Labels + generation parameters (JSON) for synthetic clips
- `label_combined.csv`: Combined labels for original dataset + synthetic positives

## Usage

```bash
# Run analysis
python analyze_ken_burns.py

# View interactive plots
python load_plot.py ken_burns_analysis.pkl
python load_plot.py homography_parameters.pkl

# Train/evaluate SVM from cached metrics (no video reprocessing)
python svm_from_cache.py --cache homography_cache.pkl --labels label.csv --out_csv svm_features_from_cache.csv

# Debug a specific video/frame pair with inliers/outliers + error heatmap
python visualize_inliers_outliers.py --video dataset/32.mp4 --frame_idx 10 --step 10 --method ORB --ransac_threshold 2.0 --error_overlay --points --points_on_error --out_dir viz_out

# Generate synthetic positive Ken Burns clips (writes to synthetic_dataset/ and synthetic_label.csv)
python generate_synthetic_kenburns.py --dataset_dir dataset --out_dir synthetic_dataset --labels_out synthetic_label.csv --num_clips 10 --num_frames 150 --fps 30 --target_width 640 --seed 123
```

## Configuration

Key parameters in `main()`:
- `method`: 'ORB' or 'SIFT' (default: 'ORB')
- `ransac_threshold`: RANSAC reprojection threshold in pixels (default: 2.0)
- `target_width`: Video width for downsampling (default: 640)
- `max_videos`: Maximum negative cases to process (default: 6)
- `force_reprocess`: Force reprocessing even if cached (default: False)

## Future Improvements

1. **Machine learning classifier**: Train a model using computed metrics
2. **Temporal consistency**: Analyze parameter trends over longer sequences
3. **Multi-scale analysis**: Process at multiple resolutions
4. **Optical flow integration**: Combine with dense optical flow for motion analysis
5. **Deep learning features**: Use learned features instead of hand-crafted ones

