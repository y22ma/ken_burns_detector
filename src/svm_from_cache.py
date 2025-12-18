#!/usr/bin/env python3
"""
Train/evaluate an SVM classifier using metrics stored in homography_cache.pkl,
and compare against a best single-metric threshold baseline.

This script does NOT reprocess videos; it relies only on cached results.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from utils import parse_labels


DEFAULT_FEATURE_COLS: List[str] = [
    # Reprojection error
    "mean_error",
    "std_error",
    "min_error",
    "max_error",
    # Inlier/outlier aggregates
    "mean_inliers",
    "mean_outliers",
    "mean_inlier_ratio",
    "std_inlier_ratio",
    "min_inlier_ratio",
    "max_inlier_ratio",
    "mean_inlier_outlier_ratio",
    "std_inlier_outlier_ratio",
    "min_inlier_outlier_ratio",
    "max_inlier_outlier_ratio",
    # Smoothness (rotation/scale/translation)
    "rotation_variance",
    "rotation_first_deriv_variance",
    "rotation_second_deriv_variance",
    "rotation_max_first_deriv",
    "scale_variance",
    "scale_first_deriv_variance",
    "scale_second_deriv_variance",
    "scale_max_first_deriv",
    "tx_variance",
    "tx_first_deriv_variance",
    "tx_second_deriv_variance",
    "tx_max_first_deriv",
    "ty_variance",
    "ty_first_deriv_variance",
    "ty_second_deriv_variance",
    "ty_max_first_deriv",
]


DEFAULT_BASELINE_METRICS: List[Tuple[str, bool]] = [
    # (metric_name, higher_is_positive)
    ("mean_error", False),
    ("std_error", False),
    ("min_error", False),
    ("max_error", False),
    ("mean_inlier_ratio", True),
    ("mean_inlier_outlier_ratio", True),
    ("mean_outliers", False),
    ("rotation_first_deriv_variance", False),
    ("rotation_second_deriv_variance", False),
    ("scale_first_deriv_variance", False),
    ("scale_second_deriv_variance", False),
    ("tx_first_deriv_variance", False),
    ("ty_first_deriv_variance", False),
]


def _safe_stem(file_name: str) -> str:
    return Path(str(file_name)).stem


def load_cache(cache_path: Path) -> Dict:
    with cache_path.open("rb") as f:
        return pickle.load(f)


def cache_to_dataframe(cache: Dict) -> pd.DataFrame:
    """Convert cache dict (key -> result dict) into a deduplicated DataFrame (1 row per video)."""
    rows = []
    for _, result in cache.items():
        if not isinstance(result, dict):
            continue
        r = dict(result)
        file_name = r.get("file_name", None)
        if file_name is None:
            continue
        r["file_stem"] = _safe_stem(file_name)
        r["_num_keys"] = len(r.keys())
        # Prefer finite mean_error if present
        me = r.get("mean_error", np.inf)
        r["_mean_error_is_finite"] = float(np.isfinite(me))
        r["_num_pairs"] = r.get("num_pairs", 0) or 0
        rows.append(r)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Deduplicate by file_stem: prefer most-complete rows with more pairs and finite error
    sort_cols = ["file_stem", "_num_keys", "_mean_error_is_finite", "_num_pairs"]
    df = df.sort_values(sort_cols, ascending=[True, False, False, False])
    df = df.drop_duplicates(subset=["file_stem"], keep="first").reset_index(drop=True)
    df = df.drop(columns=["_num_keys", "_mean_error_is_finite", "_num_pairs"], errors="ignore")
    return df


def attach_labels(df: pd.DataFrame, label_file: Path) -> pd.DataFrame:
    labels = parse_labels(str(label_file))
    df = df.copy()
    df["label_available"] = df["file_stem"].astype(str).map(lambda s: s in labels)
    df["is_ken_burn"] = df["file_stem"].astype(str).map(labels).astype("boolean")
    return df


def build_feature_table(
    df: pd.DataFrame,
    feature_cols: List[str],
    fill_inf_with_nan: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = np.nan
    X = out[feature_cols].copy()
    if fill_inf_with_nan:
        X = X.replace([np.inf, -np.inf], np.nan)
    out[feature_cols] = X
    return out


def pick_best_single_metric_threshold(
    train_df: pd.DataFrame,
    metric_specs: List[Tuple[str, bool]],
    y_col: str = "is_ken_burn",
) -> Dict:
    """Pick best metric+threshold on TRAIN only, using balanced accuracy."""
    best = {
        "metric": None,
        "threshold": None,
        "higher_is_positive": None,
        "train_balanced_accuracy": -1.0,
    }
    y = train_df[y_col].astype(int).values

    for metric, higher_is_positive in metric_specs:
        if metric not in train_df.columns:
            continue
        vals = train_df[metric].replace([np.inf, -np.inf], np.nan).astype(float)
        mask = ~vals.isna()
        if mask.sum() < 5:
            continue

        v = vals[mask].values
        yy = y[mask.values]

        # Candidate thresholds: quantiles to keep it stable with small N
        thresholds = np.unique(np.quantile(v, np.linspace(0.0, 1.0, 101)))
        for t in thresholds:
            if higher_is_positive:
                pred = (v >= t).astype(int)
            else:
                pred = (v <= t).astype(int)
            bacc = balanced_accuracy_score(yy, pred)
            if bacc > best["train_balanced_accuracy"]:
                best.update(
                    {
                        "metric": metric,
                        "threshold": float(t),
                        "higher_is_positive": higher_is_positive,
                        "train_balanced_accuracy": float(bacc),
                        "train_n": int(mask.sum()),
                    }
                )

    return best


def eval_single_metric_threshold(
    df: pd.DataFrame,
    metric: str,
    threshold: float,
    higher_is_positive: bool,
    y_true: np.ndarray,
) -> Dict:
    vals = df[metric].replace([np.inf, -np.inf], np.nan).astype(float).values
    # If NaN: predict negative (conservative)
    pred = np.zeros_like(y_true)
    ok = np.isfinite(vals)
    if higher_is_positive:
        pred[ok] = (vals[ok] >= threshold).astype(int)
    else:
        pred[ok] = (vals[ok] <= threshold).astype(int)
    return classification_summary(y_true, pred)


def classification_summary(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def train_and_eval_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int,
) -> Tuple[Pipeline, Dict]:
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svc", SVC(class_weight="balanced")),
        ]
    )

    # Keep CV small because positives are few.
    n_pos = int(y_train.sum())
    n_splits = 3 if n_pos >= 3 else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    param_grid = [
        {"svc__kernel": ["linear"], "svc__C": [0.1, 1.0, 10.0, 100.0]},
        {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1.0, 10.0, 100.0],
            "svc__gamma": ["scale", 0.01, 0.1, 1.0],
        },
    ]

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    gs.fit(X_train, y_train)

    best_model: Pipeline = gs.best_estimator_
    y_pred = best_model.predict(X_test)
    summary = classification_summary(y_test, y_pred)
    summary["best_params"] = gs.best_params_
    summary["cv_best_score_balanced_accuracy"] = float(gs.best_score_)
    summary["classification_report"] = classification_report(
        y_test, y_pred, digits=4, zero_division=0
    )
    return best_model, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=str, default="homography_cache.pkl")
    ap.add_argument("--labels", type=str, default="label.csv")
    ap.add_argument("--out_csv", type=str, default="svm_features_from_cache.csv")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cache_path = Path(args.cache)
    label_path = Path(args.labels)

    if not cache_path.exists():
        raise FileNotFoundError(f"Cache not found: {cache_path}")
    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    cache = load_cache(cache_path)
    df = cache_to_dataframe(cache)
    if df.empty:
        raise RuntimeError("Cache loaded but no usable entries found.")

    df = attach_labels(df, label_path)
    df = df[df["label_available"]].copy()
    df = build_feature_table(df, DEFAULT_FEATURE_COLS)

    # Save feature table
    out_csv = Path(args.out_csv)
    df_out = df[["file_stem", "is_ken_burn", "label_available"] + DEFAULT_FEATURE_COLS].copy()
    df_out.to_csv(out_csv, index=False)
    print(f"Saved features CSV: {out_csv} (rows={len(df_out)})")

    # Prepare X/y
    y = df["is_ken_burn"].astype(int).values
    X = df[DEFAULT_FEATURE_COLS].replace([np.inf, -np.inf], np.nan).values

    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())
    print(f"Labeled samples: {len(y)} (pos={n_pos}, neg={n_neg})")
    if n_pos < 2:
        raise RuntimeError("Not enough positive samples to train/test an SVM.")

    if args.test_size > 0:
        X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
            X,
            y,
            df,
            test_size=args.test_size,
            random_state=args.seed,
            stratify=y,
        )
    else:
        X_train = X
        y_train = y
        df_train = df
        X_test = None
        y_test = None
        df_test = None
        
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y,
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y,
    )
    print(f"Split: train={len(y_train)} (pos={int(y_train.sum())}) | test={len(y_test)} (pos={int(y_test.sum())})")

    # Baseline: best single-metric threshold (fit on train only)
    best_thr = pick_best_single_metric_threshold(df_train, DEFAULT_BASELINE_METRICS)
    if best_thr["metric"] is None:
        raise RuntimeError("Could not find a usable single-metric threshold baseline.")
    base_summary = eval_single_metric_threshold(
        df_test,
        metric=best_thr["metric"],
        threshold=best_thr["threshold"],
        higher_is_positive=best_thr["higher_is_positive"],
        y_true=y_test,
    )

    print("\n=== Baseline: best single-metric threshold (trained on TRAIN) ===")
    print(f"metric={best_thr['metric']}  threshold={best_thr['threshold']:.6f}  higher_is_positive={best_thr['higher_is_positive']}")
    print(f"train_balanced_accuracy={best_thr['train_balanced_accuracy']:.4f}  train_n={best_thr.get('train_n', 'NA')}")
    print(f"test_balanced_accuracy={base_summary['balanced_accuracy']:.4f}  test_f1={base_summary['f1']:.4f}")
    print(f"confusion_matrix={base_summary['confusion_matrix']}")

    # SVM
    print("\n=== SVM (GridSearchCV, scoring=balanced_accuracy) ===")
    model, svm_summary = train_and_eval_svm(X_train, y_train, X_test, y_test, args.seed)
    print(f"best_params={svm_summary['best_params']}")
    print(f"cv_best_balanced_accuracy={svm_summary['cv_best_score_balanced_accuracy']:.4f}")
    print(f"test_balanced_accuracy={svm_summary['balanced_accuracy']:.4f}  test_f1={svm_summary['f1']:.4f}")
    print(f"confusion_matrix={svm_summary['confusion_matrix']}")
    print("\nClassification report:\n")
    print(svm_summary["classification_report"])


if __name__ == "__main__":
    main()



