"""Metric computation for categorization and forecasting benchmarks.

All metrics are computed from real model predictions. No placeholders.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


# ============================================================================
# Categorization Metrics
# ============================================================================

def compute_categorization_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    labels: Optional[list] = None,
) -> Dict[str, Any]:
    """Compute full categorization metric suite from real predictions."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics["confusion_matrix"] = cm

    # Classification report
    report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    metrics["classification_report"] = report

    return metrics


# ============================================================================
# Forecasting Metrics
# ============================================================================

def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring zero actuals."""
    mask = np.abs(y_true) > 1e-8
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def _safe_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric MAPE."""
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 1e-8
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(2.0 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]))


def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error."""
    total = np.sum(np.abs(y_true))
    if total < 1e-8:
        return 0.0
    return float(np.sum(np.abs(y_true - y_pred)) / total)


def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute full forecasting metric suite from real predictions."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) == 0:
        return {
            "mae": 0.0, "rmse": 0.0, "mape": 0.0, "wape": 0.0, "smape": 0.0, "r2": 0.0
        }

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) > 1 else 0.0

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "mape": round(_safe_mape(y_true, y_pred), 4),
        "wape": round(_wape(y_true, y_pred), 4),
        "smape": round(_safe_smape(y_true, y_pred), 4),
        "r2": round(r2, 4),
    }


# ============================================================================
# Calibration Metrics
# ============================================================================

def compute_calibration_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute calibration metrics: ECE, Brier score, reliability curve data."""
    n_classes = y_proba.shape[1] if y_proba.ndim > 1 else 2

    # ECE (Expected Calibration Error) - multiclass
    confidences = np.max(y_proba, axis=1) if y_proba.ndim > 1 else y_proba
    predictions = np.argmax(y_proba, axis=1) if y_proba.ndim > 1 else (y_proba > 0.5).astype(int)

    # Encode y_true to match predictions
    if isinstance(y_true[0], str):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_true_enc = le.fit_transform(y_true)
    else:
        y_true_enc = np.asarray(y_true)

    correctness = (predictions == y_true_enc).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi)
        count = mask.sum()
        bin_counts.append(int(count))
        if count > 0:
            bin_acc = float(correctness[mask].mean())
            bin_conf = float(confidences[mask].mean())
            bin_accuracies.append(bin_acc)
            bin_confidences.append(bin_conf)
            ece += (count / len(confidences)) * abs(bin_acc - bin_conf)
        else:
            bin_accuracies.append(0.0)
            bin_confidences.append(0.0)

    # Brier score (one-vs-rest average for multiclass)
    brier = 0.0
    if y_proba.ndim > 1:
        for c in range(n_classes):
            y_binary = (y_true_enc == c).astype(float)
            p_c = y_proba[:, c] if c < y_proba.shape[1] else np.zeros(len(y_true_enc))
            brier += brier_score_loss(y_binary, p_c)
        brier /= n_classes
    else:
        brier = float(brier_score_loss(y_true_enc, y_proba))

    return {
        "ece": round(float(ece), 4),
        "brier_score": round(float(brier), 4),
        "reliability_curve": {
            "bin_accuracies": bin_accuracies,
            "bin_confidences": bin_confidences,
            "bin_counts": bin_counts,
        },
    }
