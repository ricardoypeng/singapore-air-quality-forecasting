"""Shared six-hour MSE and trend diagnostics."""

from typing import Any
import numpy as np


def score_forecast(actual: np.ndarray, predicted: np.ndarray, current: np.ndarray) -> dict[str, Any]:
    """
    Report MSE as the common error metric and endpoint trend as diagnostics.

    Args:
        actual    (np.ndarray): True targets, shape (samples, horizon).
        predicted (np.ndarray): Completed predictions of identical shape.
        current   (np.ndarray): Last observed PM value per origin.

    Returns:
        dict[str, Any]: MSE, per-hour MSE, direction, amplitude and event diagnostics.
    """
    if actual.shape != predicted.shape or not np.isfinite(predicted).all() or not np.isfinite(actual).all():
        raise ValueError("Expected matching finite forecast and target arrays")
    squared = np.square(predicted - actual)
    if not np.isfinite(squared).all():
        raise ValueError("Squared forecast errors overflowed")
    a, p = actual[:, -1] - current, predicted[:, -1] - current
    large, triggered = np.abs(a) > 10, np.abs(p) > 10
    hits = int(np.sum(large & triggered))
    return {"mse": float(squared.mean()), "mae": float(np.abs(predicted - actual).mean()),
            "rmse": float(np.sqrt(squared.mean())),
            "mean_horizon_rmse": float(np.sqrt(squared.mean(axis=0)).mean()),
            "per_horizon_mae": np.abs(predicted - actual).mean(axis=0).tolist(), "per_horizon_mse": squared.mean(axis=0).tolist(),
            "net6_direction_accuracy": float(np.mean(np.sign(a) == np.sign(p))),
            "net6_opposite_rate_all_windows": float(np.mean(a * p < 0)),
            "mean_abs_actual_net6": float(np.abs(a).mean()),
            "mean_abs_predicted_net6": float(np.abs(p).mean()),
            "actual_large_count": int(large.sum()), "predicted_large_count": int(triggered.sum()),
            "large_event_precision": hits / int(triggered.sum()) if triggered.any() else None,
            "large_event_recall": hits / int(large.sum()) if large.any() else None}

