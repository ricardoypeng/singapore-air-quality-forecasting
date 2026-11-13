"""Error metrics for multi-horizon forecasts."""

from __future__ import annotations

import numpy as np
import pandas as pd

# NEA 1-hour PM2.5 bands: Band I "Normal" is 0-55 µg/m³, Band II "Elevated"
# starts at 56. We call hours at or above this level "elevated" (haze) hours.
ELEVATED_THRESHOLD = 56.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def horizon_metrics(
    y_true: pd.DataFrame, y_pred: np.ndarray, horizons: tuple[int, ...]
) -> pd.DataFrame:
    """RMSE and MAE for each horizon, on all hours and on elevated hours only.

    Args:
        y_true: Actual values, one column per horizon.
        y_pred: Predictions with the same shape as ``y_true``.
        horizons: The horizon (in hours) of each column.

    Returns:
        DataFrame indexed by horizon with columns ``RMSE``, ``MAE``,
        ``RMSE_elevated``, ``MAE_elevated`` and ``n_elevated``.
    """
    actual = y_true.to_numpy()
    pred = np.asarray(y_pred)
    rows = {}
    for i, h in enumerate(horizons):
        a, p = actual[:, i], pred[:, i]
        elevated = a >= ELEVATED_THRESHOLD
        rows[h] = {
            "RMSE": rmse(a, p),
            "MAE": mae(a, p),
            "RMSE_elevated": rmse(a[elevated], p[elevated]) if elevated.any() else np.nan,
            "MAE_elevated": mae(a[elevated], p[elevated]) if elevated.any() else np.nan,
            "n_elevated": int(elevated.sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("horizon_hours")
