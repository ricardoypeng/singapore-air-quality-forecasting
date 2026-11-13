"""Baselines, candidate models and their hyper-parameter grids.

All models follow the scikit-learn API and predict every horizon at once:
``predict(X)`` returns an array of shape ``(n_rows, n_horizons)``.

No deep learning is used, as required by the project brief.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


class PersistenceModel(BaseEstimator, RegressorMixin):
    """Naive baseline: every future hour equals the latest observed value."""

    def __init__(self, region: str = "east", horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6)):
        self.region = region
        self.horizons = horizons

    def fit(self, X: pd.DataFrame, y: Any = None) -> "PersistenceModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        latest = X[f"{self.region}_lag_0"].to_numpy()
        return np.repeat(latest[:, None], len(self.horizons), axis=1)


class SeasonalNaiveModel(BaseEstimator, RegressorMixin):
    """Naive baseline: the value at ``t+h`` equals the value 24 hours earlier.

    Uses lag ``24 - h`` of the target region, so the window must be >= 24.
    """

    def __init__(self, region: str = "east", horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6)):
        self.region = region
        self.horizons = horizons

    def fit(self, X: pd.DataFrame, y: Any = None) -> "SeasonalNaiveModel":
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        columns = [f"{self.region}_lag_{24 - h}" for h in self.horizons]
        return X[columns].to_numpy()


def _with_log_target(estimator: BaseEstimator, log_target: bool) -> BaseEstimator:
    """Optionally train on log(1 + y), which suits the skewed PM2.5 values."""
    if not log_target:
        return estimator
    return TransformedTargetRegressor(regressor=estimator, func=np.log1p, inverse_func=np.expm1)


def make_model(name: str, params: dict[str, Any], region: str, horizons: tuple[int, ...]):
    """Create an unfitted model.

    Args:
        name: One of ``MODEL_NAMES``.
        params: Hyper-parameters (one entry of that model's grid).
        region: Target region (used by the baselines).
        horizons: Forecast horizons (used by the baselines).

    Returns:
        A scikit-learn compatible regressor with multi-output ``predict``.
    """
    params = dict(params)
    log_target = params.pop("log_target", False)

    if name == "persistence":
        return PersistenceModel(region=region, horizons=horizons)
    if name == "seasonal_naive":
        return SeasonalNaiveModel(region=region, horizons=horizons)
    if name == "ridge":
        # Ridge handles several targets natively.
        model = make_pipeline(StandardScaler(), Ridge(**params))
        return _with_log_target(model, log_target)
    if name == "random_forest":
        # Random forests also handle several targets natively (one fit for all horizons).
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        return _with_log_target(model, log_target)
    if name == "hist_gradient_boosting":
        # Gradient boosting predicts one target at a time: one model per horizon.
        base = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
        return MultiOutputRegressor(_with_log_target(base, log_target))
    raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")


# Hyper-parameter grids searched on the validation period.
# Kept small on purpose so the full pipeline runs in minutes on a laptop.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "persistence": {},
    "seasonal_naive": {},
    "ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "log_target": [False, True],
    },
    "random_forest": {
        "n_estimators": [100],
        "max_depth": [12, None],
        "min_samples_leaf": [5, 20],
        "max_features": [0.3],
    },
    "hist_gradient_boosting": {
        "learning_rate": [0.05, 0.1],
        "max_leaf_nodes": [15, 31],
        "max_iter": [300],
        "l2_regularization": [0.0, 1.0],
        "log_target": [False, True],
    },
}

MODEL_NAMES = list(PARAM_GRIDS)
BASELINES = ["persistence", "seasonal_naive"]
