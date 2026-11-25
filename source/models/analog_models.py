"""Historical neighbor regression for six direct PM2.5 forecasts."""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.linalg.blas import sgemm
from sklearn.neighbors import NearestNeighbors
from threadpoolctl import threadpool_limits


from ._fixed import FixedBudgetModel, compact_variants


CANDIDATE_NAMES = ("analog_local_constant", "analog_local_linear")
NEIGHBOR_COUNTS = dict(zip(CANDIDATE_NAMES, (64, 192)))
HORIZONS = 6
PREDICTION_BATCH = 96
LOCAL_RIDGE = 0.45


def _checked_design(x, columns=None):
    values = np.asarray(x, dtype=np.float64)
    if (values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0
            or (columns is not None and values.shape[1] != columns)):
        raise ValueError("Expected a nonempty forecast design with the fitted feature width")
    if not np.isfinite(values).all() or np.any(values[:, 0] < 0):
        raise ValueError("Expected finite features and observed east PM2.5 in the first column")
    return values


def _neighbor_weights(distances):
    """Use an adaptive bandwidth based on the nearest and farthest returned neighbors."""
    squared = distances**2
    relative = squared - squared[:, :1]
    bandwidth = np.maximum(relative[:, -1:], 1e-12)
    weights = np.exp(-3 * relative / bandwidth)
    return weights / weights.sum(axis=1, keepdims=True)


class AnalogForecaster:
    """Predict a global change plus a correction learned from similar training hours."""

    def __init__(self, name, x, y):
        self.name = name
        self.center = x.mean(axis=0)
        scale = x.std(axis=0)
        self.scale = np.where(scale > 1e-8, scale, 1)
        self.training_state = self._standardize(x)
        change = y - x[:, :1]
        with threadpool_limits(limits=2):
            self._fit_global(self.training_state, change)
        fitted_change = self._global_trend(self.training_state)
        self.trend_scale = np.maximum(fitted_change.std(axis=0), 1.0)
        self.residual = change - fitted_change
        neighbor_count = min(NEIGHBOR_COUNTS[name], len(x))
        self.neighbors = NearestNeighbors(n_neighbors=neighbor_count, algorithm="brute", n_jobs=1)
        self.neighbors.fit(self._embedding(self.training_state, fitted_change))

    def _fit_global(self, state, change):
        self.global_center = state.mean(axis=0, dtype=np.float64).astype(np.float32)
        self.global_intercept = change.mean(axis=0)
        centered_state = state - self.global_center
        centered_change = (change - self.global_intercept).astype(np.float32)
        covariance = sgemm(1, centered_state, centered_state, trans_a=True)
        covariance.flat[::len(covariance) + 1] += 0.05 * len(state)
        cross = sgemm(1, centered_state, centered_change, trans_a=True)
        self.global_slopes = cho_solve(cho_factor(covariance), cross)

    def _global_trend(self, state):
        trend = np.einsum("nd,dh->nh", state - self.global_center, self.global_slopes)
        return trend + self.global_intercept

    def _standardize(self, x):
        # Bound extremes so one unusually large covariate cannot dominate the distance.
        return np.clip((x - self.center) / self.scale, -6, 6).astype(np.float32)

    def _embedding(self, state, trend):
        return np.column_stack((
            state / np.sqrt(state.shape[1]),
            0.5 * state[:, :1],
            0.65 * trend / (self.trend_scale * np.sqrt(HORIZONS)),
        )).astype(np.float32)

    def _local_linear(self, state, indices, weights, residual):
        # Center the local regression at the weighted neighbor mean. The residual
        # intercept stays unpenalized, and the slopes shrink toward the global fit.
        neighbor_states = self.training_state[indices].astype(np.float64)
        mean_state = np.einsum("nk,nkd->nd", weights, neighbor_states)
        mean_residual = np.einsum("nk,nkh->nh", weights, residual)
        centered = neighbor_states - mean_state[:, None, :]
        weighted = centered * weights[:, :, None]
        covariance = np.einsum("nki,nkj->nij", weighted, centered)
        diagonal = np.arange(state.shape[1])
        covariance[:, diagonal, diagonal] += LOCAL_RIDGE
        cross = np.einsum("nkd,nkh->ndh", weighted, residual - mean_residual[:, None, :])
        slope = np.linalg.solve(covariance, cross)
        return mean_residual + np.einsum("nd,ndh->nh", state - mean_state, slope)

    def predict(self, x):
        values = _checked_design(x, len(self.center))
        state = self._standardize(values)
        output = np.empty((len(values), HORIZONS), dtype=np.float64)
        with threadpool_limits(limits=2):
            trend = self._global_trend(state)
            embedding = self._embedding(state, trend)
            for start in range(0, len(values), PREDICTION_BATCH):
                span = slice(start, start + PREDICTION_BATCH)
                distances, indices = self.neighbors.kneighbors(embedding[span])
                weights = _neighbor_weights(distances)
                residual = self.residual[indices]
                if self.name == "analog_local_linear":
                    correction = self._local_linear(state[span], indices, weights, residual)
                else:
                    correction = np.einsum("nk,nkh->nh", weights, residual)
                output[span] = values[span, :1] + trend[span] + correction
        if not np.isfinite(output).all():
            raise ValueError("Historical neighbor regression produced non-finite predictions")
        return np.maximum(output, 0)


def _fit(name, x, y, seed=42):
    """Fit a deterministic neighbor model on chronological training origins."""
    if name not in NEIGHBOR_COUNTS:
        raise ValueError(f"Unknown historical neighbor candidate {name}")
    x = _checked_design(x)
    y = np.asarray(y, dtype=np.float64)
    if y.shape != (len(x), HORIZONS) or not np.isfinite(y).all() or np.any(y < 0):
        raise ValueError("Expected six finite, nonnegative future PM2.5 values per origin")
    return AnalogForecaster(name, x, y)


class AnalogForecast(FixedBudgetModel):
    name = "analog"

    @classmethod
    def variants(cls):
        return compact_variants(CANDIDATE_NAMES, "analog_")

    def _fit_estimator(self, x, y):
        return _fit(self.recipe, x, y, seed=self.seed)
