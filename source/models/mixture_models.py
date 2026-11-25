"""Direct PM2.5 forecasting with probabilistic regime-specific Ridge experts."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


from ._fixed import FixedBudgetModel, compact_variants


REGIME_CUTOFFS = {
    "soft_experts_3": (-5.0, 5.0),
    "soft_experts_5": (-12.0, -5.0, 5.0, 12.0),
}
CANDIDATE_NAMES = tuple(REGIME_CUTOFFS)
CONCENTRATION_KNOTS = np.array([8.0, 14.0, 22.0, 35.0])
HORIZONS = 6


def _design(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or not len(x) or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError("Expected a nonempty, finite feature matrix")
    current = x[:, 0]
    if np.any(current < 0):
        raise ValueError("The first feature must be the current raw PM2.5 concentration")
    hinges = np.maximum(current[:, None] - CONCENTRATION_KNOTS, 0.0)
    return current, np.column_stack((x, hinges))


@dataclass
class SoftRegimeExperts:
    scaler: StandardScaler
    gate: LogisticRegression | None
    experts: tuple[Ridge, ...]
    lower: np.ndarray
    upper: np.ndarray

    def _probabilities(self, features, current):
        if self.gate is None:
            return np.ones((len(features), 1))
        probabilities = self.gate.predict_proba(features)
        # A strict decline bin is impossible when its upper endpoint would
        # imply negative concentrations, including the open boundary at zero.
        feasible = current[:, None] + self.upper > 0.0
        retained = probabilities * feasible
        mass = retained.sum(axis=1, keepdims=True)
        return np.divide(retained, mass, out=probabilities, where=mass > 0)

    def predict(self, x):
        current, features = _design(x)
        features = self.scaler.transform(features)
        weights = self._probabilities(features, current)
        paths = np.stack([expert.predict(features) for expert in self.experts], axis=1)
        paths = np.maximum(paths + current[:, None, None], 0.0)
        # Each true endpoint in this regime lies within these concentration
        # bounds, so its conditional mean lies there as well.
        lower = np.maximum(current[:, None] + self.lower, 0.0)
        upper = np.maximum(current[:, None] + self.upper, 0.0)
        paths[:, :, -1] = np.clip(paths[:, :, -1], lower, upper)
        return np.einsum("nr,nrh->nh", weights, paths)


def _fit(name, x, y, seed=42):
    """Train gate probabilities and direct six-horizon experts on past rows."""
    cutoffs = np.asarray(REGIME_CUTOFFS[name])
    current, features = _design(x)
    actual = np.asarray(y, dtype=np.float64)
    if (actual.shape != (len(features), HORIZONS)
            or not np.isfinite(actual).all() or np.any(actual < 0)):
        raise ValueError("Expected six finite, nonnegative concentrations per training row")
    residual = actual - current[:, None]
    regime = np.searchsorted(cutoffs, residual[:, -1], side="right")
    classes = np.unique(regime)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    gate = None
    if len(classes) > 1:
        gate = LogisticRegression(C=0.15, max_iter=200, random_state=seed).fit(features, regime)
    experts = []
    for label in classes:
        rows = regime == label
        experts.append(Ridge(alpha=20.0).fit(features[rows], residual[rows]))
    edges = np.r_[-np.inf, cutoffs, np.inf]
    return SoftRegimeExperts(scaler, gate, tuple(experts), edges[classes], edges[classes + 1])


class RegimeExpertsForecast(FixedBudgetModel):
    name = "soft_experts"

    @classmethod
    def variants(cls):
        return {
            f"regimes_{key}": recipe
            for key, recipe in compact_variants(CANDIDATE_NAMES, "soft_experts_").items()
        }

    def _fit_estimator(self, x, y):
        return _fit(self.recipe, x, y, seed=self.seed)
