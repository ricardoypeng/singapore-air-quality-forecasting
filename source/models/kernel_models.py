"""Direct six-output ridge regression with linear and random RBF features."""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import blas, cho_factor, cho_solve


from ._fixed import FixedBudgetModel, compact_variants


CANDIDATE_NAMES = ("rbf_ridge_broad", "rbf_ridge_local")
RBF_CONFIGS = {
    "rbf_ridge_broad": (0.25, 5.0),
    "rbf_ridge_local": (0.75, 5.0),
}
RFF_COMPONENTS = 1024
CHUNK_SIZE = 4096


@dataclass
class RBFKernelRidge:
    mean: np.ndarray
    scale: np.ndarray
    frequencies: np.ndarray
    phase: np.ndarray
    coefficients: np.ndarray

    def features(self, x):
        standard = np.clip((x - self.mean) / self.scale, -8.0, 8.0).astype(np.float32)
        phase = blas.sgemm(1.0, standard, self.frequencies)
        phase += self.phase
        np.cos(phase, out=phase)
        phase *= np.sqrt(2.0 / self.phase.size)
        linear = standard / np.sqrt(float(standard.shape[1]))
        transform = np.empty((len(x), linear.shape[1] + phase.shape[1] + 1), dtype=np.float32)
        transform[:, :linear.shape[1]] = linear
        transform[:, linear.shape[1]:-1] = phase
        transform[:, -1] = 1.0
        return transform

    def predict(self, x):
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.mean.size or not np.isfinite(values).all():
            raise ValueError("Prediction design must be finite and have the fitted feature width")
        predictions = np.empty((len(values), self.coefficients.shape[1]), dtype=np.float64)
        for start in range(0, len(values), CHUNK_SIZE):
            block = slice(start, start + CHUNK_SIZE)
            predictions[block] = blas.dgemm(1.0, self.features(values[block]), self.coefficients)
        if not np.isfinite(predictions).all():
            raise ValueError("Kernel regression produced non-finite predictions")
        return predictions


def _fit(name, x, y, seed=42):
    """Fit the requested RBF length scale using only the supplied training rows."""
    values = np.asarray(x, dtype=np.float64)
    labels = np.asarray(y, dtype=np.float64)
    if (values.ndim != 2 or not values.size or labels.shape != (len(values), 6)
            or not np.isfinite(values).all() or not np.isfinite(labels).all()):
        raise ValueError("Expected a finite training design and six finite labels per origin")

    gamma_factor, alpha = RBF_CONFIGS[name]
    dimensions = values.shape[1]
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale < 1e-7] = 1.0
    generator = np.random.default_rng(seed)
    # The RBF kernel is exp(-gamma * ||x - x'||²), with dimension-normalized gamma.
    width = np.sqrt(2 * gamma_factor / dimensions)
    frequencies = generator.normal(0.0, width, size=(dimensions, RFF_COMPONENTS)).astype(np.float32)
    phase = generator.uniform(0.0, 2.0 * np.pi, size=RFF_COMPONENTS).astype(np.float32)

    feature_count = dimensions + RFF_COMPONENTS + 1
    cross = np.zeros((feature_count, 6), dtype=np.float64)
    gram = np.zeros((feature_count, feature_count), dtype=np.float64)
    model = RBFKernelRidge(mean, scale, frequencies, phase, np.zeros_like(cross))
    for start in range(0, len(values), CHUNK_SIZE):
        block = slice(start, start + CHUNK_SIZE)
        features = np.asfortranarray(model.features(values[block]).T, dtype=np.float64)
        # Compute symmetric products in bounded batches without storing the full design.
        gram += blas.dsyrk(1.0, features, lower=0)
        cross += blas.dgemm(1.0, features, labels[block])
    diagonal = np.arange(feature_count - 1)
    gram[diagonal, diagonal] += alpha
    factor = cho_factor(gram, lower=False)
    model.coefficients = cho_solve(factor, cross)
    return model


class RBFKernelForecast(FixedBudgetModel):
    name = "rbf_ridge"

    @classmethod
    def variants(cls):
        return compact_variants(CANDIDATE_NAMES, "rbf_ridge_")

    def _fit_estimator(self, x, y):
        return _fit(self.recipe, x, y, seed=self.seed)
