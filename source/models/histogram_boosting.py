"""Independent direct-horizon histogram boosting for bounded experiments."""
import hashlib
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from itertools import islice
from source.models.base import ForecastModel


class HistogramBoostingForecast(ForecastModel):
    """
    Grow six independent direct-horizon histogram boosting regressors.

    Attributes:
        estimators (list): Scalar regressors in h1-to-h6 order.
        budget       (int): Current iteration budget per horizon.
        training_key (str | None): Fingerprint protecting warm-start sample identity.
    """

    name = 'hgb'

    @classmethod
    def variants(cls):
        return {name: {'parameters': {'max_leaf_nodes': leaves, 'min_samples_leaf': minimum,
                'l2_regularization': penalty, 'learning_rate': 0.05,
                'loss': 'squared_error', 'early_stopping': False},
                'budgets': [100, 200, 300], 'training_cap': 300}
                for name, leaves, minimum, penalty in [('standard', 15, 30, 1.0), ('regularized', 31, 50, 10.0)]}

    def __init__(self, seed: int = 42, threads: int = 4, max_leaf_nodes: int = 15,
                 min_samples_leaf: int = 30, l2_regularization: float = 1.0,
                 learning_rate: float = 0.05, loss: str = "squared_error", early_stopping: bool = False) -> None:
        """
        Initialize six warm-start heads with predeclared parameters.

        Args:
            seed              (int): Reproducible model seed.
            max_leaf_nodes    (int): Maximum leaves per boosting tree.
            min_samples_leaf  (int): Minimum rows per leaf.
            l2_regularization (float): Leaf-value L2 penalty.

        Returns:
            None.
        """
        self.estimators = [HistGradientBoostingRegressor(
            max_iter=1, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization, learning_rate=learning_rate, loss=loss,
            random_state=seed, warm_start=True,
        ) for _ in range(6)]
        for estimator in self.estimators:
            estimator.set_params(early_stopping=early_stopping)
        self.budget = 0
        self.training_key: str | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, budget: int) -> "HistogramBoostingForecast":
        """
        Grow each head using its own true target column.

        Args:
            x      (np.ndarray): Shared observed inputs and known calendar.
            y      (np.ndarray): Direct target matrix, shape (N,6).
            budget (int): Increasing iterations per head.

        Returns:
            HistogramBoostingForecast: This estimator.
        """
        if y.shape != (len(x), 6) or budget <= self.budget:
            raise ValueError("Expected six labels and an increasing budget")
        key = hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest()
        if self.training_key is not None and key != self.training_key:
            raise ValueError("Warm starts require identical training samples")
        self.training_key = key
        for index, estimator in enumerate(self.estimators):
            estimator.set_params(max_iter=budget)
            estimator.fit(x, y[:, index])
        self.budget = budget
        return self

    def predict(self, x: np.ndarray, budget: int | None = None) -> np.ndarray:
        """
        Predict six horizons independently from the same input rows.

        Args:
            x (np.ndarray): Shared observed inputs and known calendar.

        Returns:
            np.ndarray: Six direct predictions per row.
        """
        end = self.budget if budget is None else budget
        if not isinstance(end, int) or not 1 <= end <= self.estimators[0].n_iter_:
            raise ValueError('Prediction prefix must be within trained iterations')
        return np.column_stack([estimator.predict(x) if end == estimator.n_iter_
            else next(islice(estimator.staged_predict(x), end - 1, end))
            for estimator in self.estimators])

    def model_budget(self) -> dict[str, Any]:
        """
        Report actual fitted ensemble size.

        Returns:
            dict[str, Any]: Six heads and summed iteration count.
        """
        return {"heads": 6, "iterations_per_head": self.budget,
                "actual_total_iterations": sum(int(model.n_iter_) for model in self.estimators),
                "snapshot_policy": "independent full fitted state; public staged_predict selects prefix"}
