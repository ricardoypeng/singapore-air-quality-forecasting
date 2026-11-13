"""Joint six-output CatBoost with reproducible tree-prefix model snapshots."""

import copy
import hashlib
from typing import Any

import numpy as np
from source.models.base import ForecastModel


class CatBoostForecast(ForecastModel):
    """
    Fit one six-output ensemble and evaluate or export fixed tree prefixes.

    All CV and final models are trained once to the same declared capacity.
    Prefix selection never continues training with init_model, so checkpoint
    frequency cannot change the training trajectory.

    Attributes:
        seed             (int): Reproducible estimator seed.
        threads          (int): Positive CPU worker count.
        depth            (int): Shared symmetric-tree depth, four or six.
        estimator        (Any | None): Fitted native six-output CatBoost model.
        budget           (int): Number of retained trees used by default prediction.
        training_budget  (int): Actual capacity trained before any prefix export.
        n_features_in_   (int): Input width shared by all six output dimensions.
        training_key_    (str): Fingerprint of the original training matrix/targets.
    """

    name = 'catboost'

    @classmethod
    def variants(cls):
        return {f'depth{depth}': {'parameters': {'depth': depth, 'l2_leaf_reg': 10.0,
            'learning_rate': 0.05, 'loss_function': 'MultiRMSE', 'eval_metric': 'MultiRMSE',
            'has_time': False, 'boosting_type': 'Plain', 'grow_policy': 'SymmetricTree',
            'bootstrap_type': 'Bayesian', 'bagging_temperature': 1.0, 'random_strength': 1.0,
            'use_best_model': False, 'allow_writing_files': False, 'task_type': 'CPU'},
            'budgets': [200, 500], 'training_cap': 500} for depth in (4, 6)}

    def training_capacity(self, selected_budget: int, search_capacity: int) -> int:
        return search_capacity

    def __init__(self, seed: int = 42, threads: int = 4, depth: int = 4, **parameters) -> None:
        """
        Store the fixed CPU estimator configuration without fitting anything.

        Args:
            seed    (int): Reproducible model seed.
            threads (int): Positive CPU worker count.
            depth   (int): Symmetric-tree depth, four or six.

        Returns:
            None.
        """
        if threads < 1 or depth not in (4, 6):
            raise ValueError("Expected positive threads and depth four or six")
        self.parameters = dict(parameters)
        self.seed = seed
        self.threads = threads
        self.depth = depth
        self.estimator: Any | None = None
        self.budget = 0
        self.training_budget = 0

    def fit(self, x: np.ndarray, y: np.ndarray, budget: int) -> "CatBoostForecast":
        """
        Fit the full declared capacity exactly once, without init_model.

        Args:
            x      (np.ndarray): Observed inputs and known calendar, shape (N,P).
            y      (np.ndarray): Six true concentration labels, shape (N,6).
            budget (int): Full training capacity, normally 500 trees.

        Returns:
            CatBoostForecast: This fitted estimator.
        """
        values = np.asarray(x, dtype=np.float64)
        target = np.asarray(y, dtype=np.float64)
        if values.ndim != 2 or not len(values) or target.shape != (len(values), 6):
            raise ValueError("Expected nonempty X and six direct target columns")
        if not np.isfinite(values).all() or not np.isfinite(target).all():
            raise ValueError("CatBoost experiment expects finite inputs and labels")
        if not isinstance(budget, int) or budget < 1:
            raise ValueError("Training capacity must be a positive integer")
        key = hashlib.sha256(values.tobytes() + target.tobytes()).hexdigest()
        if self.estimator is not None:
            if key == self.training_key_ and budget == self.budget == self.training_budget:
                return self
            raise ValueError("Use a new model for refitting or a different training capacity")
        from catboost import CatBoostRegressor
        defaults: dict[str, Any] = dict(depth=self.depth, learning_rate=0.05, l2_leaf_reg=10.0,
            loss_function="MultiRMSE", eval_metric="MultiRMSE", task_type="CPU",
            boosting_type="Plain", grow_policy="SymmetricTree", bootstrap_type="Bayesian",
            bagging_temperature=1.0, random_strength=1.0, has_time=False,
            use_best_model=False, allow_writing_files=False)
        defaults.update(self.parameters)
        self.estimator = CatBoostRegressor(iterations=budget, thread_count=self.threads,
            random_seed=self.seed, verbose=False, **defaults)
        self.estimator.fit(values, target)
        actual = int(self.estimator.tree_count_)
        if actual != budget:
            raise RuntimeError("Fitted CatBoost tree count differs from declared capacity")
        self.budget = actual
        self.training_budget = actual
        self.n_features_in_ = values.shape[1]
        self.training_key_ = key
        return self

    def predict(self, x: np.ndarray, budget: int | None = None) -> np.ndarray:
        """
        Predict six outputs using a retained ensemble or an earlier prefix.

        Args:
            x      (np.ndarray): Observed inputs and known calendar, shape (N,P).
            budget (int | None): Optional prefix tree count within the retained model.

        Returns:
            np.ndarray: Six direct concentrations per row, shape (N,6).
        """
        if self.estimator is None:
            raise ValueError("Fit the model before predicting")
        values = np.asarray(x, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.n_features_in_ or not np.isfinite(values).all():
            raise ValueError("Expected finite inputs with the fitted feature width")
        end = self.budget if budget is None else budget
        if not isinstance(end, int) or not 1 <= end <= int(self.estimator.tree_count_):
            raise ValueError("Prediction prefix must be within retained tree count")
        predicted = np.asarray(self.estimator.predict(
            values, ntree_start=0, ntree_end=end, thread_count=self.threads,
        ), dtype=np.float64)
        if predicted.shape != (len(values), 6) or not np.isfinite(predicted).all():
            raise ValueError("Expected six finite native CatBoost outputs per row")
        return predicted

    def snapshot(self, budget: int) -> "CatBoostForecast":
        """
        Return a separate model containing only the requested leading trees.

        Args:
            budget (int): Positive number of trees no greater than retained capacity.

        Returns:
            CatBoostForecast: Independent prefix snapshot suitable for serialization.
        """
        if self.estimator is None or not isinstance(budget, int) or not 1 <= budget <= self.budget:
            raise ValueError("Snapshot prefix must be within the fitted model")
        result = copy.copy(self)
        estimator = self.estimator.copy()
        if estimator is None:
            raise RuntimeError("CatBoost copy returned no estimator")
        estimator.shrink(ntree_end=budget, ntree_start=0)
        if int(estimator.tree_count_) != budget:
            raise RuntimeError("Snapshot does not contain the requested tree prefix")
        result.estimator = estimator
        result.budget = budget
        return result

    def model_budget(self) -> dict[str, Any]:
        """
        Distinguish actual training capacity from the retained output model.

        Returns:
            dict[str, Any]: Joint output count, actual training and retained trees.
        """
        return {"outputs": 6, "shared_tree_structure": True,
                "actual_trained_iterations": self.training_budget,
                "retained_trees": int(self.estimator.tree_count_) if self.estimator is not None else 0,
                "prediction_budget": self.budget,
                "training_policy": "single full-capacity fit; prefix scoring and shrink snapshots"}
