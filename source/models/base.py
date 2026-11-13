"""Contract for models discovered by the shared experiment runner."""
from abc import ABC, abstractmethod
import copy
from typing import Any
import numpy as np


class ForecastModel(ABC):
    """A model owns its variants, capacity policy and estimator implementation.

    Constructors accept seed, threads and the parameters declared by variants().
    fit consumes only training X/y; predict returns (samples, horizons).
    snapshot returns an independent, serializable predictor at a chosen budget.
    """
    name: str
    budget: int = 0

    @abstractmethod
    def __init__(self, seed: int = 42, threads: int = 4, **parameters: Any) -> None:
        """Initialize unfitted state with the shared constructor arguments."""

    @classmethod
    @abstractmethod
    def variants(cls) -> dict[str, dict[str, Any]]:
        """Return variant -> {parameters, budgets, training_cap}."""

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray, budget: int) -> 'ForecastModel':
        """Train to a declared capacity without reading validation or test."""

    @abstractmethod
    def predict(self, x: np.ndarray, budget: int | None = None) -> np.ndarray:
        """Predict using the active budget or an explicitly requested prefix."""

    def training_capacity(self, selected_budget: int, search_capacity: int) -> int:
        """Final capacity; override for estimators that must train a full trajectory."""
        return selected_budget

    def snapshot(self, budget: int) -> 'ForecastModel':
        """Keep independent state and select a prediction prefix without refitting."""
        if not isinstance(budget, int) or not 1 <= budget <= self.budget:
            raise ValueError('Snapshot budget is outside the fitted capacity')
        result = copy.deepcopy(self)
        result.budget = budget
        return result

    @abstractmethod
    def model_budget(self) -> dict[str, Any]:
        """Describe actual training capacity and active prediction capacity."""
