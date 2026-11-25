"""Shared lifecycle for estimators without a prefix prediction budget."""

from abc import abstractmethod

from threadpoolctl import threadpool_limits

from .base import ForecastModel


class FixedBudgetModel(ForecastModel):
    def __init__(self, seed=42, threads=4, recipe=None):
        self.seed = seed
        self.threads = threads
        self.recipe = recipe
        self.estimator = None

    def fit(self, x, y, budget):
        if type(budget) is not int or budget != 1 or self.budget:
            raise ValueError("Use a fresh model and a training budget of 1")
        with threadpool_limits(limits=self.threads):
            self.estimator = self._fit_estimator(x, y)
        self.budget = 1
        return self

    @abstractmethod
    def _fit_estimator(self, x, y):
        """Train and return the estimator."""

    def predict(self, x, budget=None):
        if not self.budget or (budget is not None and (type(budget) is not int or budget != 1)):
            raise ValueError("Prediction requires a fitted model and budget 1")
        with threadpool_limits(limits=self.threads):
            return self.estimator.predict(x)

    def model_budget(self):
        return {"prediction_budget": self.budget, "recipe": self.recipe}


def compact_variants(names, prefix):
    return {
        name.removeprefix(prefix): {
            "parameters": {"recipe": name},
            "budgets": [1],
            "training_cap": 1,
            "input_profile": "compact",
            "shuffle": False,
        }
        for name in names
    }
