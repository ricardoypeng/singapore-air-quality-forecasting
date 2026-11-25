"""Fixed 80/20 combination of recent-weather CatBoost and analog regression."""

from dataclasses import dataclass

import numpy as np

from ._fixed import FixedBudgetModel
from .analog_models import AnalogForecast
from .catboost_variants import CatBoostVariants

RECENT_WIDTH = 295


@dataclass
class BlendPredictor:
    catboost: CatBoostVariants
    analog: AnalogForecast

    def predict(self, x):
        return (
            0.8 * self.catboost.predict(x[:, :RECENT_WIDTH])
            + 0.2 * self.analog.predict(x[:, RECENT_WIDTH:])
        )


class BlendedForecast(FixedBudgetModel):
    name = "blend"

    @classmethod
    def variants(cls):
        return {
            "catboost_80_analog_20": {
                "parameters": {},
                "budgets": [1],
                "training_cap": 1,
                "input_profile": "recent_compact",
                "shuffle": False,
            },
        }

    def _fit_estimator(self, x, y):
        catboost = CatBoostVariants(self.seed, self.threads)
        analog = AnalogForecast(self.seed, self.threads, "analog_local_constant")
        order = np.random.default_rng(self.seed).permutation(len(y))
        catboost.fit(x[order, :RECENT_WIDTH], y[order], 500)
        analog.fit(x[:, RECENT_WIDTH:], y, 1)
        return BlendPredictor(catboost, analog)

    def model_budget(self):
        return {
            "prediction_budget": self.budget,
            "catboost_trees": self.estimator.catboost.budget if self.budget else 0,
            "analog_weight": 0.2,
        }
