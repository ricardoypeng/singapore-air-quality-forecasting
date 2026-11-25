"""Lead-aware and trajectory-loss variants for the CatBoost experiments."""

import numpy as np

from .catboost_forecast import CatBoostForecast

HORIZON = 6


def lead_matrix(x):
    calendar = x[:, 144:186].reshape(-1, HORIZON + 1, 6)
    converted = np.empty((len(x), HORIZON, x.shape[1] - 42 + 13), dtype=np.float32)
    converted[:, :, :144] = x[:, None, :144]
    converted[:, :, 144:150] = calendar[:, :1]
    converted[:, :, 150:156] = calendar[:, 1:]
    converted[:, :, 156] = np.arange(1, HORIZON + 1)
    converted[:, :, 157:] = x[:, None, 186:]
    return converted.reshape(-1, converted.shape[-1])


def target_transform(weight):
    difference = np.diff(np.eye(HORIZON), axis=0)
    return np.linalg.cholesky(np.eye(HORIZON) + weight * difference.T @ difference).T


class CatBoostVariants(CatBoostForecast):
    name = "catboost_variant"
    allow_missing = True

    def __init__(self, seed=42, threads=4, lead_time=False, trajectory=0.0):
        loss = "RMSE" if lead_time else "MultiRMSE"
        super().__init__(seed, threads, depth=6, loss_function=loss, eval_metric=loss)
        self.lead_time = lead_time
        self.transform = target_transform(trajectory) if trajectory else None

    @classmethod
    def variants(cls):
        recipes = (
            ("baseline_800", "five", 800, {}, False),
            ("baseline_1200", "five", 1200, {}, False),
            ("expanded_legacy", "five", 500, {}, True),
            ("original_recent", "recent", 500, {}, False),
            ("expanded_recent", "recent", 500, {}, True),
            ("lead_time_recent", "recent", 500, {"lead_time": True}, True),
            ("trajectory_010", "recent", 500, {"trajectory": 0.10}, True),
            ("trajectory_025", "recent", 500, {"trajectory": 0.25}, True),
            ("compact", "compact", 500, {}, True),
        )
        return {
            name: {
                "parameters": params,
                "budgets": [budget],
                "training_cap": budget,
                "input_profile": profile,
                "expanded_training": expanded,
            }
            for name, profile, budget, params, expanded in recipes
        }

    def _fit_arrays(self, x, y):
        if self.lead_time:
            return lead_matrix(x), y.ravel()
        if self.transform is not None:
            return x, y @ self.transform.T
        return x, y

    def _predict_array(self, x):
        return lead_matrix(x) if self.lead_time else x

    def _decode(self, predicted):
        output = predicted.reshape(-1, HORIZON)
        if self.transform is not None:
            output = np.linalg.solve(self.transform, output.T).T
        return output

    def model_budget(self):
        return {**super().model_budget(), "shared_tree_structure": not self.lead_time}
