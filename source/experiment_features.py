"""Causal recent pollution, regional and weather state for traditional models."""

from dataclasses import dataclass

import numpy as np


def available_mean(history):
    counts = np.isfinite(history).sum(axis=1)
    return np.divide(
        np.nansum(history, axis=1), counts, out=np.full(counts.shape, np.nan), where=counts > 0,
    )


def compact_values(base, recent):
    if base.ndim != 2 or base.shape[1] != 282 or recent.shape != (len(base), 13):
        raise ValueError("Expected original five-region and 13 latest weather columns")
    block = {}
    history = base[:, :144].reshape(-1, 24, 6)
    east = history[:, :, 0]
    regions = base[:, 186:].reshape(-1, 24, 4)
    weather = history[:, :, 1:]
    current = east[:, -1]

    block["east_current"] = current
    for lag in (1, 2, 3, 6, 12, 23):
        block[f"east_lag_{lag}"] = east[:, -lag - 1]
    for lag in (1, 3, 6):
        block[f"east_change_{lag}"] = current - east[:, -lag - 1]
    for span in (3, 6, 12, 24):
        block[f"east_mean_{span}"] = east[:, -span:].mean(axis=1)
        block[f"east_std_{span}"] = east[:, -span:].std(axis=1)

    for index, region in enumerate(("north", "south", "west", "central")):
        values = regions[:, :, index]
        block[f"{region}_minus_east"] = values[:, -1] - current
        block[f"{region}_change_3"] = values[:, -1] - values[:, -4]
    regional_mean = regions.mean(axis=2)
    for lag in (0, 3, 6, 12):
        block[f"regional_minus_east_lag_{lag}"] = regional_mean[:, -lag - 1] - east[:, -lag - 1]
    block["regional_change_6"] = regional_mean[:, -1] - regional_mean[:, -7]
    block["regional_spread"] = regions[:, -1].std(axis=1)

    weather_names = ("wind_u", "wind_v", "rain", "temperature", "humidity")
    weather = weather.copy()
    weather[:, :, 2] = np.log1p(weather[:, :, 2])
    for index, name in enumerate(weather_names):
        latest = recent[:, index]
        block[f"recent_{name}"] = np.log1p(latest) if name == "rain" else latest
        block[f"recent_{name}_missing"] = ~np.isfinite(latest)
        block[f"old_{name}_avg_6"] = available_mean(weather[:, -6:, index])
        block[f"old_{name}_avg_24"] = available_mean(weather[:, :, index])
    for index, name in enumerate(("wind", "rain", "temperature", "humidity")):
        block[f"latest_{name}_coverage"] = recent[:, index + 5]
        block[f"latest_{name}_age_log"] = np.log1p(recent[:, index + 9])
    for index, name in enumerate(("hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "month_sin", "month_cos")):
        block[name] = base[:, 144 + index]

    return np.column_stack(tuple(block.values())), tuple(block)


@dataclass
class CompactEncoder:
    names: tuple
    medians: np.ndarray

    @classmethod
    def fit(cls, values, names):
        medians = np.nanmedian(values, axis=0)
        if not np.isfinite(medians).all():
            raise ValueError("A compact feature was entirely unavailable during training")
        return cls(names, medians)

    def transform(self, values, names):
        if names != self.names or not np.isfinite(values[:, 0]).all() or np.isinf(values).any():
            raise ValueError("Unexpected compact feature order or invalid PM values")
        return np.where(np.isnan(values), self.medians, values)
