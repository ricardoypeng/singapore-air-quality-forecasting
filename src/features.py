"""Turn the hourly series into a supervised learning table.

Each row is one *forecast origin* ``t`` (the latest hour we have observed).
Features only use values observed at or before ``t``; targets are the
target region's values at ``t+1`` ... ``t+H``. This makes it impossible for a
feature to "see the future".

Feature groups (for target region ``r`` and window ``W``):

* ``r_lag_0 ... r_lag_{W-1}``: the last ``W`` hourly values of ``r``.
* ``<other>_lag_0 ... <other>_lag_{k-1}``: recent values of the other regions.
* rolling mean / std / max of ``r`` over 6 and 24 hours, and the 1-hour change.
* island-wide mean at ``t``.
* calendar: hour of day and month encoded as sine/cosine, day of week.

The time split keeps a gap of ``H`` hours between train / validation / test
so no target value from a later split is used as a label in an earlier one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data import REGIONS


@dataclass(frozen=True)
class FeatureConfig:
    """Settings that define the feature table.

    Attributes:
        region: Region to forecast (one of ``REGIONS``).
        window: Number of past hours of the target region used as lags.
        other_region_lags: Number of past hours used for each other region.
        horizons: Forecast horizons in hours.
    """

    region: str = "east"
    window: int = 24
    other_region_lags: int = 3
    horizons: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

    def __post_init__(self) -> None:
        if self.region not in REGIONS:
            raise ValueError(f"region must be one of {REGIONS}, got {self.region!r}")
        if self.window < 24:
            raise ValueError("window must be at least 24 (the rolling features use 24 hours)")

    @property
    def max_horizon(self) -> int:
        return max(self.horizons)

    def target_columns(self) -> list[str]:
        return [f"target_h{h}" for h in self.horizons]


def build_features(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Build the feature columns for every forecast origin.

    Rows near the start (not enough history) contain NaN and should be dropped
    by the caller; :func:`make_dataset` does this.

    Args:
        df: Cleaned hourly data from :func:`src.data.load_clean`.
        config: Feature settings.

    Returns:
        DataFrame with the same index as ``df`` and one column per feature.
    """
    r = config.region
    series = df[r]
    features: dict[str, pd.Series] = {}

    for lag in range(config.window):
        features[f"{r}_lag_{lag}"] = series.shift(lag)

    for other in REGIONS:
        if other == r:
            continue
        for lag in range(config.other_region_lags):
            features[f"{other}_lag_{lag}"] = df[other].shift(lag)

    for hours in (6, 24):
        rolling = series.rolling(hours)
        features[f"{r}_mean_{hours}h"] = rolling.mean()
        features[f"{r}_std_{hours}h"] = rolling.std()
        features[f"{r}_max_{hours}h"] = rolling.max()
    features[f"{r}_change_1h"] = series.diff(1)
    features["island_mean"] = df[REGIONS].mean(axis=1)

    hour = df.index.hour
    month = df.index.month
    features["hour_sin"] = pd.Series(np.sin(2 * np.pi * hour / 24), index=df.index)
    features["hour_cos"] = pd.Series(np.cos(2 * np.pi * hour / 24), index=df.index)
    features["month_sin"] = pd.Series(np.sin(2 * np.pi * (month - 1) / 12), index=df.index)
    features["month_cos"] = pd.Series(np.cos(2 * np.pi * (month - 1) / 12), index=df.index)
    features["day_of_week"] = pd.Series(df.index.day_of_week, index=df.index)

    return pd.DataFrame(features)


def build_targets(df: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Future values of the target region, one column per horizon."""
    series = df[config.region]
    return pd.DataFrame(
        {f"target_h{h}": series.shift(-h) for h in config.horizons}, index=df.index
    )


def make_dataset(df: pd.DataFrame, config: FeatureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Features ``X`` and targets ``y`` with incomplete rows removed.

    Returns:
        ``(X, y)`` sharing the same forecast-origin index.
    """
    X = build_features(df, config)
    y = build_targets(df, config)
    complete = X.notna().all(axis=1) & y.notna().all(axis=1)
    return X[complete], y[complete]


@dataclass(frozen=True)
class SplitDates:
    """Last forecast-origin date of the train and validation periods.

    Everything after ``val_end`` is the test period.
    """

    train_end: str = "2022-12-31 23:00"
    val_end: str = "2023-12-31 23:00"


def time_split(
    X: pd.DataFrame, y: pd.DataFrame, dates: SplitDates, gap_hours: int
) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    """Split by time into train / validation / test, never shuffling.

    The last ``gap_hours`` origins of train and of validation are dropped, so
    their targets (up to ``gap_hours`` ahead) never fall in the next split.

    Returns:
        ``{"train": (X, y), "val": (X, y), "test": (X, y)}``.
    """
    train_end = pd.Timestamp(dates.train_end)
    val_end = pd.Timestamp(dates.val_end)
    gap = pd.Timedelta(hours=gap_hours)

    masks = {
        "train": X.index <= train_end - gap,
        "val": (X.index > train_end) & (X.index <= val_end - gap),
        "test": X.index > val_end,
    }
    return {name: (X[mask], y[mask]) for name, mask in masks.items()}
