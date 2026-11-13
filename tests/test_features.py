"""Tests for features, time split and baselines, on synthetic data."""

import numpy as np
import pandas as pd
import pytest

from src.data import REGIONS
from src.features import FeatureConfig, SplitDates, build_features, make_dataset, time_split
from src.models import PersistenceModel, SeasonalNaiveModel


@pytest.fixture
def hourly():
    """Ten days of synthetic hourly data where east = hour counter 0, 1, 2, ..."""
    index = pd.date_range("2022-12-25", periods=24 * 10, freq="h")
    counter = np.arange(len(index), dtype=float)
    return pd.DataFrame({r: counter + i * 1000 for i, r in enumerate(REGIONS)}, index=index)


def test_features_use_only_past_values(hourly):
    config = FeatureConfig(region="east")
    X, y = make_dataset(hourly, config)
    east_now = hourly["east"].loc[X.index]
    # lag k must equal the value k hours before the origin
    assert (X["east_lag_0"] == east_now).all()
    assert (X["east_lag_5"] == east_now - 5).all()
    # target h must equal the value h hours after the origin
    assert (y["target_h3"] == east_now + 3).all()


def test_changing_future_does_not_change_features(hourly):
    config = FeatureConfig(region="east")
    origin = hourly.index[100]
    before = build_features(hourly, config).loc[origin]
    changed = hourly.copy()
    changed.loc[changed.index > origin] = -999.0
    after = build_features(changed, config).loc[origin]
    pd.testing.assert_series_equal(before, after)


def test_time_split_has_gap_and_no_overlap(hourly):
    config = FeatureConfig(region="east")
    X, y = make_dataset(hourly, config)
    dates = SplitDates(train_end="2022-12-29 23:00", val_end="2022-12-31 23:00")
    splits = time_split(X, y, dates, gap_hours=config.max_horizon)

    train_last = splits["train"][0].index.max()
    val_first, val_last = splits["val"][0].index.min(), splits["val"][0].index.max()
    test_first = splits["test"][0].index.min()
    # the latest target of each split is still before the next split starts
    assert train_last + pd.Timedelta(hours=config.max_horizon) <= pd.Timestamp(dates.train_end)
    assert val_first > pd.Timestamp(dates.train_end)
    assert val_last + pd.Timedelta(hours=config.max_horizon) <= pd.Timestamp(dates.val_end)
    assert test_first > pd.Timestamp(dates.val_end)


def test_baselines(hourly):
    config = FeatureConfig(region="east")
    X, y = make_dataset(hourly, config)
    persistence = PersistenceModel("east", config.horizons).fit(X, y).predict(X)
    assert (persistence[:, 0] == X["east_lag_0"]).all()
    seasonal = SeasonalNaiveModel("east", config.horizons).fit(X, y).predict(X)
    # value at t+h one day earlier = east(t) + h - 24
    expected = X["east_lag_0"].to_numpy()[:, None] + np.array(config.horizons) - 24
    assert np.allclose(seasonal, expected)


def test_invalid_config():
    with pytest.raises(ValueError):
        FeatureConfig(region="mars")
    with pytest.raises(ValueError):
        FeatureConfig(window=12)
