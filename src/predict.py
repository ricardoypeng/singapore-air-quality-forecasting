"""Batch prediction demo: forecast the next 1–6 hours from recent data.

This shows how the trained model would be used in practice. Give it a CSV of
recent hourly readings (same columns as the cleaned data: ``timestamp`` plus
the five regions, at least 24 consecutive hours) and it prints a forecast for
each of the next hours.

Examples::

    # Use the last 48 hours of the NEA data as "recent readings"
    python -m src.predict

    # Use your own file
    python -m src.predict --input my_recent_readings.csv

Monitoring if deployed: log prediction latency per request, and compare
forecasts with the readings that later arrive (rolling RMSE). A rising error,
or input values outside the training range, signal data drift and the need
to retrain.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import pandas as pd

from src.data import REGIONS, load_clean
from src.features import build_features
from src.train import MODEL_PATH

# Same NEA bands as src.metrics, for a human-readable label.
BANDS = [(0, "Normal"), (56, "Elevated"), (151, "High"), (251, "Very high")]


def band(value: float) -> str:
    """NEA 1-hour PM2.5 band name for a concentration."""
    return [name for lower, name in BANDS if value >= lower][-1]


def load_recent(path: Path | None, hours: int = 48) -> pd.DataFrame:
    """Read recent readings, or take the last ``hours`` of the NEA data."""
    if path is None:
        return load_clean().tail(hours)
    recent = pd.read_csv(path, parse_dates=["timestamp"], index_col="timestamp")
    missing = set(REGIONS) - set(recent.columns)
    if missing:
        raise ValueError(f"Input is missing columns: {sorted(missing)}")
    return recent.sort_index()[REGIONS].astype(float)


def forecast(recent: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Forecast every horizon from the latest hour in ``recent``.

    Args:
        recent: Consecutive hourly readings, oldest first.
        bundle: Contents of ``models/best_model.joblib``.

    Returns:
        One row per horizon with the forecast time, value and NEA band.
    """
    config = bundle["config"]
    if len(recent) < config.window:
        raise ValueError(f"Need at least {config.window} hours of readings, got {len(recent)}")
    steps = recent.index.to_series().diff().dropna()
    if (steps != pd.Timedelta(hours=1)).any():
        raise ValueError("Readings must be consecutive hours with no gaps")

    features = build_features(recent, config)[bundle["feature_columns"]].tail(1)
    if features.isna().any().any():
        raise ValueError("Not enough history to compute all features")

    values = bundle["model"].predict(features)[0]
    origin = recent.index[-1]
    return pd.DataFrame({
        "forecast_for": [origin + pd.Timedelta(hours=h) for h in config.horizons],
        "hours_ahead": list(config.horizons),
        f"{config.region}_pm25": values.round(1),
        "nea_band": [band(v) for v in values],
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast PM2.5 for the next hours.")
    parser.add_argument("--input", type=Path, help="CSV of recent hourly readings")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="trained model file")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    recent = load_recent(args.input)

    start = time.perf_counter()
    result = forecast(recent, bundle)
    latency_ms = 1000 * (time.perf_counter() - start)

    print(f"Model: {bundle['name']}   region: {bundle['config'].region}")
    print(f"Latest reading: {recent.index[-1]}  "
          f"({bundle['config'].region} = {recent[bundle['config'].region].iloc[-1]:.0f} µg/m³)")
    print(result.to_string(index=False))
    print(f"Prediction took {latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
