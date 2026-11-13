"""End-to-end smoke test: quick training run, then a prediction."""

import json

import joblib
import pytest

from src.data import load_clean
from src.features import FeatureConfig
from src.predict import band, forecast
from src.train import run


def test_quick_pipeline_and_prediction(tmp_path):
    results_dir, model_path = tmp_path / "results", tmp_path / "model.joblib"
    summary = run(quick=True, results_dir=results_dir, model_path=model_path, audit=False)

    for name in ["tuning.csv", "test_metrics.csv", "test_predictions.csv", "summary.json"]:
        assert (results_dir / name).exists()
    assert json.loads((results_dir / "summary.json").read_text())["best_model"] == summary["best_model"]

    bundle = joblib.load(model_path)
    result = forecast(load_clean().tail(48), bundle)
    assert list(result["hours_ahead"]) == [1, 2, 3, 4, 5, 6]
    assert result.iloc[:, 2].between(0, 1000).all()


def test_forecast_rejects_gaps():
    recent = load_clean().tail(48)
    recent = recent.drop(recent.index[10])  # remove one hour in the middle
    with pytest.raises(ValueError, match="consecutive"):
        forecast(recent, {"config": FeatureConfig()})


def test_band_names():
    assert band(10) == "Normal"
    assert band(56) == "Elevated"
    assert band(200) == "High"
    assert band(300) == "Very high"
