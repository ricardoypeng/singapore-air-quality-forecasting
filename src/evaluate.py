"""Evaluation figures and tables, built from the outputs of ``src.train``.

Run after training::

    python -m src.evaluate

Reads ``results/*.csv`` and ``models/best_model.joblib`` and writes figures
``07_`` to ``12_`` into ``reports/figures/`` plus ``results/worst_errors.csv``.
"""

from __future__ import annotations

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data import load_clean
from src.eda import save_figure
from src.features import SplitDates, make_dataset, time_split
from src.metrics import ELEVATED_THRESHOLD, rmse
from src.models import RANDOM_STATE
from src.train import MODEL_PATH, RESULTS_DIR

# Readable model names for figures.
LABELS = {
    "persistence": "Persistence (baseline)",
    "seasonal_naive": "Same hour yesterday (baseline)",
    "ridge": "Ridge regression",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Gradient boosting",
}

# PM2.5 bands used for the error-by-level audit (µg/m³). The upper bands
# follow NEA's 1-hour PM2.5 bands; the lowest band is split at the WHO
# 24-hour guideline (15) to show behaviour on ordinary days.
LEVEL_BANDS = [0, 15, 35, ELEVATED_THRESHOLD, 150, np.inf]
LEVEL_LABELS = ["0-15", "15-35", "35-56", "56-150 (elevated)", "150+ (high)"]


def load_results() -> dict:
    """Load all training outputs into one dictionary."""
    return {
        "metrics": pd.read_csv(RESULTS_DIR / "test_metrics.csv"),
        "predictions": pd.read_csv(RESULTS_DIR / "test_predictions.csv", parse_dates=["origin"]),
        "summary": json.loads((RESULTS_DIR / "summary.json").read_text()),
        "audit_path": RESULTS_DIR / "region_audit.csv",
        "bundle": joblib.load(MODEL_PATH),
    }


def plot_error_by_horizon(metrics: pd.DataFrame, column: str = "RMSE") -> plt.Figure:
    """Test error of every model as the forecast horizon grows."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, part in metrics.groupby("model", sort=False):
        style = "--" if name in ("persistence", "seasonal_naive") else "-"
        ax.plot(part["horizon_hours"], part[column], style, marker="o", label=LABELS.get(name, name))
    which = "elevated hours only (≥ 56 µg/m³)" if "elevated" in column else "all hours"
    ax.set(title=f"Test {column.split('_')[0]} by forecast horizon, {which}",
           xlabel="Hours ahead", ylabel=f"{column.split('_')[0]} (µg/m³)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    return fig


def plot_forecast_example(
    predictions: pd.DataFrame, best: str, horizon: int, start: str, end: str
) -> plt.Figure:
    """Actual values vs the best model and persistence for one horizon and period.

    The x-axis is the time being forecast (origin + horizon).
    """
    part = predictions[predictions["horizon_hours"] == horizon].copy()
    part["target_time"] = part["origin"] + pd.to_timedelta(horizon, unit="h")
    part = part[(part["target_time"] >= start) & (part["target_time"] <= end)]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(part["target_time"], part["actual"], color="black", linewidth=1.2, label="Actual")
    ax.plot(part["target_time"], part["persistence"], linewidth=0.9, alpha=0.8,
            label=LABELS["persistence"])
    ax.plot(part["target_time"], part[best], linewidth=0.9, label=LABELS.get(best, best))
    ax.axhline(ELEVATED_THRESHOLD, color="tab:red", linestyle=":", linewidth=1,
               label="NEA 'elevated' band (56)")
    ax.set(title=f"{horizon}-hour-ahead forecasts, {start} to {end}",
           xlabel="Time", ylabel="PM2.5 (µg/m³)")
    ax.legend(loc="upper left")
    return fig


def error_by_level(predictions: pd.DataFrame, model: str) -> pd.DataFrame:
    """Bias (mean of prediction - actual) and MAE by actual PM2.5 band.

    A negative bias means the model under-predicts, i.e. it would warn people
    too little.
    """
    data = predictions.assign(
        band=pd.cut(predictions["actual"], LEVEL_BANDS, labels=LEVEL_LABELS, right=False),
        error=predictions[model] - predictions["actual"],
    )
    table = data.groupby(["band", "horizon_hours"], observed=True)["error"].agg(
        bias="mean", MAE=lambda e: e.abs().mean(), hours="size"
    )
    return table.reset_index()


def plot_error_by_level(predictions: pd.DataFrame, model: str) -> plt.Figure:
    """Bar chart of bias and MAE per PM2.5 band, for 1 h and 6 h ahead."""
    table = error_by_level(predictions, model)
    table = table[table["horizon_hours"].isin([1, 6])]

    fig, (ax_bias, ax_mae) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, column in [(ax_bias, "bias"), (ax_mae, "MAE")]:
        pivot = table.pivot(index="band", columns="horizon_hours", values=column)
        pivot.columns = [f"{h} h ahead" for h in pivot.columns]
        pivot.plot.bar(ax=ax, rot=20)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set(xlabel="Actual PM2.5 band (µg/m³)", ylabel=f"{column} (µg/m³)")
    ax_bias.set_title(f"{LABELS.get(model, model)}: bias (prediction − actual)")
    ax_mae.set_title(f"{LABELS.get(model, model)}: mean absolute error")
    return fig


def plot_region_audit(audit: pd.DataFrame) -> plt.Figure:
    """Mean test RMSE of the best model vs persistence for every region."""
    pivot = audit.pivot(index="region", columns="model", values="RMSE_mean")
    pivot = pivot.rename(columns=LABELS)

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot.bar(ax=ax, rot=0)
    ax.set(title="Test RMSE by region (mean over 1–6 h)", xlabel="Region", ylabel="RMSE (µg/m³)")
    ax.legend(loc="lower right")
    return fig


def feature_importance(bundle: dict, n_rows: int = 3000, n_repeats: int = 3) -> pd.Series:
    """Permutation importance of the saved model on a sample of the test split.

    Each feature is shuffled in turn; the increase in mean RMSE (over all
    horizons) shows how much the model relies on it.
    """
    config = bundle["config"]
    summary = json.loads((RESULTS_DIR / "summary.json").read_text())
    dates = SplitDates(**summary["split"])
    X, y = make_dataset(load_clean(), config)
    X_test, y_test = time_split(X, y, dates, gap_hours=config.max_horizon)["test"]
    sample = X_test.sample(n=min(n_rows, len(X_test)), random_state=RANDOM_STATE)

    def neg_rmse(model, X_part, y_part):
        pred = model.predict(X_part)
        return -np.mean([rmse(y_part.iloc[:, i], pred[:, i]) for i in range(y_part.shape[1])])

    result = permutation_importance(
        bundle["model"], sample, y_test.loc[sample.index], scoring=neg_rmse,
        n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=X_test.columns).sort_values(ascending=False)


def plot_feature_importance(importance: pd.Series, top: int = 15) -> plt.Figure:
    """Horizontal bar chart of the most important features."""
    fig, ax = plt.subplots(figsize=(7, 5))
    importance.head(top).iloc[::-1].plot.barh(ax=ax)
    ax.set(title=f"Top {top} features (permutation importance)",
           xlabel="Increase in RMSE when shuffled (µg/m³)")
    return fig


def worst_errors(predictions: pd.DataFrame, model: str, horizon: int, n: int = 10) -> pd.DataFrame:
    """The ``n`` forecasts with the largest absolute error, for discussion."""
    part = predictions[predictions["horizon_hours"] == horizon].copy()
    part["target_time"] = part["origin"] + pd.to_timedelta(horizon, unit="h")
    part["error"] = part[model] - part["actual"]
    part = part.reindex(part["error"].abs().sort_values(ascending=False).index).head(n)
    numeric = ["actual", "persistence", model, "error"]
    part[numeric] = part[numeric].round(1)
    return part[["origin", "target_time", *numeric]]


def main() -> None:
    """Create every evaluation figure and table."""
    results = load_results()
    best = results["summary"]["best_model"]
    metrics, predictions = results["metrics"], results["predictions"]

    figures = {
        "07_test_rmse_by_horizon": plot_error_by_horizon(metrics, "RMSE"),
        "08_test_rmse_elevated_by_horizon": plot_error_by_horizon(metrics, "RMSE_elevated"),
        "09_forecast_haze_2026_h1": plot_forecast_example(predictions, best, 1, "2026-08-15", "2026-10-01"),
        "10_forecast_haze_2026_h6": plot_forecast_example(predictions, best, 6, "2026-08-15", "2026-10-01"),
        "11_error_by_level": plot_error_by_level(predictions, best),
        "12_feature_importance": plot_feature_importance(feature_importance(results["bundle"])),
    }
    if results["audit_path"].exists():
        figures["13_region_audit"] = plot_region_audit(pd.read_csv(results["audit_path"]))

    for name, fig in figures.items():
        print(f"saved {save_figure(fig, name).name}")
        plt.close(fig)

    worst = worst_errors(predictions, best, horizon=6)
    worst.to_csv(RESULTS_DIR / "worst_errors.csv", index=False)
    print("Largest 6-hour-ahead errors of the best model:")
    print(worst.to_string(index=False))


if __name__ == "__main__":
    main()
