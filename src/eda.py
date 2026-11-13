"""Plots and summaries for exploring the cleaned PM2.5 data.

Every plotting function takes the cleaned DataFrame from
:func:`src.data.load_clean` (or one column of it) and returns a matplotlib
``Figure``, so the notebook decides whether to show or save it.

All figures in ``reports/figures/`` are recreated by running
``notebooks/01_data_exploration.ipynb`` from top to bottom.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT, REGIONS

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"

# WHO 2021 guideline for 24-hour mean PM2.5, in µg/m³.
WHO_24H_GUIDELINE = 15


def save_figure(fig: plt.Figure, name: str, folder: Path = FIGURE_DIR) -> Path:
    """Save a figure as PNG under ``reports/figures`` and return its path."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_daily_series(df: pd.DataFrame) -> plt.Figure:
    """Daily mean PM2.5 (average of all regions) over the whole period."""
    daily = df[REGIONS].mean(axis=1).resample("D").mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily.values, linewidth=0.6)
    ax.axhline(WHO_24H_GUIDELINE, color="tab:red", linestyle="--", linewidth=1,
               label=f"WHO 24-h guideline ({WHO_24H_GUIDELINE} µg/m³)")
    ax.set(title="Daily mean PM2.5 across Singapore", xlabel="Date", ylabel="PM2.5 (µg/m³)")
    ax.legend(loc="upper right")
    return fig


def plot_distribution(series: pd.Series) -> plt.Figure:
    """Histogram of hourly values, on a normal and a log scale."""
    fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(12, 4))
    ax_lin.hist(series, bins=100)
    ax_lin.set(title=f"{series.name}: hourly PM2.5", xlabel="PM2.5 (µg/m³)", ylabel="Hours")
    ax_log.hist(np.log1p(series), bins=100)
    ax_log.set(title=f"{series.name}: log(1 + PM2.5)", xlabel="log(1 + PM2.5)", ylabel="Hours")
    return fig


def plot_hourly_profile(df: pd.DataFrame) -> plt.Figure:
    """Average PM2.5 for each hour of the day, one line per region."""
    profile = df[REGIONS].groupby(df.index.hour).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    profile.plot(ax=ax, marker="o", markersize=3)
    ax.set(title="Average PM2.5 by hour of day", xlabel="Hour of day (SGT)",
           ylabel="PM2.5 (µg/m³)", xticks=range(0, 24, 2))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    return fig


def plot_monthly_profile(df: pd.DataFrame) -> plt.Figure:
    """Average PM2.5 for each calendar month, one line per region."""
    profile = df[REGIONS].groupby(df.index.month).mean()

    fig, ax = plt.subplots(figsize=(8, 4))
    profile.plot(ax=ax, marker="o", markersize=3)
    ax.set(title="Average PM2.5 by month", xlabel="Month", ylabel="PM2.5 (µg/m³)",
           xticks=range(1, 13))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    return fig


def plot_region_correlation(df: pd.DataFrame) -> plt.Figure:
    """Correlation between the hourly values of the five regions."""
    corr = df[REGIONS].corr()

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(corr, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(REGIONS)), REGIONS, rotation=45)
    ax.set_yticks(range(len(REGIONS)), REGIONS)
    for i in range(len(REGIONS)):
        for j in range(len(REGIONS)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax)
    ax.set_title("Correlation between regions")
    return fig


def autocorrelation(series: pd.Series, max_lag: int) -> pd.Series:
    """Correlation of the series with itself shifted by 1..``max_lag`` hours."""
    lags = range(1, max_lag + 1)
    return pd.Series([series.autocorr(lag) for lag in lags], index=lags, name="autocorrelation")


def plot_autocorrelation(series: pd.Series, max_lag: int = 168) -> plt.Figure:
    """How strongly a value is linked to the value ``lag`` hours earlier."""
    acf = autocorrelation(series, max_lag)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(acf.index, acf.values, width=1.0)
    for day in range(24, max_lag + 1, 24):
        ax.axvline(day, color="grey", linestyle=":", linewidth=0.8)
    ax.set(title=f"{series.name}: autocorrelation (dotted lines = whole days)",
           xlabel="Lag (hours)", ylabel="Correlation", ylim=(0, 1))
    return fig


def persistence_errors(series: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """Errors of the naive forecast "value in h hours = value now".

    This is the baseline any real model has to beat.

    Args:
        series: Hourly PM2.5 values for one region.
        horizons: Forecast horizons in hours, e.g. ``[1, 2, ..., 6]``.

    Returns:
        DataFrame indexed by horizon with ``RMSE`` and ``MAE`` columns.
    """
    rows = {}
    for h in horizons:
        error = (series.shift(-h) - series).dropna()
        rows[h] = {"RMSE": np.sqrt((error**2).mean()), "MAE": error.abs().mean()}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("horizon_hours")
