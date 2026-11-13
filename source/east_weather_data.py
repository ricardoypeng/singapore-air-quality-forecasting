"""Load causal east-only concentration/weather windows from prepared start arrays."""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from source.data import load_window_starts

CHANNELS = ("east", "weather_wind_u_ms", "weather_wind_v_ms", "weather_rainfall_mm",
            "weather_temperature_c", "weather_relative_humidity_pct")
WINDOW = 24


@dataclass
class EastWeatherWindows:
    """
    Raw chronological histories and east-only future labels for one split.

    Attributes:
        history      (np.ndarray): Shape (origins, 24, 6), oldest hour first.
        target       (np.ndarray): East concentration, shape (origins, horizon).
        origins      (np.ndarray): Last observed hour of each input window.
        target_times (np.ndarray): Actual future label times.
        start_indices (np.ndarray): Precomputed history starts in concatenated data.
    """

    history: np.ndarray
    target: np.ndarray
    origins: np.ndarray
    target_times: np.ndarray
    start_indices: np.ndarray


def load_windows(data_dir: Path, horizon: int) -> dict[str, EastWeatherWindows]:
    """
    Load east and five weather channels without using other-region concentrations.

    Only stored legal starts are accepted. Histories can precede a split boundary;
    every future target stays in its assigned split. Publication times are checked
    on all source rows, and every selected input/target interval is continuous.
    Nothing is standardized, imputed or shuffled by this loader.

    Args:
        data_dir (Path): Prepared hourly dataset with weather and legal starts.
        horizon  (int): One training label or six recursive evaluation labels.

    Returns:
        dict[str, EastWeatherWindows]: Chronological train, validation and test windows.

    Raises:
        ValueError: If metadata, chronology, causality or stored starts are invalid.
    """
    if horizon not in (1, 6):
        raise ValueError("East weather structure supports training horizon 1 and evaluation horizon 6")
    timestamps, matrices, labels = [], [], []
    for split in ("train", "validation", "test"):
        with (data_dir / f"{split}.pkl").open("rb") as handle:
            part = pickle.load(handle)
        if part["schema_version"] != 2 or part["split"] != split or part["timezone"] != "Asia/Singapore":
            raise ValueError("Expected the version-2 hourly weather dataset")
        if part["weather_station"] != "S107" or part["weather_observation_lag_hours"] != 1:
            raise ValueError("Expected causal S107 observations with the prepared one-hour availability lag")
        if not (len(part["timestamps"]) == len(part["values"]) == len(part["weather_last_published_at"])):
            raise ValueError("Source row lengths differ")
        if any(p > t for p, t in zip(part["weather_last_published_at"], part["timestamps"])):
            raise ValueError("Source weather was not available at its feature timestamp")
        indices = [part["columns"].index(channel) for channel in CHANNELS]
        matrix = np.asarray(part["values"], dtype=np.float64)[:, indices]
        timestamps.extend(part["timestamps"])
        matrices.append(matrix)
        labels.extend([split] * len(matrix))
    times = np.asarray(timestamps, dtype="datetime64[h]")
    values = np.concatenate(matrices)
    label_array = np.asarray(labels)
    if not np.isfinite(values).all() or np.any(np.diff(times) <= np.timedelta64(0, "h")):
        raise ValueError("Expected finite increasing hourly observations")
    starts = load_window_starts(data_dir, WINDOW, horizon)
    result = {}
    for split, origins in starts.items():
        begin = origins - WINDOW + 1
        end = origins + horizon
        if np.any(begin < 0) or np.any(end >= len(times)):
            raise ValueError("Stored window lies outside available rows")
        if np.any(times[end] - times[begin] != np.timedelta64(WINDOW - 1 + horizon, "h")):
            raise ValueError("Stored window crosses a data interruption")
        targets = origins[:, None] + np.arange(1, horizon + 1)
        if not np.all(label_array[targets] == split):
            raise ValueError("Stored target crosses a split boundary")
        result[split] = EastWeatherWindows(
            history=values[origins[:, None] + np.arange(-WINDOW + 1, 1)],
            target=values[targets, 0], origins=times[origins], target_times=times[targets],
            start_indices=begin,
        )
    return result
