"""Load binary observations and generate windows inside continuous segments."""

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class WindowData:
    """
    Chronological forecasting windows belonging to one target split.

    Attributes:
        x            (np.ndarray): Raw hourly concentrations only, oldest to newest.
        y            (np.ndarray): Future hourly concentration targets.
        origins      (np.ndarray): Last observed timestamp for each window.
        current      (np.ndarray): Concentration at each forecast origin.
        target_times (np.ndarray): Timestamp of every target value.
        segments     (list[dict[str, Any]]): Continuous segment and window-count audit.
    """

    x: np.ndarray
    y: np.ndarray
    origins: np.ndarray
    current: np.ndarray
    target_times: np.ndarray
    segments: list[dict[str, Any]]


def load_data(data_dir: Path, region: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the three chronological binary splits for one region.

    Args:
        data_dir (Path): Directory containing the three prepared Pickle files.
        region   (str): Regional concentration column to forecast.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Hourly timestamps,
            concentrations and original split labels in chronological order.

    Raises:
        ValueError: If split metadata, ordering or concentration values are invalid.
    """
    timestamps, values, labels = [], [], []
    for name in ("train", "validation", "test"):
        with (data_dir / f"{name}.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        if payload["split"] != name or payload["timezone"] != "Asia/Singapore":
            raise ValueError(f"Unexpected metadata in {name}.pkl")
        column = payload["columns"].index(region)
        if len(payload["timestamps"]) != len(payload["values"]):
            raise ValueError(f"Timestamp/value lengths differ in {name}.pkl")
        timestamps.extend(payload["timestamps"])
        values.extend(row[column] for row in payload["values"])
        labels.extend([name] * len(payload["timestamps"]))
    times = np.asarray(timestamps, dtype="datetime64[h]")
    concentrations = np.asarray(values, dtype=np.float64)
    if len(times) == 0 or np.any(np.diff(times) <= np.timedelta64(0, "h")):
        raise ValueError("Timestamps must be nonempty, unique and increasing")
    if not np.all(np.isfinite(concentrations)) or np.any(concentrations < 0):
        raise ValueError("Concentrations must be finite and nonnegative")
    return times, concentrations, np.asarray(labels)


def continuous_segments(times: np.ndarray) -> list[tuple[int, int]]:
    """
    Find maximal runs with exactly one hour between adjacent observations.

    Args:
        times (np.ndarray): Strictly increasing hourly timestamps.

    Returns:
        list[tuple[int, int]]: Segment index pairs, start inclusive and end exclusive.
    """
    if len(times) == 0:
        return []
    boundaries = np.flatnonzero(np.diff(times) != np.timedelta64(1, "h")) + 1
    edges = [0, *boundaries.tolist(), len(times)]
    return list(zip(edges[:-1], edges[1:]))


def load_window_starts(data_dir: Path, window: int, horizon: int) -> dict[str, np.ndarray]:
    """
    Load precomputed window origins and verify their source data fingerprints.

    Args:
        data_dir (Path): Directory containing prepared data and window arrays.
        window   (int): Historical lookback length recorded in the arrays.
        horizon  (int): Future target count recorded in the arrays.

    Returns:
        dict[str, np.ndarray]: Chronological global origin indices for each split.

    Raises:
        ValueError: If metadata, array ordering or data fingerprints do not match.
        FileNotFoundError: If window arrays have not been generated for this lookback.
    """
    fingerprints = np.asarray([
        hashlib.sha256((data_dir / f"{name}.pkl").read_bytes()).hexdigest()
        for name in ("train", "validation", "test")
    ])
    result = {}
    for name in ("train", "validation", "test"):
        suffix = "" if horizon == 6 else f"_horizon{horizon}"
        with np.load(data_dir / f"{name}_windows_{window}h{suffix}.npz", allow_pickle=False) as saved:
            if int(saved["window"]) != window or int(saved["horizon"]) != horizon or str(saved["split"]) != name:
                raise ValueError(f"Window metadata mismatch for {name}; regenerate the arrays")
            if not np.array_equal(saved["source_sha256"], fingerprints):
                raise ValueError(f"Window array source mismatch for {name}; regenerate the arrays")
            origins = saved["origin_indices"].copy()
            if origins.ndim != 1 or not np.issubdtype(origins.dtype, np.integer):
                raise ValueError("Window origin indices must be a one-dimensional integer array")
            if np.any(np.diff(origins) <= 0) or not np.array_equal(saved["start_indices"], origins - window + 1):
                raise ValueError("Invalid ordering or historical start indices")
            if not np.array_equal(saved["target_start_indices"], origins + 1):
                raise ValueError("Invalid target start indices")
            result[name] = origins
    return result


def make_windows(
    times: np.ndarray,
    values: np.ndarray,
    labels: np.ndarray,
    window: int,
    horizon: int,
    comparison_window: int,
    data_dir: Path,
) -> dict[str, WindowData]:
    """
    Load valid starts and construct raw concentration windows within continuous runs.

    Only starts supporting the largest compared lookback are eligible. This
    gives all lookback candidates identical forecast origins. Historical inputs
    may precede a split boundary; every target must belong to the same split.
    Inputs contain only raw hourly concentrations, ordered oldest to newest.
    No statistics or calendar encodings are appended. No window crosses a gap.

    Args:
        times             (np.ndarray): Sorted hourly timestamps.
        values            (np.ndarray): One region's hourly concentrations.
        labels            (np.ndarray): Original split label for each observation.
        window            (int): Number of historical hours used as input.
        horizon           (int): Number of future hourly targets.
        comparison_window (int): Largest lookback in this comparison.
        data_dir          (Path): Directory containing the precomputed start arrays.

    Returns:
        dict[str, WindowData]: Chronological windows and segment audit per split.

    Raises:
        ValueError: If lengths or window settings are invalid, or a split is empty.
    """
    if not (0 < window <= comparison_window and horizon > 0):
        raise ValueError("Require 0 < window <= comparison_window and horizon > 0")
    if not (len(times) == len(values) == len(labels)):
        raise ValueError("Input array lengths differ")
    saved_origins = load_window_starts(data_dir, window, horizon)
    segments = continuous_segments(times)
    segment_ids = np.empty(len(times), dtype=np.int64)
    for segment_id, (start, end) in enumerate(segments):
        segment_ids[start:end] = segment_id
    starts = np.asarray([start for start, _ in segments])
    ends = np.asarray([end for _, end in segments])
    all_origins = []
    for name, origins in saved_origins.items():
        if np.any(origins < window - 1) or np.any(origins + horizon >= len(times)):
            raise ValueError("Precomputed window indices lie outside the observations")
        ids = segment_ids[origins]
        if np.any(origins - window + 1 < starts[ids]) or np.any(origins + horizon >= ends[ids]):
            raise ValueError("Precomputed window crosses a continuous segment boundary")
        if np.any(labels[origins + 1] != name) or np.any(labels[origins + horizon] != name):
            raise ValueError("Precomputed targets cross a split boundary")
        # Restrict short lookbacks to the same origins available to longer ones.
        common = origins - comparison_window + 1 >= starts[ids]
        all_origins.append(origins[common])
    origins = np.sort(np.concatenate(all_origins))
    segment_audit = []
    for segment_id, (start, end) in enumerate(segments):
        in_segment = origins[segment_ids[origins] == segment_id]
        segment_audit.append({
            "start": str(times[start]), "end": str(times[end - 1]),
            "hours": end - start,
            "eligible_after_split_filter": len(in_segment),
            "windows_by_split": {},
        })
        for name in ("train", "validation", "test"):
            valid = (labels[in_segment + 1] == name) & (labels[in_segment + horizon] == name)
            segment_audit[-1]["windows_by_split"][name] = int(valid.sum())
    history = values[origins[:, None] + np.arange(-window + 1, 1)]
    targets = values[origins[:, None] + np.arange(1, horizon + 1)]
    result = {}
    for name in ("train", "validation", "test"):
        mask = (labels[origins + 1] == name) & (labels[origins + horizon] == name)
        indices = origins[mask]
        if len(indices) == 0:
            raise ValueError(f"No eligible windows in {name}")
        result[name] = WindowData(
            x=history[mask], y=targets[mask], origins=times[indices],
            current=values[indices],
            target_times=times[indices[:, None] + np.arange(1, horizon + 1)],
            segments=segment_audit,
        )
    return result
