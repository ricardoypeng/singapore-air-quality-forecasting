"""Model-independent history and calendar feature construction."""
import hashlib
import pickle
from pathlib import Path
from typing import Any
import numpy as np

OTHER_REGIONS = ("north", "south", "west", "central")
EAST_CHANNELS = ("east", "weather_wind_u_ms", "weather_wind_v_ms", "weather_rainfall_mm",
                 "weather_temperature_c", "weather_relative_humidity_pct")

def calendar_features(origins: np.ndarray) -> np.ndarray:
    """
    Encode known origin and future dates without future observations.

    Args:
        origins (np.ndarray): Local origin hours, shape (N,).

    Returns:
        np.ndarray: Hour/weekday/month sine/cosine at offsets zero to six.
    """
    hours = origins.astype("datetime64[h]")[:, None] + np.arange(7).astype("timedelta64[h]")
    hour = hours.astype(np.int64) % 24
    weekday = (hours.astype("datetime64[D]").astype(np.int64) + 3) % 7
    month = hours.astype("datetime64[M]").astype(np.int64) % 12
    blocks = []
    for index in range(7):
        for values, period in ((hour, 24), (weekday, 7), (month, 12)):
            angle = 2 * np.pi * values[:, index] / period
            blocks.extend((np.sin(angle)[:, None], np.cos(angle)[:, None]))
    return np.concatenate(blocks, axis=1)



def spatial_histories(data_dir: Path, parts: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """
    Reconstruct other-region observed histories on established legal windows.

    Args:
        data_dir (Path): Prepared hourly data directory.
        parts    (dict[str, Any]): Existing east-only legal windows.

    Returns:
        tuple[dict[str, np.ndarray], dict[str, Any]]: Other-region histories and audit.
    """
    timestamps, east_blocks, spatial_blocks = [], [], []
    for split in ("train", "validation", "test"):
        with (data_dir / f"{split}.pkl").open("rb") as handle:
            part = pickle.load(handle)
        if part["timezone"] != "Asia/Singapore" or part["split"] != split:
            raise ValueError("Unexpected source timezone or split")
        columns = part["columns"]
        matrix = np.asarray(part["values"], dtype=np.float64)
        timestamps.extend(part["timestamps"])
        east_blocks.append(matrix[:, [columns.index(name) for name in EAST_CHANNELS]])
        spatial_blocks.append(matrix[:, [columns.index(name) for name in OTHER_REGIONS]])
    times = np.asarray(timestamps, dtype="datetime64[h]")
    east, spatial = np.concatenate(east_blocks), np.concatenate(spatial_blocks)
    if not np.isfinite(spatial).all() or not np.isfinite(east).all():
        raise ValueError("Prepared spatial observations must be finite")
    result, audit = {}, {}
    for split, part in parts.items():
        history_indices = part.start_indices[:, None] + np.arange(24)
        if np.any(history_indices < 0) or np.any(history_indices >= len(times)):
            raise ValueError("History starts are outside prepared observations")
        np.testing.assert_array_equal(times[history_indices[:, -1]], part.origins)
        np.testing.assert_array_equal(times[history_indices[:, -1]] - times[history_indices[:, 0]],
                                      np.full(len(part.origins), np.timedelta64(23, "h")))
        np.testing.assert_array_equal(east[history_indices], part.history)
        if np.any(times[history_indices] > part.origins[:, None]):
            raise ValueError("Spatial input includes a future observation")
        result[split] = spatial[history_indices]
        audit[split] = {"windows": len(part.origins), "other_region_shape": list(result[split].shape),
                        "original_east_weather_equal": True, "observations_no_later_than_origin": True,
                        "continuous_24_hours": True,
                        "origin_sha256": hashlib.sha256(part.origins.tobytes()).hexdigest()}
    return result, audit



def design(part: Any, config: dict[str, Any], spatial: np.ndarray) -> np.ndarray:
    """
    Build identical east/calendar blocks with optional additional spatial inputs.

    Args:
        part    (Any): One prepared split.
        config  (dict[str, Any]): Predeclared candidate configuration.
        spatial (np.ndarray): Other-region history, shape (N,24,4).

    Returns:
        np.ndarray: East186 or five-region282 observed/calendar design.
    """
    if part.history.shape[1:] != (24, 6) or spatial.shape != (len(part.history), 24, 4):
        raise ValueError("Unexpected history dimensions")
    blocks = [part.history.reshape(len(part.history), 144), calendar_features(part.origins)]
    if config["scope"] == "five":
        blocks.append(spatial.reshape(len(spatial), 96))
    result = np.column_stack(blocks)
    if result.shape[1] != config["input_schema"]["total_columns"] or not np.isfinite(result).all():
        raise ValueError("Input schema width or finiteness mismatch")
    return result



def input_schema(scope: str) -> dict[str, Any]:
    if scope not in ('east', 'five'):
        raise ValueError(f'Unknown input scope: {scope}')
    return {"east_history_shape": [24, 6], "east_raw_columns": 144,
        "east_channels": list(EAST_CHANNELS), "calendar_columns": 42,
        "calendar_offsets_hours": list(range(7)),
        "calendar_order": "offset-major:hour sin/cos,weekday sin/cos,month sin/cos",
        "other_regions": list(OTHER_REGIONS) if scope == "five" else [],
        "other_region_columns": 96 if scope == "five" else 0,
        "other_region_order": "history-hour-major:north,south,west,central",
        "total_columns": 282 if scope == "five" else 186,
        "column_blocks": ["east144", "calendar42"] + (["other_regions96"] if scope == "five" else []),
        "timezone": "Asia/Singapore"}
