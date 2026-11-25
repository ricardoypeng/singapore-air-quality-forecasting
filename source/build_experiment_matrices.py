"""Build aligned 24h→6h matrices for the causal weather ablation."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from source.east_weather_data import load_windows
from source.features import EAST_CHANNELS, OTHER_REGIONS, calendar_features, design, input_schema, spatial_histories
from source.prepare_data import REGIONS
from .experiment_paths import PREPARATION, PROJECT, WEATHER
from .experiment_schema import BASE_COLUMNS, QUALITY_COLUMNS, RECENT_COLUMNS


SPLITS = ("train", "validation", "test")
EXPECTED_COUNTS = (25_668, 3_331, 3_818)
TIME_AXIS = np.arange("2017-01-01", "2025-01-01", dtype="datetime64[h]")
HOUR = np.timedelta64(1, "h")
HISTORY_OFFSETS = np.arange(-23, 1)
TARGET_OFFSETS = np.arange(1, 7)

def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pm():
    """Load the previously validated PM-only days on the full experiment axis."""
    clean = PREPARATION / "data-preparation/pm25"
    raw = PROJECT / "data/singapore-pm25-hourly-raw.csv"
    manifest = json.loads((clean / "manifest.json").read_text(encoding="utf-8"))
    source_sha = sha256(raw)
    if source_sha != manifest["source_sha256"]:
        raise ValueError("PM-only preparation does not match the raw project CSV")

    values = np.full((len(TIME_AXIS), len(REGIONS)), np.nan)
    for split in SPLITS:
        with (clean / f"{split}.pkl").open("rb") as handle:
            part = pickle.load(handle)
        if part["schema_version"] != 1 or part["split"] != split or part["columns"] != REGIONS:
            raise ValueError(f"Unexpected schema in archived PM part {split}")
        if part["timezone"] != "Asia/Singapore":
            raise ValueError("PM hours must be in Singapore local time")
        times = np.asarray(part["timestamps"], dtype="datetime64[h]")
        matrix = np.asarray(part["values"], dtype=np.float64)
        if matrix.shape != (len(times), len(REGIONS)) or not np.isfinite(matrix).all() or (matrix < 0).any():
            raise ValueError(f"Invalid concentrations in archived PM part {split}")
        if (np.diff(times) <= np.timedelta64(0, "h")).any():
            raise ValueError(f"Duplicate or unordered hours in archived PM part {split}")
        eligible = (times >= TIME_AXIS[0]) & (times <= TIME_AXIS[-1])
        positions = (times[eligible] - TIME_AXIS[0]).astype(np.int64)
        if np.isfinite(values[positions]).any():
            raise ValueError("PM-only split timestamps overlap")
        values[positions] = matrix[eligible]
    return values, source_sha


def read_historical():
    """Get the baseline matrices from the normal project feature entry point."""
    prepared = PREPARATION / "prepared-data/weather_2024"
    parts = load_windows(prepared, horizon=6)
    spatial, _ = spatial_histories(prepared, parts)
    config = {"scope": "five", "input_schema": input_schema("five")}
    features = {split: design(parts[split], config, spatial[split]) for split in SPLITS}
    if tuple(len(parts[split].origins) for split in SPLITS) != EXPECTED_COUNTS:
        raise ValueError("Historical baseline origin counts differ from the established experiment")
    return parts, features


def positions_for(origins):
    positions = (origins - TIME_AXIS[0]).astype(np.int64)
    if np.any(positions < 23) or np.any(positions + 6 >= len(TIME_AXIS)):
        raise ValueError("Selected origin lacks a full input and target span on the weather hour axis")
    return positions


def predictors(pm, legacy, origins, positions):
    history = positions[:, None] + HISTORY_OFFSETS
    east = pm[history, REGIONS.index("east")]
    east_weather = np.concatenate((east[:, :, None], legacy[history]), axis=2)
    spatial = pm[history][:, :, [REGIONS.index(region) for region in OTHER_REGIONS]]
    return np.column_stack((east_weather.reshape(len(origins), 144), calendar_features(origins),
                            spatial.reshape(len(origins), 96)))


def build(weather_path):
    pm, pm_sha = read_pm()
    original, original_features = read_historical()
    with np.load(weather_path, allow_pickle=False) as weather:
        np.testing.assert_array_equal(weather["times"], TIME_AXIS)
        np.testing.assert_array_equal(weather["legacy_names"], EAST_CHANNELS[1:])
        legacy = weather["legacy"]
        recent = weather["recent"]
        recent_names = weather["recent_names"]
        if not np.array_equal(recent_names, RECENT_COLUMNS):
            raise ValueError("Recent weather columns are missing or out of order")
        quality = weather.get("quality", np.empty((len(TIME_AXIS), 0)))
        quality_names = weather.get("quality_names", np.array([], dtype="U1"))
    if not (np.array_equal(quality_names, QUALITY_COLUMNS) or quality_names.size == 0):
        raise ValueError("Legacy quality columns are out of order")
    if legacy.shape != (len(TIME_AXIS), 5):
        raise ValueError("Expected five legacy weather columns on the original hourly axis")
    for values, names in ((recent, recent_names), (quality, quality_names)):
        if values.shape != (len(TIME_AXIS), len(names)) or len(names) != len(set(names)):
            raise ValueError("Weather matrix and feature names disagree")
    if any(np.isinf(values).any() for values in (legacy, recent, quality)):
        raise ValueError("Weather missing values must be NaN rather than infinity")

    validation_start = original["validation"].target_times[0, 0]
    valid_pm_history_and_future = sliding_window_view(np.isfinite(pm).all(axis=1), 30).all(axis=1)
    potential_origins = TIME_AXIS[23:-6]
    train = potential_origins[valid_pm_history_and_future & (potential_origins + 6 * HOUR < validation_start)]
    if len(train) != 59_601:
        raise ValueError(f"Expected 59,601 expanded training origins but got {len(train):,}")
    origins_by_split = {"train": train, "validation": original["validation"].origins, "test": original["test"].origins}
    baseline_train = np.isin(train, original["train"].origins)
    np.testing.assert_array_equal(train[baseline_train], original["train"].origins)

    result = {
        "original_train": baseline_train,
        "base_names": np.asarray(BASE_COLUMNS),
        "recent_names": recent_names,
        "quality_names": quality_names,
    }
    audit = {"pm_raw_sha256": pm_sha, "weather_archive_sha256": sha256(weather_path),
             "validation_first_target": str(validation_start), "splits": {}}
    for split, origins in origins_by_split.items():
        positions = positions_for(origins)
        y = pm[positions[:, None] + TARGET_OFFSETS, REGIONS.index("east")]
        current = pm[positions, REGIONS.index("east")]
        x = predictors(pm, legacy, origins, positions)
        if x.shape != (len(origins), len(BASE_COLUMNS)) or not np.isfinite(y).all() or not np.isfinite(current).all():
            raise ValueError(f"Invalid PM or predictor matrix in {split}")

        baseline = baseline_train if split == "train" else slice(None)
        np.testing.assert_array_equal(y[baseline], original[split].target)
        np.testing.assert_array_equal(current[baseline], original[split].history[:, -1, 0])
        np.testing.assert_array_equal((origins[:, None] + TARGET_OFFSETS * HOUR)[baseline], original[split].target_times)
        np.testing.assert_allclose(x[baseline], original_features[split], atol=1e-12, rtol=1e-12, equal_nan=False)
        max_difference = float(np.max(np.abs(x[baseline] - original_features[split])))
        if max_difference:
            x[baseline] = original_features[split]
        result.update({f"x_{split}": x, f"y_{split}": y, f"origins_{split}": origins, f"current_{split}": current,
                       f"recent_{split}": recent[positions], f"quality_{split}": quality[positions]})
        audit["splits"][split] = {
            "origins": len(origins), "original_origins": len(original[split].origins),
            "first_origin": str(origins[0]), "last_origin": str(origins[-1]),
            "max_abs_legacy_design_difference_before_exact_overlay": max_difference,
            "origins_with_some_missing_legacy_predictors": int(np.isnan(x).any(axis=1).sum()),
            "h6_ge_35": int((y[:, -1] >= 35).sum()),
        }
    result["audit_json"] = np.asarray(json.dumps(audit, sort_keys=True))
    return result, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", type=Path, default=WEATHER / "weather_features.npz")
    parser.add_argument("--output", type=Path, default=WEATHER / "matrices.npz")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Matrix archive already exists at {args.output}")
    data, audit = build(args.weather)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = args.output.with_name(args.output.name + ".pending")
    try:
        with pending.open("wb") as handle:
            np.savez_compressed(handle, **data)
        pending.replace(args.output)
    finally:
        pending.unlink(missing_ok=True)
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
