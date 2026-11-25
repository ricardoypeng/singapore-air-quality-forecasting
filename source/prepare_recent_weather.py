"""Rebuild S107 legacy and recent weather with release-aware hourly cutoffs."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from source.prepare_weather_data import adequate
from .experiment_paths import PREPARATION, PROJECT, WEATHER
from .experiment_schema import QUALITY_CHANNELS, QUALITY_COLUMNS, RECENT_COLUMNS

START = datetime(2017, 1, 1)
END = datetime(2025, 1, 1)
HOUR = timedelta(hours=1)
SGT_OFFSET = timedelta(hours=8)
RAIN_GRID = set(range(0, 60, 5))
SENSORS = {
    "wind_speed": ({"knots"}, 0, math.inf),
    "wind_direction": ({"degrees"}, 0, 360),
    "rainfall": ({"mm"}, 0, math.inf),
    "temperature": ({"deg C"}, -90, 60),
    "humidity": ({"percentage", "%"}, 0, 100),
}
LEGACY_COLUMNS = (
    "weather_wind_u_ms", "weather_wind_v_ms", "weather_rainfall_mm",
    "weather_temperature_c", "weather_relative_humidity_pct",
)


def minute_groups(path, kind, year):
    """Keep the latest version published by each of the two feature cutoffs."""
    units, low, high = SENSORS[kind]
    groups = [defaultdict(dict), defaultdict(dict)]
    counts = {"rows": 0, "invalid": 0, "eligible": [0, 0], "duplicates": [0, 0]}
    with gzip.open(path, "rt", newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            counts["rows"] += 1
            stamp = datetime.fromisoformat(row["timestamp"])
            if (row["station_id"] != "S107" or stamp.year != year
                    or stamp.utcoffset() != SGT_OFFSET or row["reading_unit"] not in units):
                raise ValueError(f"Unexpected station, year, units or timezone in {path}")
            stamp = stamp.replace(tzinfo=None)
            if kind == "rainfall" and stamp.minute % 5 == 4 and stamp.second == 59 and not stamp.microsecond:
                stamp += timedelta(seconds=1)
            hour = stamp.replace(minute=0, second=0, microsecond=0)
            try:
                published = datetime.fromisoformat(row["reading_update_timestamp"])
                value = float(row["reading_value"])
                if published.utcoffset() != SGT_OFFSET or not math.isfinite(value) or not low <= value <= high:
                    raise ValueError("Invalid publication timezone or physical reading")
            except (ValueError, TypeError):
                counts["invalid"] += 1
                continue
            published = published.replace(tzinfo=None)
            for index, cutoff in enumerate((hour + 2 * HOUR, hour + HOUR)):
                if published > cutoff:
                    continue
                counts["eligible"][index] += 1
                minutes = groups[index][hour]
                previous = minutes.get(stamp.minute)
                if previous is not None:
                    counts["duplicates"][index] += 1
                    if published <= previous[1]:
                        continue
                minutes[stamp.minute] = (value, published)
    return groups, counts


def aggregate(parts, cutoff, hour, *, recent):
    """Aggregate each variable independently and retain its coverage and age."""
    paired = set(parts["wind_speed"]) & set(parts["wind_direction"])
    rain = parts["rainfall"]
    rain_bins = set(rain) & RAIN_GRID
    temp, humidity = parts["temperature"], parts["humidity"]
    values = [np.nan] * 5
    if adequate(paired):
        speeds = [parts["wind_speed"][m][0] * 0.5144444444444445 for m in paired]
        angles = [math.radians(parts["wind_direction"][m][0]) for m in paired]
        values[0] = sum(-s * math.sin(a) for s, a in zip(speeds, angles)) / len(paired)
        values[1] = sum(-s * math.cos(a) for s, a in zip(speeds, angles)) / len(paired)
    if recent and rain_bins:
        values[2] = sum(item[0] for minute, item in rain.items() if minute in rain_bins)
    if not recent and set(rain) == RAIN_GRID:
        values[2] = sum(item[0] for item in rain.values())
    if adequate(set(temp)):
        values[3] = sum(item[0] for item in temp.values()) / len(temp)
    if adequate(set(humidity)):
        values[4] = sum(item[0] for item in humidity.values()) / len(humidity)

    minute_sets = (paired, rain_bins, temp, humidity)
    fractions = [len(minutes) / possible for minutes, possible in zip(minute_sets, (60, 12, 60, 60))]
    age_at_hour_start = (cutoff - hour).total_seconds() / 60
    ages = [age_at_hour_start - max(minutes) if minutes else np.nan for minutes in minute_sets]
    last_publication = max((item[1] for source in parts.values() for item in source.values()), default=None)
    if last_publication is not None and last_publication > cutoff:
        raise AssertionError("A weather reading was released after the feature hour")
    return values, fractions + ages


def fill_missing_ages(quality):
    """Age the last eligible past reading through entirely missing hours."""
    for column in range(len(QUALITY_CHANNELS), quality.shape[1]):
        last_age = np.nan
        for row in range(len(quality)):
            age = quality[row, column]
            if math.isnan(age):
                age = last_age + 60
                quality[row, column] = age
            last_age = age


def verify_legacy(prepared, legacy):
    """Verify reconstructed rows and preserve the exact stored baseline floats."""
    verified = {}
    start = np.datetime64(START, "h")
    for split in ("train", "validation", "test"):
        with (prepared / f"{split}.pkl").open("rb") as stream:
            data = pickle.load(stream)
        hours = (np.array(data["timestamps"], dtype="datetime64[h]") - start).astype(np.int64)
        expected = np.asarray(data["values"])[:, [data["columns"].index(column) for column in LEGACY_COLUMNS]]
        actual = legacy[hours]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
        if any(p > t for p, t in zip(data["weather_last_published_at"], data["timestamps"])):
            raise ValueError("Prepared baseline contains weather released after the feature hour")
        verified[split] = {
            "rows": len(hours),
            "num_rounded_values_before_overlay": int(np.count_nonzero(actual != expected)),
            "max_abs_diff_before_overlay": float(np.abs(actual - expected).max()),
        }
        legacy[hours] = expected
        np.testing.assert_array_equal(legacy[hours], expected)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=WEATHER)
    args = parser.parse_args()
    raw = PROJECT / "data/weather_raw_s107"
    prepared = PREPARATION / "prepared-data/weather_2024"
    sources = json.loads((raw / "sources.json").read_text())["files"]
    times = np.arange(np.datetime64(START, "h"), np.datetime64(END, "h"))
    features = np.full((2, len(times), len(LEGACY_COLUMNS)), np.nan)
    quality = np.full((2, len(times), len(QUALITY_COLUMNS)), np.nan)
    quality[:, :, :len(QUALITY_CHANNELS)] = 0
    audit = {"by_year": {}}
    for year in range(START.year, END.year):
        by_sensor, counters = {}, {}
        for kind in SENSORS:
            path = raw / f"{kind}_{year}_S107.csv.gz"
            if hashlib.sha256(path.read_bytes()).hexdigest() != sources[path.name]["sha256"]:
                raise ValueError(f"Unexpected raw bytes in {path}")
            by_sensor[kind], counters[kind] = minute_groups(path, kind, year)

        for index, lag in enumerate((2, 1)):
            grouped = {kind: groups[index] for kind, groups in by_sensor.items()}
            hours = set().union(*(set(sensor) for sensor in grouped.values()))
            for hour in sorted(hours):
                cutoff = hour + lag * HOUR
                delta = cutoff - START
                position = delta.days * 24 + delta.seconds // 3600
                if 0 <= position < len(times):
                    parts = {kind: grouped[kind].get(hour, {}) for kind in SENSORS}
                    values, measurements = aggregate(parts, cutoff, hour, recent=(lag == 1))
                    features[index, position] = values
                    quality[index, position] = measurements
        audit["by_year"][str(year)] = counters
        print(f"{year} prepared", flush=True)

    for rows in quality:
        fill_missing_ages(rows)
    audit["exact_legacy_rows"] = verify_legacy(prepared, features[0])
    audit["hour_grid"] = {"first": str(times[0]), "last": str(times[-1]), "size": len(times)}
    audit["nonmissing_values"] = {
        "legacy": dict(zip(LEGACY_COLUMNS, np.isfinite(features[0]).sum(axis=0).tolist())),
        "recent": dict(zip(RECENT_COLUMNS[:5], np.isfinite(features[1]).sum(axis=0).tolist())),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_dir / "weather_features.npz",
        times=times, legacy=features[0], legacy_names=np.asarray(LEGACY_COLUMNS),
        recent=np.column_stack((features[1], quality[1])), recent_names=np.asarray(RECENT_COLUMNS),
        quality=quality[0], quality_names=np.asarray(QUALITY_COLUMNS),
    )
    (args.output_dir / "weather_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Verified exact legacy weather rows", audit["exact_legacy_rows"], flush=True)


if __name__ == "__main__":
    main()
