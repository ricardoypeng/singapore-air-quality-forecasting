"""
Remove abnormal calendar days and save chronological 80/10/10 splits.

Run from any directory in the hw310 environment. NumPy saves valid window arrays.
"""

import argparse
import csv
import hashlib
import json
import math
import pickle
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGIONS = ["north", "south", "east", "west", "central"]


def save_window_starts(
    output_dir: Path, windows: list[int], horizon: int,
) -> dict[str, Any]:
    """
    Save all valid window starts inside continuous segments of prepared data.

    Indices address the chronological concatenation of train, validation and
    test observations. History may use observations from the preceding split;
    all future targets must belong to the requested split. No shuffling occurs.

    Args:
        output_dir (Path): Directory containing the three prepared Pickle files.
        windows    (list[int]): Historical window lengths to enumerate.
        horizon    (int): Number of consecutive future hourly targets required.

    Returns:
        dict[str, Any]: Continuous segment descriptions and window array metadata.

    Raises:
        ValueError: If lookback or horizon lengths are not positive.
    """
    if not windows or min(windows) < 1 or horizon < 1:
        raise ValueError("Window lengths and horizon must be positive")
    all_times, all_labels = [], []
    for name in ("train", "validation", "test"):
        with (output_dir / f"{name}.pkl").open("rb") as handle:
            payload = pickle.load(handle)
        all_times.extend(payload["timestamps"])
        all_labels.extend([name] * len(payload["timestamps"]))
    times = np.asarray(all_times, dtype="datetime64[h]")
    labels = np.asarray(all_labels)
    breaks = np.flatnonzero(np.diff(times) != np.timedelta64(1, "h")) + 1
    edges = np.asarray([0, *breaks.tolist(), len(times)], dtype=np.int64)
    segment_starts, segment_ends = edges[:-1], edges[1:]
    info = {"horizon": horizon, "index_space": "concatenate train, validation, test in this order",
            "segments": [{"id": i, "start_index": int(start), "end_index_exclusive": int(end),
                          "start": str(times[start]), "end": str(times[end - 1]), "hours": int(end - start)}
                         for i, (start, end) in enumerate(zip(segment_starts, segment_ends))],
            "files": {}}
    for window in sorted(set(windows)):
        origin_parts, segment_parts = [], []
        for segment_id, (start, end) in enumerate(zip(segment_starts, segment_ends)):
            origins = np.arange(start + window - 1, end - horizon, dtype=np.int64)
            origin_parts.append(origins)
            segment_parts.append(np.full(len(origins), segment_id, dtype=np.int64))
        origins = np.concatenate(origin_parts)
        segment_ids = np.concatenate(segment_parts)
        for name in ("train", "validation", "test"):
            mask = (labels[origins + 1] == name) & (labels[origins + horizon] == name)
            selected = origins[mask]
            starts = selected - window + 1
            suffix = "" if horizon == 6 else f"_horizon{horizon}"
            filename = f"{name}_windows_{window}h{suffix}.npz"
            np.savez_compressed(
                output_dir / filename, start_times=times[starts], start_indices=starts,
                origin_times=times[selected], origin_indices=selected,
                target_start_indices=selected + 1, segment_ids=segment_ids[mask],
                segment_start_indices=segment_starts, segment_end_indices=segment_ends,
                window=np.asarray(window), horizon=np.asarray(horizon),
                split=np.asarray(name), source_sha256=np.asarray([
                    hashlib.sha256((output_dir / f"{part}.pkl").read_bytes()).hexdigest()
                    for part in ("train", "validation", "test")
                ]),
            )
            info["files"][filename] = {"split": name, "window": window, "count": len(starts)}
    return info


def prepare_data(raw_path: Path, output_dir: Path) -> dict[str, Any]:
    """
    Remove abnormal days and save chronological training, validation and test data.

    Each retained date contains exactly 24 hourly observations. Split the
    remaining dates in chronological order at 80% and 90%, rounding down.
    Save three Pickle files, 24/48-hour six-step evaluation start arrays,
    24-hour single-step training start arrays and a JSON manifest.

    Args:
        raw_path   (Path): Path to the original hourly PM2.5 CSV file.
        output_dir (Path): Directory in which to save the binary data and manifest.

    Returns:
        dict[str, Any]: Manifest containing the source checksum, removed dates
            and reasons, row counts, and date ranges for all three splits.

    Raises:
        ValueError: If required columns are missing, a record's date cannot be
            identified, the input has no dated records, or too few complete
            dates remain to create three nonempty splits.
    """
    records = defaultdict(list)
    reasons = defaultdict(set)
    raw_rows = 0
    with raw_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"1hr_pm25", *REGIONS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Required CSV columns: {sorted(required)}")
        for line_number, row in enumerate(reader, start=2):
            raw_rows += 1
            text = row["1hr_pm25"].strip()
            try:
                timestamp = datetime.strptime(text, "%d/%m/%Y %H:%M")
            except ValueError:
                # Recover only the date, never guess the malformed hour.
                match = re.match(r"^(\d{4}-\d{2}-\d{2})(?:\s|$)", text)
                if match:
                    day = datetime.strptime(match[1], "%Y-%m-%d").date()
                else:
                    match = re.match(r"^(\d{1,2}/\d{1,2}/\d{4})(?:\s|$)", text)
                    if not match:
                        raise ValueError(f"Cannot identify date at CSV line {line_number}: {text!r}")
                    day = datetime.strptime(match[1], "%d/%m/%Y").date()
                reasons[day].add("invalid timestamp format")
                records[day]  # Include dates containing only malformed timestamps.
                continue
            day = timestamp.date()
            try:
                values = [float(row[region]) for region in REGIONS]
                if any(not math.isfinite(value) or value < 0 for value in values):
                    raise ValueError("non-finite or negative concentration")
            except (ValueError, TypeError):
                reasons[day].add("invalid PM2.5 concentration")
                values = None
            records[day].append((timestamp, values))

    if not records:
        raise ValueError("The CSV contains no dated records")
    day = min(records)
    last_day = max(records)
    while day <= last_day:
        rows = records[day]
        hours = Counter(timestamp.hour for timestamp, _ in rows)
        missing = sorted(set(range(24)) - hours.keys())
        duplicated = sorted(hour for hour, count in hours.items() if count > 1)
        if missing:
            reasons[day].add(f"missing hours: {missing}")
        if duplicated:
            reasons[day].add(f"duplicate hours: {duplicated}")
        day += timedelta(days=1)

    clean_days = sorted(day for day in records if day not in reasons)
    cut1 = len(clean_days) * 8 // 10
    cut2 = len(clean_days) * 9 // 10
    groups = {"train": clean_days[:cut1], "validation": clean_days[cut1:cut2], "test": clean_days[cut2:]}
    if any(not days for days in groups.values()):
        raise ValueError("Not enough complete days for three nonempty splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "format": "Python pickle, protocol 4",
        "source_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "raw_rows": raw_rows,
        "timezone_convention": "Asia/Singapore local time; stored timestamps are naive datetimes",
        "columns": REGIONS,
        "unit": "ug/m3",
        "split_rule": "Sort remaining complete dates; cuts at floor(N*0.8) and floor(N*0.9)",
        "removed_dates": {str(day): sorted(items) for day, items in sorted(reasons.items())},
        "splits": {},
    }
    for name, days in groups.items():
        rows = sorted(record for day in days for record in records[day])
        payload = {
            "schema_version": 1,
            "split": name,
            "columns": REGIONS,
            "timestamps": [timestamp for timestamp, _ in rows],
            "values": [values for _, values in rows],
            "timezone": "Asia/Singapore",
            "unit": "ug/m3",
        }
        filename = f"{name}.pkl"
        with (output_dir / filename).open("wb") as handle:
            pickle.dump(payload, handle, protocol=4)
        manifest["splits"][name] = {
            "file": filename, "days": len(days), "rows": len(rows),
            "start": rows[0][0].isoformat(), "end": rows[-1][0].isoformat(),
        }
    manifest["clean_rows"] = sum(item["rows"] for item in manifest["splits"].values())
    manifest["removed_rows"] = raw_rows - manifest["clean_rows"]
    manifest["window_starts"] = save_window_starts(output_dir, [24, 48], horizon=6)
    manifest["single_step_window_starts"] = save_window_starts(output_dir, [24], horizon=1)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    """
    Parse command-line arguments, prepare the data and print the manifest.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "data/singapore-pm25-hourly-raw.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "tmp/_project/data-preparation/pm25")
    args = parser.parse_args()
    manifest = prepare_data(args.input, args.output_dir)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
