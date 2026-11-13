"""Join hourly NEA weather and five-region PM2.5 through 2024, without training."""

import argparse
import csv
import gzip
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

COLLECTIONS = {"wind_speed": 2280, "wind_direction": 2281, "rainfall": 2279,
               "temperature": 2246, "humidity": 2278}
from source.prepare_data import REGIONS, save_window_starts

ROOT = Path(__file__).resolve().parents[1]
WEATHER_COLUMNS = ["weather_wind_speed_ms", "weather_wind_direction_deg",
                   "weather_wind_u_ms", "weather_wind_v_ms", "weather_rainfall_mm",
                   "weather_temperature_c", "weather_relative_humidity_pct"]
UNITS = {**dict.fromkeys(REGIONS, "ug/m3"), **dict(zip(WEATHER_COLUMNS,
         ["m/s", "degree", "m/s", "m/s", "mm", "deg C", "%"]))}


def read_station(path: Path, kind: str, station: str, year: int) -> tuple[dict[datetime, tuple[float, datetime]], dict[str, int]]:
    """
    Validate minute observations and retain the latest version available in time.

    A weather observation in hour h is eligible for the feature row at h+2h.
    This leaves one hour between its completed observation hour and feature use.
    Revised records published after that cutoff cannot enter that feature row.

    Args:
        path    (Path): Filtered official CSV compressed with gzip.
        kind    (str): Weather variable key.
        station (str): Expected station identifier.
        year    (int): Expected source calendar year.

    Returns:
        tuple[dict[datetime, tuple[float, datetime]], dict[str, int]]:
            Unique observation minutes and row-quality counters.
    """
    values, counts = {}, Counter()
    expected_units = {"wind_speed": {"knots"}, "wind_direction": {"degrees"},
                      "rainfall": {"mm"}, "temperature": {"deg C"},
                      "humidity": {"percentage", "%"}}
    with gzip.open(path, "rt", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            counts["source_rows"] += 1
            if row["station_id"] != station:
                raise ValueError(f"Unexpected station in {path.name}")
            stamp = datetime.fromisoformat(row["timestamp"])
            if stamp.utcoffset() != timedelta(hours=8) or stamp.year != year:
                raise ValueError(f"Unexpected time zone/year in {path.name}")
            if row["reading_unit"] not in expected_units[kind]:
                raise ValueError(f"Unexpected {kind} unit: {row['reading_unit']}")
            stamp = stamp.replace(tzinfo=None)
            # Early official rain records use HH:04:59 rather than HH:05:00.
            # Snap only this one-second acquisition jitter to the scheduled grid.
            if kind == "rainfall" and stamp.minute % 5 == 4 and stamp.second == 59 and stamp.microsecond == 0:
                stamp += timedelta(seconds=1)
                counts["rainfall_one_second_grid_normalizations"] += 1
            stamp = stamp.replace(second=0, microsecond=0)
            cutoff = stamp.replace(minute=0) + timedelta(hours=2)
            try:
                published = datetime.fromisoformat(row["reading_update_timestamp"])
                if published.utcoffset() != timedelta(hours=8):
                    raise ValueError("Invalid publication time zone")
                published = published.replace(tzinfo=None)
                value = float(row["reading_value"])
                valid = math.isfinite(value)
                if kind in ("wind_speed", "rainfall"):
                    valid = valid and value >= 0
                elif kind == "wind_direction":
                    valid = valid and 0 <= value <= 360
                elif kind == "humidity":
                    valid = valid and 0 <= value <= 100
                else:
                    valid = valid and -90 <= value <= 60
                if not valid:
                    raise ValueError("Invalid physical reading")
            except (ValueError, TypeError):
                counts["invalid_rows"] += 1
                continue
            if published > cutoff:
                counts["published_after_feature_cutoff"] += 1
                continue
            if stamp in values:
                counts["duplicate_minute_rows"] += 1
                if published <= values[stamp][1]:
                    continue
            values[stamp] = (value, published)
    counts["unique_eligible_minutes"] = len(values)
    return values, dict(counts)


def group_hours(values: dict[datetime, tuple[float, datetime]]) -> dict[datetime, dict[int, tuple[float, datetime]]]:
    """
    Group unique eligible minutes by their observation hour.

    Args:
        values (dict[datetime, tuple[float, datetime]]): Minute readings and publication times.

    Returns:
        dict[datetime, dict[int, tuple[float, datetime]]]: Per-hour minute maps.
    """
    groups = defaultdict(dict)
    for stamp, value in values.items():
        groups[stamp.replace(minute=0)][stamp.minute] = value
    return dict(groups)


def adequate(minutes: set[int], minimum: int = 45) -> bool:
    """
    Require at least 75 percent minute coverage and no gap longer than 15 minutes.

    Args:
        minutes (set[int]): Observed minute positions in one hour.
        minimum (int): Required unique minute count.

    Returns:
        bool: Whether the observed hourly coverage passes both rules.
    """
    ordered = [-1, *sorted(minutes), 60]
    return len(minutes) >= minimum and max(b - a - 1 for a, b in zip(ordered, ordered[1:])) <= 15


def hourly_weather(raw_dir: Path, station: str, start_year: int, end_year: int) -> tuple[dict[datetime, tuple[list[float], datetime]], dict[str, Any]]:
    """
    Aggregate co-located weather observations into causal hourly feature rows.

    Rainfall requires every five-minute bin, with no interpolation or rescaling.
    Wind components use matched speed/direction minutes and meteorological
    direction-from convention. Temperature/humidity use observed-minute means.

    Args:
        raw_dir    (Path): Downloaded station observations and source manifest.
        station    (str): Co-located weather station.
        start_year (int): First source calendar year.
        end_year   (int): Last source calendar year.

    Returns:
        tuple[dict[datetime, tuple[list[float], datetime]], dict[str, Any]]:
            Feature rows indexed by availability-safe hour and quality audit.
    """
    output, audit = {}, {}
    for year in range(start_year, end_year + 1):
        groups, counters = {}, {}
        for kind in COLLECTIONS:
            path = raw_dir / f"{kind}_{year}_{station}.csv.gz"
            minutes, counters[kind] = read_station(path, kind, station, year)
            groups[kind] = group_hours(minutes)
            del minutes
        failures = Counter()
        hours = sorted(set().union(*(set(group) for group in groups.values())))
        for hour in hours:
            part = {kind: groups[kind].get(hour, {}) for kind in COLLECTIONS}
            paired = set(part["wind_speed"]) & set(part["wind_direction"])
            checks = {"wind_coverage": adequate(paired),
                      "temperature_coverage": adequate(set(part["temperature"])),
                      "humidity_coverage": adequate(set(part["humidity"])),
                      "rainfall_complete_12_bins": set(part["rainfall"]) == set(range(0, 60, 5))}
            for reason, passed in checks.items():
                if not passed:
                    failures[reason] += 1
            if not all(checks.values()):
                continue
            speeds = [part["wind_speed"][m][0] * 0.5144444444444445 for m in paired]
            angles = [math.radians(part["wind_direction"][m][0]) for m in paired]
            u = sum(-s * math.sin(a) for s, a in zip(speeds, angles)) / len(paired)
            v = sum(-s * math.cos(a) for s, a in zip(speeds, angles)) / len(paired)
            direction = math.degrees(math.atan2(-u, -v)) % 360 if math.hypot(u, v) > 1e-12 else 0.0
            row = [sum(speeds) / len(speeds), direction, u, v,
                   sum(value[0] for value in part["rainfall"].values()),
                   sum(value[0] for value in part["temperature"].values()) / len(part["temperature"]),
                   sum(value[0] for value in part["humidity"].values()) / len(part["humidity"])]
            published = max(value[1] for kind in part for value in part[kind].values())
            feature_time = hour + timedelta(hours=2)
            if published > feature_time:
                raise ValueError("Weather feature contains information published in the future")
            output[feature_time] = (row, published)
        audit[str(year)] = {"variables": counters, "candidate_observation_hours": len(hours),
                            "invalid_hour_reasons": dict(failures),
                            "valid_weather_hours": sum(t.year == year for t in output)}
        print(f"Aggregated {year}: {len(output):,} cumulative hourly weather rows", flush=True)
    return output, audit


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """
    Remove incomplete dates, split remaining dates 80/10/10 and save binary data.

    Existing cleaned PM2.5 splits are concatenated to recover observations only;
    their historical split labels are discarded. All five concentration columns
    are retained. Weather and PM2.5 gaps invalidate the entire calendar day.
    No imputation, training, standardization or random split is performed.

    Args:
        args (argparse.Namespace): Raw weather, existing PM2.5 and new output paths.

    Returns:
        dict[str, Any]: Source fingerprints, removed dates, splits and legal windows.
    """
    if (args.output_dir / "manifest.json").exists() and not args.overwrite:
        raise FileExistsError("Prepared dataset already exists; use --overwrite explicitly")
    source = json.loads((args.raw_dir / "sources.json").read_text())
    for filename, item in source["files"].items():
        if hashlib.sha256((args.raw_dir / filename).read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError(f"Raw checksum mismatch: {filename}")
    weather, audit = hourly_weather(args.raw_dir, args.station, args.start_year, args.end_year)
    pm, fingerprints = {}, {}
    for split in ("train", "validation", "test"):
        path = args.pm_dir / f"{split}.pkl"
        fingerprints[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        payload = pickle.loads(path.read_bytes())
        if payload["columns"] != REGIONS or payload["timezone"] != "Asia/Singapore":
            raise ValueError("Unexpected original PM2.5 schema")
        for stamp, row in zip(payload["timestamps"], payload["values"]):
            if stamp in pm or not all(math.isfinite(v) and v >= 0 for v in row):
                raise ValueError("Duplicate or invalid original PM2.5 observation")
            pm[stamp] = row
    first, last = datetime(args.start_year, 1, 1), datetime(args.end_year, 12, 31)
    clean, removed = [], {}
    day = first
    while day <= last:
        hours = [day + timedelta(hours=h) for h in range(24)]
        missing_pm = [t.hour for t in hours if t not in pm]
        missing_weather = [t.hour for t in hours if t not in weather]
        if missing_pm or missing_weather:
            removed[day.date().isoformat()] = {"missing_pm25_hours": missing_pm,
                                               "invalid_or_missing_weather_hours": missing_weather}
        else:
            clean.append(day)
        day += timedelta(days=1)
    cut1, cut2 = len(clean) * 8 // 10, len(clean) * 9 // 10
    groups = {"train": clean[:cut1], "validation": clean[cut1:cut2], "test": clean[cut2:]}
    if any(not days for days in groups.values()):
        raise ValueError("Insufficient complete dates after weather alignment")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 2, "station": args.station,
        "station_name": "East Coast Parkway", "station_coordinates": {"latitude": 1.3135, "longitude": 103.9625},
        "timezone": "Asia/Singapore", "requested_start": first.isoformat(), "requested_end": (last + timedelta(hours=23)).isoformat(),
        "columns": REGIONS + WEATHER_COLUMNS, "units": UNITS,
        "column_roles": {"regional_concentrations": REGIONS, "observed_weather_covariates": WEATHER_COLUMNS,
                         "forecast_target": "east", "future_observed_weather_allowed": False},
        "weather_alignment": "Feature row t aggregates observation timestamps in [t-2h,t-1h); publication <= t required",
        "aggregation": {"wind": "matched-minute vector mean; >=45 matched minutes and no gap >15 minutes; scalar speed mean; knots converted to m/s",
            "temperature_humidity": "observed-minute means; >=45 unique minutes and no gap >15 minutes",
            "rainfall": "normalize HH:04:59-type one-second jitter to next scheduled five-minute tick; sum twelve unique totals HH:00,05,...,55; missing is never zero",
            "wind_direction": "direction from vector mean; 0 if vector magnitude <=1e-12"},
        "split_rule": "Chronological complete dates; cuts floor(0.8*N), floor(0.9*N)",
        "day_rule": "Remove whole calendar date if any PM2.5/weather feature hour is unavailable",
        "original_pm25_sha256": fingerprints, "weather_sources": source,
        "weather_quality": audit, "removed_dates": removed, "retained_days": len(clean), "splits": {}}
    with gzip.open(args.output_dir / "hourly.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", *manifest["columns"], "weather_last_published_at", "split"])
        for name, days in groups.items():
            timestamps = [day + timedelta(hours=h) for day in days for h in range(24)]
            values = [pm[t] + weather[t][0] for t in timestamps]
            published = [weather[t][1] for t in timestamps]
            payload = {"schema_version": 2, "split": name, "timestamps": timestamps,
                "columns": manifest["columns"], "values": values, "timezone": "Asia/Singapore",
                "unit": "mixed; see units", "units": UNITS, "weather_station": args.station,
                "weather_last_published_at": published, "weather_observation_lag_hours": 1}
            path = args.output_dir / f"{name}.pkl"
            with path.open("wb") as target:
                pickle.dump(payload, target, protocol=4)
            for t, row, publication in zip(timestamps, values, published):
                writer.writerow([t.isoformat(), *row, publication.isoformat(), name])
            manifest["splits"][name] = {"days": len(days), "rows": len(timestamps),
                "start": timestamps[0].isoformat(), "end": timestamps[-1].isoformat(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest["window_starts"] = save_window_starts(args.output_dir, [24, 48], 6)
    manifest["single_step_window_starts"] = save_window_starts(args.output_dir, [24, 48], 1)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    write_readme(args.output_dir, manifest)
    print(json.dumps({"retained_days": len(clean), "removed_days": len(removed), "splits": manifest["splits"]}, indent=2))
    return manifest


def write_readme(output_dir: Path, manifest: dict[str, Any]) -> None:
    """
    Describe the completed dataset, causal aggregation and binary interfaces.

    Args:
        output_dir (Path): Completed binary dataset directory.
        manifest   (dict[str, Any]): Audited preparation metadata.

    Returns:
        None.
    """
    lines = ["# 截至 2024 年的 PM2.5 与气象小时数据", "",
        "五区域 PM2.5 加东区 East Coast Parkway（S107）同站气象观测。按完整日期清洗，按时间顺序切分 8:1:1。原 PM2.5 数据版本保留。", "",
        "## 数据与来源", "",
        "- [全部小时数据 CSV（gzip）](hourly.csv.gz)、[完整元数据及剔除日期](manifest.json)。",
        "- [官方原始观测与来源说明](../../../../data/weather_raw_s107/README.md)、[原始文件指纹](../../../../data/weather_raw_s107/sources.json)。",
        "- [下载程序](../../../../source/prepare.py)、[转换程序](../../../../source/prepare_weather_data.py)、[无训练检查程序](../../../../source/prepare.py)。", "",
        "| 分区 | 日期范围（SGT） | 完整天数 | 小时数 |", "| --- | --- | ---: | ---: |"]
    for name, item in manifest["splits"].items():
        lines.append(f"| [{name}]({name}.pkl) | {item['start']} 至 {item['end']} | {item['days']} | {item['rows']} |")
    lines += ["", "## 列与单位", "", "| 列 | 单位 |", "| --- | --- |"]
    for column in manifest["columns"]:
        lines.append(f"| `{column}` | {UNITS[column]} |")
    lines += ["", "## 小时对齐与缺测", "",
        "- 时间戳 t 的 PM2.5 为原始区域小时观测；气象按观测时间戳落在 [t-2h,t-1h) 的记录分箱，并要求记录的读数更新时间不晚于 t。预留一小时发布延迟，不使用未来实测气象。",
        "- 温湿度为有效分钟均值；每小时至少 45 个不同分钟，连续缺口不超过 15 分钟。风速/风向要求至少 45 个配对分钟，并满足相同缺口限制。",
        "- 风速从 knots 换算为 m/s；u/v 按气象学来向定义计算并求平均；风向由平均矢量还原。平均矢量近零时角度约定为 0，应优先使用 u/v 表示。",
        "- 小时雨量为观测时间戳 HH:00、05、……、55 的全部 12 个五分钟累计读数之和，采用明确的左闭右开时间戳分箱。缺一个时间片，该小时无效；缺测不视为无雨。",
        "- 早期雨量 HH:04:59 等距下个五分钟刻度恰好一秒的记录，归整到 HH:05:00；归整数量单列记录。原始文件不修改，不补造记录。",
        "- 不插值、不填零、不标准化。若一天任何小时缺 PM2.5 或有效气象，该整天剔除。",
        "- 以剩余完整日期数量 N 计算 floor(0.8N)、floor(0.9N) 切分位置；所以日期范围内可能存在已剔除的中断。", "",
        "## 二进制读取与窗口", "",
        "三个 Pickle 文件使用 protocol 4，包含 timestamps、values、columns、units、split、timezone 和 weather_last_published_at。values 的形状为 (小时数, 12)，列序以 columns 为准。时间为不带 tzinfo 的新加坡本地 datetime。",
        "", "```python", "import pickle", "from pathlib import Path", "",
        "data_dir = Path('data/processed_weather_2024')", "with (data_dir / 'train.pkl').open('rb') as handle:",
        "    train = pickle.load(handle)", "east_index = train['columns'].index('east')", "```", "",
        "所有窗口数组索引指向 train、validation、test 顺序拼接后的小时数据。输入可引用分区边界之前已经观测到的历史；所有未来目标必须在自己的分区内。起点仅在连续小时段中枚举，提供 24/48 小时输入、1/6 小时目标两种数组。", ""]
    for group in ("single_step_window_starts", "window_starts"):
        for filename, item in manifest[group]["files"].items():
            lines.append(f"- [{filename}]({filename})：{item['count']} 个合法起点。")
    lines += ["", "数组包含 start_times、origin_times、start_indices、origin_indices、target_start_indices、segment_ids 和三个源 Pickle 的 SHA-256。", "",
        "当前旧训练读取接口按 region 只取浓度列，不会自动使用新增气象列。天气模型须显式选择协变量并定义递归更新，未来真实气象不能作为预测输入。已实现的[东区专用输入与模型结构](../../../../README.zh-CN.md)使用 24×6 通道和未知天气持久性假设，通过统一入口的 east-weather 子命令调用。本次数据生成没有训练模型。", ""]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """
    Parse the new dataset paths and prepare complete hourly observations.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/weather_raw_s107")
    parser.add_argument("--pm-dir", type=Path, default=ROOT / "tmp/_project/data-preparation/pm25")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tmp/_project/prepared-data/weather_2024")
    parser.add_argument("--station", choices=["S107"], default="S107")
    parser.add_argument("--start-year", type=int, default=2017)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    prepare(args)


if __name__ == "__main__":
    main()
