"""Load and clean the raw NEA hourly PM2.5 data.

The raw CSV (``data/singapore-pm25-hourly-raw.csv``) has one row per hour and
one column per region. It is stored in time order, but four rows carry a
mistyped timestamp:

* three rows repeat an earlier hour (e.g. ``7/3/2016 0:00`` written where
  ``7/3/2016 12:00`` should be), which looks like "duplicates" plus
  "missing hours";
* one row uses a different, broken format (``2016-04-04 010:00:00``).

Each of these rows sits exactly where the missing hour belongs, between two
correctly labelled neighbours one hour before and after. So the fix is to
rebuild the timestamp from its position: ``first timestamp + row number``
hours. Every correction is checked against the neighbours and reported, so
nothing is changed silently.

Timestamps are Singapore local time (SGT, UTC+8) and are kept timezone-naive.

Usage from the command line (writes ``data/processed/pm25_hourly_clean.csv``)::

    python -m src.data
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Project paths, resolved relative to this file so the code runs from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "singapore-pm25-hourly-raw.csv"
CLEAN_PATH = PROJECT_ROOT / "data" / "processed" / "pm25_hourly_clean.csv"

# The raw time column is misleadingly named "1hr_pm25"; it holds timestamps.
RAW_TIME_COLUMN = "1hr_pm25"
REGIONS = ["north", "south", "east", "west", "central"]
TIME_FORMAT = "%d/%m/%Y %H:%M"

# Safety limit: if many rows disagree with the hourly grid, the file is not
# simply mistyped (e.g. real gaps or shuffled rows) and needs a human look.
MAX_CORRECTIONS = 10


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV exactly as downloaded, with timestamps as text.

    Args:
        path: Location of the raw NEA CSV file.

    Returns:
        DataFrame with a ``raw_time`` text column and one column per region.
    """
    df = pd.read_csv(path, dtype={RAW_TIME_COLUMN: str})
    df = df.rename(columns={RAW_TIME_COLUMN: "raw_time"})

    missing = set(REGIONS) - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing region columns: {sorted(missing)}")
    return df


def repair_timestamps(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Give every row its true hourly timestamp, based on its position.

    A row is only re-labelled when both neighbours confirm it: the row before
    must be exactly one hour earlier and the row after exactly one hour later
    than the position-based timestamp.

    Args:
        raw: Output of :func:`load_raw`, in original file order.

    Returns:
        A ``(clean, corrections)`` pair. ``clean`` is indexed by
        ``timestamp`` and holds the region columns as floats. ``corrections``
        lists every re-labelled row (row number, original text, new timestamp).

    Raises:
        ValueError: If the first row cannot be parsed, a mismatch is not
            confirmed by its neighbours, or there are more than
            ``MAX_CORRECTIONS`` mismatches.
    """
    parsed = pd.to_datetime(raw["raw_time"], format=TIME_FORMAT, errors="coerce")
    if pd.isna(parsed.iloc[0]):
        raise ValueError(f"Cannot parse the first timestamp: {raw['raw_time'].iloc[0]!r}")

    expected = pd.Series(
        pd.date_range(parsed.iloc[0], periods=len(raw), freq="h"), index=raw.index
    )
    mismatched = parsed != expected  # NaT (unparseable) also counts as a mismatch

    if mismatched.sum() > MAX_CORRECTIONS:
        raise ValueError(
            f"{mismatched.sum()} rows disagree with the hourly grid "
            f"(limit {MAX_CORRECTIONS}); check the file by hand."
        )

    one_hour = pd.Timedelta(hours=1)
    for i in raw.index[mismatched]:
        prev_ok = i == raw.index[0] or parsed[i - 1] == expected[i] - one_hour
        next_ok = i == raw.index[-1] or parsed[i + 1] == expected[i] + one_hour
        if not (prev_ok and next_ok):
            raise ValueError(
                f"Row {i} ({raw['raw_time'][i]!r}) does not fit between its neighbours; "
                "cannot repair it safely."
            )

    corrections = pd.DataFrame(
        {
            "row": raw.index[mismatched],
            "raw_time": raw.loc[mismatched, "raw_time"].values,
            "corrected_time": expected[mismatched].values,
        }
    )

    clean = raw[REGIONS].astype(float)
    clean.index = pd.DatetimeIndex(expected, name="timestamp")
    return clean, corrections


def validate(clean: pd.DataFrame) -> None:
    """Check the cleaned data meets the assumptions later steps rely on.

    Raises:
        ValueError: If timestamps are not a continuous, increasing, unique hourly
            series, or if any concentration is missing or negative.
    """
    index = clean.index
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("Timestamps must be unique and increasing.")
    if len(index) > 1 and (index.to_series().diff().dropna() != pd.Timedelta(hours=1)).any():
        raise ValueError("Timestamps must be exactly one hour apart.")
    if clean[REGIONS].isna().any().any():
        raise ValueError("Concentrations contain missing values.")
    if (clean[REGIONS] < 0).any().any():
        raise ValueError("Concentrations contain negative values.")


def load_clean(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Load the raw file and return the cleaned, validated hourly data.

    This is the function other modules and notebooks should call.

    Args:
        path: Location of the raw NEA CSV file.

    Returns:
        DataFrame indexed by hourly ``timestamp`` with one float column per region.
    """
    clean, _ = repair_timestamps(load_raw(path))
    validate(clean)
    return clean


def main() -> None:
    """Clean the raw file, report what changed, and save the result."""
    raw = load_raw()
    clean, corrections = repair_timestamps(raw)
    validate(clean)

    print(f"Rows: {len(clean):,}  ({clean.index[0]} to {clean.index[-1]})")
    print(f"Repaired {len(corrections)} mislabelled timestamps:")
    print(corrections.to_string(index=False))

    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH)
    print(f"Saved cleaned data to {CLEAN_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
