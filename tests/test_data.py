"""Tests for src/data.py, using small hand-made CSVs plus the real file."""

import pandas as pd
import pytest

from src.data import RAW_PATH, load_clean, load_raw, repair_timestamps

HEADER = "1hr_pm25,north,south,east,west,central\n"


def write_csv(tmp_path, rows):
    """Write a tiny raw-format CSV and return its path."""
    path = tmp_path / "raw.csv"
    path.write_text(HEADER + "".join(f"{r},1,2,3,4,5\n" for r in rows))
    return path


def test_clean_file_needs_no_corrections(tmp_path):
    path = write_csv(tmp_path, ["1/4/2014 1:00", "1/4/2014 2:00", "1/4/2014 3:00"])
    clean, corrections = repair_timestamps(load_raw(path))
    assert corrections.empty
    assert clean.index[-1] == pd.Timestamp("2014-04-01 03:00")


def test_repeated_hour_is_relabelled(tmp_path):
    # "0:00" sits where 12:00 belongs, like row 16955 of the real file.
    path = write_csv(tmp_path, ["7/3/2016 11:00", "7/3/2016 0:00", "7/3/2016 13:00"])
    clean, corrections = repair_timestamps(load_raw(path))
    assert clean.index[1] == pd.Timestamp("2016-03-07 12:00")
    assert len(corrections) == 1


def test_broken_format_is_relabelled(tmp_path):
    path = write_csv(tmp_path, ["3/4/2016 23:00", "2016-04-04 010:00:00", "4/4/2016 1:00"])
    clean, _ = repair_timestamps(load_raw(path))
    assert clean.index[1] == pd.Timestamp("2016-04-04 00:00")


def test_real_gap_is_rejected(tmp_path):
    # 02:00 is genuinely missing, so the rows after it no longer fit the grid.
    path = write_csv(tmp_path, ["1/4/2014 1:00", "1/4/2014 3:00", "1/4/2014 4:00"])
    with pytest.raises(ValueError):
        repair_timestamps(load_raw(path))


def test_real_file_is_continuous_hourly():
    clean = load_clean(RAW_PATH)
    assert len(clean) == 109_584
    assert clean.index[0] == pd.Timestamp("2014-04-01 01:00")
    assert clean.index[-1] == pd.Timestamp("2026-10-01 00:00")
