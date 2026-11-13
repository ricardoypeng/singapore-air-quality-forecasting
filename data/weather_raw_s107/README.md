# NEA S107 Raw Weather Observations

[Data documentation](../README.md) | [中文数据说明](../README.zh-CN.md)

This directory contains 40 losslessly compressed station-filtered observation CSVs for S107, five variables over 2017–2024. The [provenance manifest](sources.json) lists every observation file, official dataset ID, station filter and SHA256. The [repository checksum inventory](../raw_sources.json) covers these observations and the raw PM CSV.

These files retain original observation values and publication timestamps; they are not the hourly feature matrix. Use the [preparation entry point](../../source/prepare.py) to rebuild inputs. No network download is required.
