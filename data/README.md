# Raw Data

English | [简体中文](README.zh-CN.md)

## Included sources

- [NEA five-region hourly PM2.5 CSV](singapore-pm25-hourly-raw.csv): 109,584 rows, 2014-04-01 01:00 through 2026-10-01 00:00, in Singapore local time. The first column contains timestamps; the other columns are north, south, east, west and central, in µg/m³. The original file is unchanged.
- [S107 weather provenance](weather_raw_s107/sources.json): 40 files covering five variables and eight years (2017–2024), filtered to East Coast Parkway station S107. Each CSV retains timestamps, station, values, units and publication timestamps, compressed losslessly with gzip. No hourly aggregation or imputation has been applied to these raw files.
- [SHA256 inventory](raw_sources.json): checksums for the PM CSV and all 40 weather observation files, plus expected window counts.

PM source: [NEA Historical 1-hr PM2.5](https://data.gov.sg/datasets/d_6c0d5fc34838b12472475fdb73c0af29/view). Weather source collections: [wind speed](https://data.gov.sg/collections/2280/view), [wind direction](https://data.gov.sg/collections/2281/view), [rainfall](https://data.gov.sg/collections/2279/view), [temperature](https://data.gov.sg/collections/2246/view), [humidity](https://data.gov.sg/collections/2278/view). The weather provenance manifest records per-file dataset IDs, source URLs, station filters, sizes and checksums. Contains NEA data accessed through data.gov.sg under the [Singapore Open Data Licence](https://data.gov.sg/open-data-licence).

## Preparation policy

[PM preparation](../source/prepare_data.py) removes an entire calendar date if it has malformed timestamps, missing/duplicate hours or invalid concentrations. It does not repair timestamps, interpolate or fill values. Its temporary split labels are discarded when joining weather.

[Weather preparation](../source/prepare_weather_data.py) uses historical observations in [t−2h,t−1h) for feature time t, requiring publication no later than t. Temperature/humidity require at least 45 unique minutes per hour with no gap over 15 minutes; wind uses equally sufficient paired speed/direction minutes, converts knots to m/s and calculates u/v. Rainfall requires all twelve five-minute bins; one-second acquisition jitter is normalized to the scheduled grid. Missing rain is never treated as zero.

The requested joint period is 2017–2024. Any date missing one PM or valid weather hour is removed. The 1,753 retained dates are split chronologically at floor(0.8N) and floor(0.9N), giving 1,402/175/176 days. Histories may use known observations before a split boundary, while all six future targets stay in their own split. Windows cannot cross interruptions. There are 25,668/3,331/3,818 legal 24h→6h windows.

## Rebuild

From the repository root:

```bash
python -B -m source.prepare --verify-only
python -B -m source.prepare
python -B main.py --check
```

The [preparation entry point](../source/prepare.py) reads the included raw observations and generates the hourly dataset and legal training windows. See the [project README](../README.md) for training commands. Raw files are checked before rebuilding and are never modified.
