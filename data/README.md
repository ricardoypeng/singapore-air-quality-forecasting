# Historical Hourly PM2.5 Data for Singapore

English | [简体中文](README.zh-CN.md)

## Source and provenance

- **Data:** [Original CSV](singapore-pm25-hourly-raw.csv), preserved without cleaning, sorting, or modification.
- **Publisher:** National Environment Agency (NEA), Singapore.
- **Source:** [Historical 1-hr PM2.5](https://data.gov.sg/datasets/d_6c0d5fc34838b12472475fdb73c0af29/view).
- **Dataset ID:** `d_6c0d5fc34838b12472475fdb73c0af29`.
- **Downloaded:** 21 October 2026.
- **Licence:** [Singapore Open Data Licence](https://data.gov.sg/open-data-licence); attribute the source when using the data.
- **Download method:** Request the [official download endpoint](https://api-open.data.gov.sg/v1/public/api/datasets/d_6c0d5fc34838b12472475fdb73c0af29/poll-download), then retrieve the CSV from the temporary URL in the response. Request a fresh URL when it expires.

## Snapshot and schema

- 109,584 data rows and six columns; 3,150,004 bytes.
- First record: 2014-04-01 01:00. Last record: 2026-10-01 00:00. The source page describes coverage as April 2014 through September 2026.
- SHA-256: `2fb76e1c4e4208e92f49670602e303c86e21eaa3704651633323e50f51015de4`.

| Column | Meaning |
| --- | --- |
| `1hr_pm25` | Timestamp, usually day/month/year hour:minute; this column stores time, not concentration |
| `north` | Hourly PM2.5 concentration in the North region |
| `south` | Hourly PM2.5 concentration in the South region |
| `east` | Hourly PM2.5 concentration in the East region |
| `west` | Hourly PM2.5 concentration in the West region |
| `central` | Hourly PM2.5 concentration in the Central region |

Concentrations are in µg/m³. These are regional readings, not PSI/AQI values or campus-specific measurements. Timestamp strings do not include timezone offsets; document the timezone convention during preprocessing.

## Initial quality checks

All five concentration columns contain finite numeric values, with no empty, non-numeric, or negative entries found. The time axis requires investigation:

- CSV line 17,617, counting the header, contains `2016-04-04 010:00:00`. Do not silently interpret it as either 10:00 or 00:00; inspect surrounding records and source evidence first.
- Excluding that malformed row, 109,583 rows parse using `%d/%m/%Y %H:%M`. There are three extra rows with duplicate timestamps: 2016-03-07 00:00, 2018-04-24 00:00, and 2018-08-16 02:00. Concentrations differ between records at the same time, so these are not identical duplicate rows.
- Excluding the malformed row, four timestamps are absent from the continuous hourly grid: 2016-03-07 12:00, 2016-04-04 00:00, 2018-04-25 00:00, and 2018-08-16 03:00. The intended timestamp of the malformed row is unresolved.
- Rows are not strictly ordered chronologically.

No timestamps have been corrected, duplicates resolved, or missing hours filled. Establish and document a cleaning strategy before creating sliding windows: 24 adjacent rows do not necessarily represent 24 consecutive hours. Write derived data separately from this raw snapshot and avoid using future observations to construct historical inputs.
