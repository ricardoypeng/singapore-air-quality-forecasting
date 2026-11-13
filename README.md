# Singapore Air Quality Forecasting

English | [简体中文](README.zh-CN.md)

A project exploring air pollution forecasting under the theme of AI for Social Good.

## Goal

Use a sliding window of the most recent 24–48 hours of PM2.5 observations to predict hourly concentrations for the next 1–6 hours in Singapore, helping people plan travel and outdoor activities.

## Results at a glance

East region, test period 2024-01 → 2026-09 (never used for any modelling choice):

| Model | Mean RMSE (1–6 h) | Mean MAE (1–6 h) | Mean RMSE on elevated hours (≥ 56 µg/m³) |
| --- | --- | --- | --- |
| Persistence baseline ("stays the same") | 6.51 | 4.63 | **24.0** |
| Same hour yesterday baseline | 9.28 | 6.30 | 47.6 |
| Ridge regression | 5.65 | 4.02 | 26.2 |
| Random forest | 5.62 | 4.02 | 25.1 |
| **Gradient boosting (selected on validation)** | **5.63** | **4.01** | 25.9 |

- The selected model cuts RMSE by **7 % at 1 hour and 16 % at 6 hours** compared
  with persistence, and beats it in all five regions.
- **Main weakness:** during haze (elevated hours) it under-predicts and is
  slightly worse than persistence. Sudden haze spikes cannot be foreseen from
  past PM2.5 alone. See `notebooks/02_modelling.ipynb`, section 4.

All numbers come from `results/` and can be recreated with the commands below.

## Repository structure

```
data/
  singapore-pm25-hourly-raw.csv   Original NEA data (unchanged)
  README.md                       Data source, fields and quality checks
src/
  data.py        Load the raw CSV and repair 4 mistyped timestamps
  features.py    Lag, rolling and calendar features; time-based split
  models.py      Baselines, candidate models, hyper-parameter grids
  metrics.py     RMSE / MAE per horizon, also on elevated (haze) hours
  train.py       End-to-end pipeline: tune, test, pick best, region audit
  evaluate.py    Evaluation figures and the worst-error table
  predict.py     Batch prediction demo for the next 1–6 hours
  eda.py         Plots used in the exploration notebook
notebooks/
  01_data_exploration.ipynb   What the data looks like and how it shapes the model
  02_modelling.ipynb          Model comparison, evaluation and failure cases
reports/figures/  All figures (recreated by the commands below)
results/          Metrics tables and summary (recreated by src.train)
tests/            Automated tests (pytest)
```

## Installation

Requires Python 3.12. Developed on macOS with an Apple M-series CPU (10 cores);
no GPU is needed.

```bash
git clone https://github.com/ricardoypeng/singapore-air-quality-forecasting.git
cd singapore-air-quality-forecasting
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dataset is included in `data/`, so nothing else needs to be downloaded.

## Reproducing every result

Run these from the project folder with the environment activated. All random
seeds are fixed (`random_state=42`).

| Step | Command | Output | Time* |
| --- | --- | --- | --- |
| 1. Clean the data (optional) | `python -m src.data` | `data/processed/pm25_hourly_clean.csv` | seconds |
| 2. Train and compare models | `python -m src.train` | `results/`, `models/best_model.joblib` | ~2 min |
| 3. Evaluation figures | `python -m src.evaluate` | `reports/figures/07_*` to `13_*`, `results/worst_errors.csv` | ~10 s |
| 4. Exploration figures | run `notebooks/01_data_exploration.ipynb` | `reports/figures/01_*` to `06_*` | seconds |
| 5. Forecast demo | `python -m src.predict` | forecast for the next 1–6 hours | < 1 s |
| Tests | `pytest` | all tests pass | ~15 s |

\*On a 10-core laptop. Every step cleans the raw data itself, so step 1 is only
needed to inspect the cleaned file. `python -m src.train --quick` runs tiny
grids for a fast check.

`models/best_model.joblib` is not stored in the repository (it is rebuilt by
step 2).

### Docker (optional)

```bash
docker build -t sg-pm25 .
docker run --rm -v "$PWD/results:/app/results" -v "$PWD/models:/app/models" sg-pm25
docker run --rm -v "$PWD/models:/app/models" sg-pm25 python -m src.predict
```

## Method

1. **Cleaning** (`src/data.py`): the raw file is a continuous hourly series in
   time order, but 4 rows carry a mistyped timestamp exactly where an hour
   seemed to be "missing". Each is re-labelled from its position after
   checking both neighbours.
2. **Features** (`src/features.py`): for forecast origin *t*: the last 24
   hourly values of the target region; the last 3 hours of the other four
   regions; 6 h and 24 h rolling mean / std / max; the 1-hour change; the
   island-wide mean; hour and month (sine/cosine) and day of week (49 in total).
   Targets are the values at *t*+1 … *t*+6. Features use only data up to *t*.
3. **Split by time** (never shuffled), with a 6-hour gap between splits:
   train 2014-04 → 2022, validation 2023, test 2024 → 2026-09.
4. **Models** (`src/models.py`), no deep learning: two baselines, ridge
   regression, random forest and histogram gradient boosting, with and without
   a log-transformed target (32 settings in total). Settings are chosen on the
   validation year; final models are refitted on train + validation and
   scored once on test. The best model is chosen by validation score.
5. **Evaluation** (`src/metrics.py`, `src/evaluate.py`): RMSE and MAE per
   horizon on all hours and on elevated hours (≥ 56 µg/m³, NEA Band II), bias
   by pollution level, per-region audit, permutation feature importance, and
   the largest errors.

## Recommendations and limitations

- Useful for everyday planning (e.g. "will the next few hours stay in the
  normal band?"), where it is clearly better than assuming no change.
- Should not be the only haze warning: it under-predicts high pollution.
  Adding wind, hotspot (fire) counts and regional forecasts, or weighting haze
  hours more during training, are the obvious next steps.
- Only 2026 has substantial haze in the test period, so the elevated-hour
  numbers rest on about 144 hours per horizon.
- Region values are area averages from NEA, not street-level exposure.

## Forecast demo

```bash
python -m src.predict                          # uses the last 48 hours of NEA data
python -m src.predict --input recent.csv       # your own readings
```

The input CSV needs a `timestamp` column and the five region columns, with at
least 24 consecutive hours. If deployed, monitor prediction latency and the
rolling error against readings that arrive later; a rising error or inputs
outside the training range indicate drift and the need to retrain.

## Collaboration

Please fork this repository, create a `dev` branch in your fork, and submit changes through a Pull Request (PR) to this repository's `main` branch. Where possible, keep AI-generated instruction or constraint files, such as `AGENTS.md` and `CLAUDE.md`, local rather than committing them to the repository.

## Data licence

Contains data from the National Environment Agency (NEA), accessed via
data.gov.sg under the [Singapore Open Data Licence](https://data.gov.sg/open-data-licence).
