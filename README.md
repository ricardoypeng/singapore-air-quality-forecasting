# Singapore Air Quality Forecasting

English | [简体中文](README.zh-CN.md)

Predict East Singapore PM2.5 for the next 1–6 hours directly from the previous 24 hours. Active models are HGB (six independent horizon heads) and CatBoost (joint six-output trees), without manual routing or recursive feedback. East inputs contain 186 columns; five-region inputs contain 282.

## Repository contents

The repository contains model and preparation code, required raw observations, dependency specifications and documentation.

- [Entry point](main.py), [training workflow](source/workflow.py), [HGB](source/models/histogram_boosting.py), [CatBoost](source/models/catboost_forecast.py).
- [Raw-data documentation](data/README.md) and [source checksums](data/raw_sources.json).
- [Raw-data preparation entry point](source/prepare.py) and [full experiment runner](source/reproduce.py).
- [Pinned training dependencies](requirements.txt), used with Python 3.10.

## Reproduce from a fresh clone

No data download is needed: the repository includes the original five-region PM CSV and 40 losslessly compressed station-filtered weather observation files. Follow the data documentation for provenance and cleaning rules.

```bash
python3.10 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -B -m source.prepare
python -B main.py --check
python -B main.py
```

Preparation verifies all 41 raw observation checksums, rebuilds hourly weather/PM inputs and chronological splits, and checks the expected 25,668 / 3,331 / 3,818 legal windows. It does not fit models. Do not replace the established whole-day removal policy with timestamp repair when reproducing these experiments.

The runner scans the model package and completes every discovered candidate's validation stage before any test stage. The two current classes declare eight candidates; there is no fixed candidate count in the trainer. Budgets are selected by five-fold chronological training CV with six-hour target purging; the independent validation set chooses the model. Test does not change selection. Use `python -B main.py --dry-run` to inspect discovered classes, candidates and execution stages without training. The older `python -B -m source.reproduce` entry point calls the same runner. Completed experiment reports are not overwritten; use a new `--output-dir` for a separate run.

## Reference behaviour

The established validation-selected configuration is five-region CatBoost depth6, retaining 500 trees. Historical validation/test MSE were 34.450067 / 18.850461. East HGB standard selected 200 rounds and obtained 36.474691 / 18.862571. These values are reference documentation; no saved model or result files are included. Use Python 3.10 with the pinned dependencies to reproduce the experiments. Numeric training equality across platforms has not been verified.

## Collaboration

Fork this repository, create a dev branch and submit PRs to main. Keep AI instruction files local.

## Automatic models, samples and figures

[Model discovery and interface](source/models/README.md) allow a new model class to participate without edits to the common workflow. Candidate settings belong to their model classes; [features](source/features.py), training, evaluation and reporting are shared.

```bash
python -B main.py --list-models
python -B main.py --models hgb --scopes east
python -B main.py --sample-count 10
# Re-export saved results and plots without training:
python -B main.py --stage report --sample-count 10
# Contract tests use a temporary third plugin, not the real estimators:
python -B -m unittest discover -s tests -v
```

A full run trains all selected candidates, freezes validation selection, tests every candidate and then exports comparison/per-horizon metrics, long-form validation/test predictions, and ten shared test examples with their observed history, feature vectors, actual targets and all model predictions. Examples use equally spaced chronological origins, never error-based selection. Figures include overall MSE, per-horizon test MSE, learning curves, CV budget curves and the common examples. Tables and samples are generated in the run directory’s exports subdirectory, plots in figures, and checkpoints under each candidate. Reporting can be rerun independently and does not call fit or predict.

For incremental execution, use `--stage validation --config <discovered-candidate>` for each selected candidate with the same run options, followed by `--stage test`. Testing is blocked until every frozen candidate finishes validation. The `--stage report` option only exports existing results. Existing validation/test results are never overwritten; use a fresh output directory after changes to code, candidate set or run settings.
