# Model Plugin Contract

The shared [workflow](../workflow.py) scans this package for concrete subclasses of [ForecastModel](base.py). It never branches on an algorithm name. Add a Python module containing a class defined in that module, give it a unique `name`, and implement this contract; the next process discovers it automatically. Imported/re-exported classes and abstract helper classes are skipped. Import errors, duplicate names and invalid budget declarations are reported instead of silently dropping a model.

## Required interface

| Member | Contract |
| --- | --- |
| `name` | Unique lowercase identifier, such as `my_regressor` |
| `__init__(seed=42, threads=4, **parameters)` | Store settings and initialize unfitted state; no training or file writes |
| `variants()` | Class method returning variant names and their `parameters`, increasing positive `budgets`, and `training_cap` |
| `fit(X, y, budget)` | Fit using training arrays only; return self. X is `(N,P)`, y is `(N,6)` |
| `predict(X, budget=None)` | Return finite `(N,6)` predictions. An explicit budget selects a prefix of the fitted trajectory |
| `snapshot(budget)` | Return independent, joblib-serializable state that predicts at that budget; never modify the original model |
| `model_budget()` | Return JSON-compatible actual capacity and active prediction budget metadata |
| `training_capacity(selected_budget, search_capacity)` | Optional override; determines final fit capacity without changing the shared trainer |

The base class implements an independent deep-copy snapshot and final capacity equal to the selected budget. A model that requires a different snapshot or full-capacity final fit overrides these methods. Models without meaningful prefixes can declare a single unit budget of 1 and keep their fixed estimator settings in the variant parameters; the shared runner then requests only that one snapshot.

Each variant is automatically combined with the requested input scopes. For example, a class named `my_regressor` with variant `default` yields `my_regressor_east_default` and `my_regressor_five_default`. Candidate parameters and budget rules live in the model class, not in the workflow.

The shared feature builder supplies either 186 or 282 columns. Seed, threads, folds, checkpoint spacing and selected model/scopes are controlled by the CLI. All candidate recipes, implementation hashes and prepared-data hashes are frozen before training; a changed model set or code requires a new output directory.

## Existing implementations

- [HGB](histogram_boosting.py): public `staged_predict` selects earlier budgets. Snapshots retain independent full fitted state and select the requested prediction prefix; no private sklearn fields are truncated. Final capacity equals the CV-selected budget.
- [CatBoost](catboost_forecast.py): fit the declared full capacity once, score prefixes and export independent physically shrunk snapshots. Final fit capacity remains the search capacity.

Training and result export are independent. The [reporter](../reporting.py) reads locked reports and saved prediction arrays, verifies common sample alignment, and writes tables, samples and figures without constructing or fitting models.

The [contract tests](../../tests/test_pipeline.py) dynamically add a third temporary plugin and exercise the full validation/test/report path without editing a registry or the trainer.
