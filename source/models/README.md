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

Each ordinary variant is combined with the requested input scopes. For example,
a class named `my_regressor` with variant `default` yields
`my_regressor_east_default` and `my_regressor_five_default`. The command-line
workflow supplies 186 or 282 input columns and freezes its candidate recipes,
implementation hashes and prepared-data hashes before training.

A variant can declare an `input_profile` to join the notebook workflow with
a model-specific feature representation. The
[common comparison code](../comparison.py) builds the design and calls the
same model constructor, `fit`, `training_capacity`, `snapshot`, and `predict` methods.
The notebook run manifest also fingerprints the feature and model implementation.
The matrix has aligned East, five-region, recent-weather and compact inputs.
A recipe can set `expanded_training` and `shuffle` to control its training
cohort and order.

## Existing implementations

- [HGB](histogram_boosting.py): public `staged_predict` selects earlier budgets. Snapshots retain independent full fitted state and select the requested prediction prefix; no private sklearn fields are truncated. Final capacity equals the CV-selected budget.
- [CatBoost](catboost_forecast.py): fit the declared full capacity once, score prefixes and export independent physically shrunk snapshots. Final fit capacity remains the search capacity.

## Additional model implementations

- [CatBoost variants](catboost_variants.py) for weather inputs, tree budgets,
  lead-aware prediction and the adjacent-change objective.
- [RBF kernel Ridge](kernel_models.py).
- [Historical analog regression](analog_models.py).
- [Probabilistic regime experts](mixture_models.py).
- [Fixed CatBoost and analog blend](blended_forecast.py).

Training Notebook [English](../../model_training.ipynb) ·
[简体中文](../../model_training.zh-CN.ipynb)

Test inference Notebook [English](../../model_test_comparison.ipynb) ·
[简体中文](../../model_test_comparison.zh-CN.ipynb)

Training and result export are independent. The [reporter](../reporting.py) reads locked reports and saved prediction arrays, verifies common sample alignment, and writes tables, samples and figures without constructing or fitting models.

The [contract tests](../../tests/test_pipeline.py) dynamically add a third temporary plugin and exercise the full validation/test/report path without editing a registry or the trainer.
