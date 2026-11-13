"""End-to-end training pipeline.

Steps:

1. Load and clean the raw data (``src.data``).
2. Build features and targets, split by time (``src.features``).
3. For each model, try every hyper-parameter setting: fit on *train*, score
   on *validation* (mean RMSE over horizons). Keep the best setting.
4. Refit each model's best setting on *train + validation* and score it on
   *test*, which no earlier step has looked at.
5. Pick the overall best model **by validation score** and save it.
6. Bias audit: retrain the best setting for every region and score on test.

Outputs:

* ``results/tuning.csv``: every setting tried and its validation RMSE.
* ``results/test_metrics.csv``: test RMSE / MAE per model and horizon.
* ``results/region_audit.csv``: best model vs persistence for each region.
* ``results/test_predictions.csv``: test predictions of all models.
* ``results/summary.json``: chosen model, settings and headline numbers.
* ``models/best_model.joblib``: the fitted best model and its config.

Run from the project folder::

    python -m src.train            # full pipeline (about 10 minutes on a laptop)
    python -m src.train --quick    # small grids, for a fast check
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src.data import PROJECT_ROOT, REGIONS, load_clean
from src.features import FeatureConfig, SplitDates, make_dataset, time_split
from src.metrics import horizon_metrics, rmse
from src.models import BASELINES, MODEL_NAMES, PARAM_GRIDS, RANDOM_STATE, make_model

RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"

# Reduced grids for --quick runs (used by the tests and for fast checks).
QUICK_GRIDS = {
    "persistence": {},
    "seasonal_naive": {},
    "ridge": {"alpha": [1.0]},
    "random_forest": {"n_estimators": [20], "max_depth": [8], "max_features": [0.3]},
    "hist_gradient_boosting": {"max_iter": [50]},
}


def tune(
    name: str, grid: dict, splits: dict, config: FeatureConfig
) -> tuple[dict, pd.DataFrame]:
    """Try every setting in ``grid`` and return the best one.

    Each setting is fitted on the train split and scored by its mean RMSE over
    all horizons on the validation split.

    Returns:
        ``(best_params, table)`` where ``table`` lists every setting tried.
    """
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    rows = []
    for params in ParameterGrid(grid):
        start = time.perf_counter()
        model = make_model(name, params, config.region, config.horizons)
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        score = np.mean([rmse(y_val.iloc[:, i], pred[:, i]) for i in range(y_val.shape[1])])
        rows.append({
            "model": name,
            "params": json.dumps(params),
            "val_rmse_mean": score,
            "fit_seconds": time.perf_counter() - start,
        })
        print(f"  {name:<24} {json.dumps(params):<90} val RMSE {score:.3f}")

    table = pd.DataFrame(rows)
    best_params = json.loads(table.loc[table["val_rmse_mean"].idxmin(), "params"])
    return best_params, table


def fit_and_test(name: str, params: dict, splits: dict, config: FeatureConfig):
    """Refit on train + validation, then predict the test split.

    Returns:
        ``(model, test_predictions, metrics_table, predict_ms_per_row)``.
    """
    X_fit = pd.concat([splits["train"][0], splits["val"][0]])
    y_fit = pd.concat([splits["train"][1], splits["val"][1]])
    X_test, y_test = splits["test"]

    model = make_model(name, params, config.region, config.horizons)
    model.fit(X_fit, y_fit)

    start = time.perf_counter()
    pred = model.predict(X_test)
    ms_per_row = 1000 * (time.perf_counter() - start) / len(X_test)

    return model, pred, horizon_metrics(y_test, pred, config.horizons), ms_per_row


def region_audit(
    df: pd.DataFrame, name: str, params: dict, base_config: FeatureConfig, dates: SplitDates
) -> pd.DataFrame:
    """Score the chosen model and the persistence baseline on every region."""
    rows = []
    for region in REGIONS:
        config = FeatureConfig(
            region=region,
            window=base_config.window,
            other_region_lags=base_config.other_region_lags,
            horizons=base_config.horizons,
        )
        X, y = make_dataset(df, config)
        splits = time_split(X, y, dates, gap_hours=config.max_horizon)
        for model_name, model_params in [(name, params), ("persistence", {})]:
            _, _, metrics, _ = fit_and_test(model_name, model_params, splits, config)
            rows.append({
                "region": region,
                "model": model_name,
                "RMSE_mean": metrics["RMSE"].mean(),
                "MAE_mean": metrics["MAE"].mean(),
                "RMSE_elevated_mean": metrics["RMSE_elevated"].mean(),
            })
        print(f"  audited {region}")
    return pd.DataFrame(rows)


def run(quick: bool = False, config: FeatureConfig | None = None,
        dates: SplitDates | None = None, results_dir: Path = RESULTS_DIR,
        model_path: Path = MODEL_PATH, audit: bool = True) -> dict:
    """Run the whole pipeline and write all outputs. Returns the summary."""
    np.random.seed(RANDOM_STATE)
    config = config or FeatureConfig()
    dates = dates or SplitDates()
    grids = QUICK_GRIDS if quick else PARAM_GRIDS
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("1/5 Loading data and building features")
    df = load_clean()
    X, y = make_dataset(df, config)
    splits = time_split(X, y, dates, gap_hours=config.max_horizon)
    for split_name, (X_part, _) in splits.items():
        print(f"  {split_name:<5} {len(X_part):>6,} rows  {X_part.index[0]} to {X_part.index[-1]}")
    print(f"  {X.shape[1]} features")

    print("2/5 Tuning on the validation period")
    best_params, tuning_tables = {}, []
    for name in MODEL_NAMES:
        best_params[name], table = tune(name, grids[name], splits, config)
        tuning_tables.append(table)
    tuning = pd.concat(tuning_tables, ignore_index=True)
    tuning.to_csv(results_dir / "tuning.csv", index=False)

    print("3/5 Refitting on train + validation and scoring on test")
    y_test = splits["test"][1]
    test_tables, fitted, latency = [], {}, {}
    predictions = {"origin": y_test.index.repeat(len(config.horizons)),
                   "horizon_hours": np.tile(config.horizons, len(y_test)),
                   "actual": y_test.to_numpy().ravel()}
    for name in MODEL_NAMES:
        model, pred, metrics, ms = fit_and_test(name, best_params[name], splits, config)
        fitted[name], latency[name] = model, ms
        predictions[name] = pred.ravel()
        test_tables.append(metrics.assign(model=name).reset_index())
        print(f"  {name:<24} test RMSE (mean over horizons) {metrics['RMSE'].mean():.3f}")
    test_metrics = pd.concat(test_tables, ignore_index=True)
    test_metrics.to_csv(results_dir / "test_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(results_dir / "test_predictions.csv", index=False)

    print("4/5 Choosing the best model by validation score")
    best_val = tuning.groupby("model")["val_rmse_mean"].min()
    candidates = best_val.drop(BASELINES)
    best_name = candidates.idxmin()
    print(f"  best model: {best_name} {best_params[best_name]}")
    joblib.dump(
        {"model": fitted[best_name], "name": best_name, "params": best_params[best_name],
         "config": config, "feature_columns": list(X.columns)},
        model_path,
    )

    audit_table = None
    if audit:
        print("5/5 Bias audit across regions")
        audit_table = region_audit(df, best_name, best_params[best_name], config, dates)
        audit_table.to_csv(results_dir / "region_audit.csv", index=False)

    test_mean = test_metrics.groupby("model")[["RMSE", "MAE"]].mean()
    summary = {
        "best_model": best_name,
        "best_params": best_params[best_name],
        "region": config.region,
        "window_hours": config.window,
        "horizons": list(config.horizons),
        "split": {"train_end": dates.train_end, "val_end": dates.val_end},
        "validation_rmse_mean": best_val.round(4).to_dict(),
        "test_rmse_mean": test_mean["RMSE"].round(4).to_dict(),
        "test_mae_mean": test_mean["MAE"].round(4).to_dict(),
        "predict_ms_per_row": {k: round(v, 4) for k, v in latency.items()},
        "quick_run": quick,
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Done. Outputs in {results_dir} and {model_path}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small grids for a fast check")
    parser.add_argument("--region", default="east", choices=REGIONS)
    parser.add_argument("--no-audit", action="store_true", help="skip the per-region audit")
    args = parser.parse_args()
    run(quick=args.quick, config=FeatureConfig(region=args.region), audit=not args.no_audit)


if __name__ == "__main__":
    main()
