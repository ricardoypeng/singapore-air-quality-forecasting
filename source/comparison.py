"""Train and evaluate discovered forecast models on aligned input profiles."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .experiment_features import CompactEncoder, compact_values
from .experiment_schema import BASE_COLUMNS, RECENT_COLUMNS
from .forecasting import score_forecast
from .workflow import discover_configs, forecast, new_model, save_json, source_identity


CHECKPOINT_SCHEMA = 2


def comparison_configs(seed):
    """Combine the original scopes and the additional model-declared profiles."""
    configs = discover_configs(seed, include_profiles=True)
    for config in configs.values():
        if "scope" in config:
            config.update(
                input_profile=config["scope"],
                expanded_training=False,
                shuffle=True,
            )
    return configs


@dataclass
class Cohort:
    base: np.ndarray
    recent: np.ndarray
    origins: np.ndarray
    actual: np.ndarray
    current: np.ndarray
    original_rows: np.ndarray

    def take(self, rows):
        return Cohort(
            self.base[rows],
            self.recent[rows],
            self.origins[rows],
            self.actual[rows],
            self.current[rows],
            self.original_rows[rows],
        )

    def design(self, profile, encoder=None):
        if profile == "east":
            return self.base[:, :186]
        if profile == "five":
            return self.base
        if profile == "recent":
            return np.column_stack((self.base, self.recent))
        if profile in ("compact", "recent_compact"):
            compact = encoder.transform(*compact_values(self.base, self.recent))
            if profile == "compact":
                return compact
            return np.column_stack((self.base, self.recent, compact))
        raise ValueError(f"Unknown input profile: {profile}")


def load_cohort(path, split):
    expected = {"train": 59_601, "validation": 3_331, "test": 3_818}[split]
    with np.load(path, allow_pickle=False) as data:
        for name, columns in (("base_names", BASE_COLUMNS), ("recent_names", RECENT_COLUMNS)):
            if name not in data or not np.array_equal(data[name], columns):
                raise ValueError(f"Matrix columns are missing or out of order: {name}")
        cohort = Cohort(
            data[f"x_{split}"],
            data[f"recent_{split}"],
            data[f"origins_{split}"],
            data[f"y_{split}"],
            data[f"current_{split}"],
            data["original_train"] if split == "train" else np.ones(expected, dtype=bool),
        )
    shapes = (
        (expected, len(BASE_COLUMNS)), (expected, len(RECENT_COLUMNS)),
        (expected,), (expected, 6), (expected,),
    )
    arrays = (cohort.base, cohort.recent, cohort.origins, cohort.actual, cohort.current)
    if tuple(array.shape for array in arrays) != shapes:
        raise ValueError(f"Unexpected matrix shape in {split}")
    if (cohort.original_rows.shape != (expected,) or cohort.original_rows.dtype != np.bool_
            or (split == "train" and cohort.original_rows.sum() != 25_668)):
        raise ValueError("Invalid mask for original training origins")
    if (np.isnat(cohort.origins).any() or np.any(np.diff(cohort.origins).astype(int) <= 0)
            or not np.isfinite(cohort.actual).all() or not np.isfinite(cohort.current).all()):
        raise ValueError(f"Invalid label or origin in {split}")
    np.testing.assert_array_equal(cohort.current, cohort.base[:, 138])
    return cohort


def sample_test(cohort, seed, count):
    """Choose unique chronological test origins with a seed isolated from training."""
    if type(count) is not int or not 1 <= count <= len(cohort.origins):
        raise ValueError("Sample count must be between 1 and the test cohort size")
    indices = np.random.default_rng(seed).choice(len(cohort.origins), count, replace=False)
    return cohort.take(np.sort(indices))


def file_sha256(path):
    with Path(path).open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_designs(configs, train, valid):
    """Build each train/validation representation and fit compact imputation on train."""
    if train.origins[-1] + np.timedelta64(6, "h") >= valid.origins[0] + np.timedelta64(1, "h"):
        raise ValueError("Training targets overlap validation targets")
    designs = {}
    trainings = {False: train.take(train.original_rows), True: train}
    for config in configs.values():
        profile = config["input_profile"]
        expanded = config["expanded_training"]
        key = (profile, expanded)
        if key not in designs:
            cohort = trainings[expanded]
            encoder = None
            if profile in ("compact", "recent_compact"):
                encoder = CompactEncoder.fit(*compact_values(cohort.base, cohort.recent))
            designs[key] = (
                cohort.design(profile, encoder),
                cohort.actual,
                valid.design(profile, encoder),
                encoder,
            )
    return designs


def train_candidate(config, designs, valid, threads):
    """Fit via ForecastModel, select the prediction budget on validation and snapshot."""
    key = (config["input_profile"], config["expanded_training"])
    x, y, xv, encoder = designs[key]
    model = new_model(config, threads)
    with threadpool_limits(limits=threads):
        if config["shuffle"]:
            order = np.random.default_rng(config["seed"]).permutation(len(y))
            x, y = x[order], y[order]
        search_capacity = config["training_cap"]
        model.fit(x, y, search_capacity)
        metrics = {
            budget: score_forecast(valid.actual, forecast(model, xv, budget), valid.current)
            for budget in config["budgets"]
        }
        budget = min(metrics, key=lambda value: (metrics[value]["mse"], value))
        capacity = model.training_capacity(budget, search_capacity)
        if capacity != search_capacity:
            model = new_model(config, threads).fit(x, y, capacity)
        snapshot = model.snapshot(budget)
        validation = score_forecast(valid.actual, forecast(snapshot, xv), valid.current)
    return {
        "model": snapshot,
        "config": config,
        "budget": budget,
        "encoder": encoder,
        "training_origins": len(y),
        "validation": validation,
    }


def implementation_hashes(configs):
    """Fingerprint the candidate classes and their shared feature and model code."""
    hashes = {}
    by_algorithm = {config["algorithm"]: config for config in configs.values()}
    for config in by_algorithm.values():
        hashes.update(source_identity(config))

    source = Path(__file__).resolve().parent
    modules = [
        Path(__file__).resolve(),
        source / "experiment_features.py",
        source / "experiment_schema.py",
        *(source / "models").glob("*.py"),
    ]
    hashes.update({str(path.relative_to(source.parent)): file_sha256(path) for path in modules})
    return hashes


def train_comparison(matrix, run_dir, seed=42, threads=4):
    """Write independent checkpoints and publish the manifest when all models are ready."""
    if threads < 1:
        raise ValueError("Use at least one CPU thread")
    run_dir = Path(run_dir)
    if run_dir.exists():
        raise FileExistsError(f"Use a fresh RUN_DIR for training: {run_dir}")
    digest = file_sha256(matrix)
    configs = comparison_configs(seed)
    implementation = implementation_hashes(configs)
    train = load_cohort(matrix, "train")
    valid = load_cohort(matrix, "validation")
    designs = prepare_designs(configs, train, valid)
    run_dir.mkdir(parents=True, exist_ok=False)
    entries = {}
    for name, config in configs.items():
        payload = train_candidate(config, designs, valid, threads)
        filename = f"{name}.joblib"
        joblib.dump(payload, run_dir / filename, compress=3)
        entries[name] = {
            "checkpoint": filename,
            "sha256": file_sha256(run_dir / filename),
            "budget": payload["budget"],
            "training_origins": payload["training_origins"],
            "validation_mse": payload["validation"]["mse"],
        }
        print(f"{name:40s} {entries[name]['validation_mse']:.4f}", flush=True)
    if file_sha256(matrix) != digest or implementation_hashes(configs) != implementation:
        raise ValueError("Comparison inputs or code changed during training")
    save_json(run_dir / "manifest.json", {
        "schema_version": CHECKPOINT_SCHEMA,
        "implementation_sha256": implementation,
        "matrix_sha256": digest,
        "seed": seed,
        "threads": threads,
        "candidates": entries,
    })
    return entries


def predict_comparison(matrix, run_dir, seed=42, count=512):
    """Load trusted local checkpoints and predict the same seeded test sample."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != CHECKPOINT_SCHEMA:
        raise ValueError("Checkpoint schema changed. Retrain in a fresh RUN_DIR")
    if manifest["matrix_sha256"] != file_sha256(matrix):
        raise ValueError("Checkpoints require the matrix archive used during training")
    configs = comparison_configs(manifest["seed"])
    if (manifest["implementation_sha256"] != implementation_hashes(configs)
            or manifest["candidates"].keys() != configs.keys()):
        raise ValueError("Model or feature code changed. Retrain in a fresh RUN_DIR")
    sample = sample_test(load_cohort(matrix, "test"), seed, count)
    predictions = {}
    metrics = {}
    with threadpool_limits(limits=manifest["threads"]):
        for name, entry in manifest["candidates"].items():
            path = run_dir / entry["checkpoint"]
            if file_sha256(path) != entry["sha256"]:
                raise ValueError(f"Checkpoint changed: {name}")
            payload = joblib.load(path)
            x = sample.design(payload["config"]["input_profile"], payload["encoder"])
            predictions[name] = forecast(payload["model"], x, payload["budget"])
            metrics[name] = score_forecast(sample.actual, predictions[name], sample.current)
    return sample, predictions, metrics
