"""Model-agnostic discovery, chronological selection, checkpoints and test evaluation."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any

sys.dont_write_bytecode = True
import joblib
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from threadpoolctl import threadpool_limits
from source.models import discover_models
from source.models.base import ForecastModel
from source.features import input_schema, design, spatial_histories
from source.forecasting import score_forecast

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def discover_configs(seed=SEED, scopes=('east', 'five')):
    configs = {}
    for name, cls in discover_models().items():
        for scope in scopes:
            for variant, recipe in cls.variants().items():
                key = f'{name}_{scope}_{variant}'
                configs[key] = {'algorithm': name, 'scope': scope, 'complexity': variant,
                    'budgets': list(recipe['budgets']), 'training_cap': recipe['training_cap'],
                    'model_parameters': dict(recipe['parameters']), 'input_schema': input_schema(scope),
                    'target_mode': 'six_absolute_levels', 'forecast_strategy': 'direct', 'seed': seed}
    return configs


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def new_model(config, threads):
    cls = discover_models()[config['algorithm']]
    return cls(seed=config['seed'], threads=threads, **config['model_parameters'])


def forecast(model, x, budget=None):
    predicted = np.asarray(model.predict(x, budget=budget))
    if predicted.ndim != 2 or len(predicted) != len(x) or not np.isfinite(predicted).all():
        raise ValueError('Expected a finite (samples, horizons) prediction matrix')
    return predicted


def source_identity(config):
    cls = discover_models()[config['algorithm']]
    paths = [Path(inspect.getfile(cls)), *[ROOT / name for name in
        ('source/models/base.py', 'source/models/__init__.py', 'source/workflow.py',
         'source/features.py', 'source/forecasting.py', 'source/data.py', 'source/east_weather_data.py')]]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:
            hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def fingerprint(config, hashes, selected_budget):
    value = {'config': config, 'source_sha256': hashes, 'selected_budget': selected_budget,
             'implementation': source_identity(config)}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def freeze_run(args, configs, hashes):
    manifest = {'schema_version': 3, 'configs': configs, 'source_sha256': hashes,
                'implementation': {name: source_identity(c) for name, c in configs.items()},
                'folds': args.folds, 'checkpoint_every': args.checkpoint_every, 'threads': args.threads}
    path = args.output_dir / 'run_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError('Run inputs, discovered candidates, code or settings changed; use a new output directory')
    else:
        save_json(path, manifest)


def lock_if_complete(root, configs=None):
    configs = discover_configs() if configs is None else configs
    files = [root / name / 'validation_report.json' for name in configs]
    if not all(p.exists() for p in files):
        return
    reports = [json.loads(p.read_text()) for p in files]
    hashes = reports[0]['source_sha256']
    for name, report in zip(configs, reports):
        if report['config'] != configs[name] or report['source_sha256'] != hashes:
            raise ValueError('Validation reports do not match the frozen candidate set')
        if report['fingerprint'] != fingerprint(configs[name], hashes, report['selected_budget']):
            raise ValueError('Validation implementation fingerprint changed')
    ranking = sorted(reports, key=lambda r: (r['validation']['mse'], r['name']))
    cv = sorted(reports, key=lambda r: (r['cv_mean_mse'], r['name']))
    save_json(root / 'locked_selection.json', {'schema_version': 3,
        'candidates': {r['name']: {k: r[k] for k in ('config', 'selected_budget', 'fingerprint')} for r in reports},
        'source_sha256': hashes, 'selected_by_validation': ranking[0]['name'],
        'selected_by_cv': cv[0]['name'],
        'validation_ranking': [{'name': r['name'], 'mse': r['validation']['mse']} for r in ranking],
        'selection_note': 'All discovered candidates finish validation before test. Test cannot change selection.'})


def save_predictions(path, part, x, predicted):
    if predicted.shape != part.target.shape:
        raise ValueError('Forecast horizon count differs from target horizon count')
    np.savez_compressed(path, origins=part.origins, target_times=part.target_times,
        actual=part.target, predicted=predicted, current=part.history[:, -1, 0],
        input_history=part.history, features=x)


def validation_stage(args, parts, spatial, input_audit, config, hashes, score=score_forecast, configs=None):
    folder = args.output_dir / args.config
    if (folder / 'validation_report.json').exists() or (args.output_dir / 'locked_selection.json').exists():
        raise FileExistsError('Completed or locked validation cannot be overwritten')
    train, valid = parts['train'], parts['validation']
    if not len(train.target) or not len(valid.target):
        raise ValueError('Training and validation windows must be nonempty')
    x, xv = design(train, config, spatial['train']), design(valid, config, spatial['validation'])
    seed = config['seed']
    budgets, folds = config['budgets'], []
    for index, (_, held) in enumerate(TimeSeriesSplit(n_splits=args.folds).split(train.origins), 1):
        cutoff = train.origins[held[0]]
        fit = np.flatnonzero(train.target_times[:, -1] < cutoff)
        if not len(fit) or np.intersect1d(fit, held).size:
            raise ValueError('Training labels must strictly precede held-out origins')
        order = np.random.default_rng(seed).permutation(fit)
        model = new_model(config, args.threads)
        model.fit(x[order], train.target[order], config['training_cap'])
        scores = []
        for budget in budgets:
            metrics = score(train.target[held], forecast(model, x[held], budget), train.history[held, -1, 0])
            scores.append({'budget': budget, 'score': metrics})
            print(f"{args.config} CV {index}/{args.folds} budget={budget} MSE={metrics['mse']:.6f}", flush=True)
        folds.append({'fold': index, 'train_windows': len(order), 'validation_windows': len(held),
            'training_last_label': str(train.target_times[fit[-1], -1]),
            'validation_first_origin': str(cutoff), 'labels_strictly_precede_origin': True,
            'budget_scores': scores, 'model_budget_at_training_cap': model.model_budget()})
        save_json(folder / 'cv_progress.json', folds)
    summaries = [{'budget': budget, 'fold_mse': [next(s['score']['mse'] for s in f['budget_scores']
                  if s['budget'] == budget) for f in folds]} for budget in budgets]
    for row in summaries:
        row['mean_mse'] = float(np.mean(row['fold_mse']))
    chosen = min(summaries, key=lambda r: (r['mean_mse'], r['budget']))
    selected = chosen['budget']
    save_json(folder / 'cv.json', {'folds': folds, 'summaries': summaries, 'selected_budget': selected})
    model = new_model(config, args.threads)
    capacity = model.training_capacity(selected, config['training_cap'])
    order = np.random.default_rng(seed).permutation(len(x))
    model.fit(x[order], train.target[order], capacity)
    checkpoints = sorted(set(range(args.checkpoint_every, selected + 1, args.checkpoint_every)) | {selected})
    history = []
    snapshot: ForecastModel | None = None
    ts: dict[str, Any] | None = None
    vs: dict[str, Any] | None = None
    pv: np.ndarray | None = None
    for budget in checkpoints:
        snapshot = model.snapshot(budget)
        pt, pv = forecast(snapshot, x), forecast(snapshot, xv)
        ts, vs = score(train.target, pt, train.history[:, -1, 0]), score(valid.target, pv, valid.history[:, -1, 0])
        history.append({'round': budget, 'train_mse': ts['mse'], 'validation_mse': vs['mse']})
        path = folder / 'checkpoints' / f'checkpoint_{budget:03d}.joblib'
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({'model': snapshot, 'config': config, 'budget': budget, 'selected_budget': selected,
            'source_sha256': hashes, 'fingerprint': fingerprint(config, hashes, selected)}, path)
        restored = joblib.load(path)
        np.testing.assert_allclose(forecast(restored['model'], xv[:8]), pv[:8], rtol=1e-11, atol=1e-11)
        save_json(folder / 'history.json', history)
    if snapshot is None or ts is None or vs is None or pv is None:
        raise RuntimeError('No final checkpoint was produced')
    save_predictions(folder / 'validation_predictions.npz', valid, xv, pv)
    save_json(folder / 'validation_report.json', {'name': args.config, 'config': config,
        'fingerprint': fingerprint(config, hashes, selected), 'source_sha256': hashes,
        'selected_budget': selected, 'training_cap': capacity, 'cv_mean_mse': chosen['mean_mse'],
        'train': ts, 'validation': vs, 'history': history, 'model_budget': snapshot.model_budget(),
        'input_audit': input_audit, 'training_windows': len(x), 'validation_windows': len(xv),
        'input_columns': x.shape[1], 'test_used': False})
    lock_if_complete(args.output_dir, configs)


def test_stage(args, parts, spatial, config, hashes, score=score_forecast, configs=None):
    configs = discover_configs() if configs is None else configs
    lock = json.loads((args.output_dir / 'locked_selection.json').read_text())
    if set(lock['candidates']) != set(configs) or lock['source_sha256'] != hashes:
        raise ValueError('All frozen candidates must complete validation before test')
    expected = lock['candidates'][args.config]
    selected = expected['selected_budget']
    if expected['config'] != config or expected['fingerprint'] != fingerprint(config, hashes, selected):
        raise ValueError('Locked model recipe or implementation changed')
    folder = args.output_dir / args.config
    if (folder / 'test_report.json').exists():
        raise FileExistsError('Test already evaluated; use saved predictions for analysis')
    payload = joblib.load(folder / 'checkpoints' / f'checkpoint_{selected:03d}.joblib')
    if payload['budget'] != selected or any(payload[k] != expected[k] for k in ('config', 'selected_budget', 'fingerprint')) or payload['source_sha256'] != hashes:
        raise ValueError('Checkpoint differs from the locked model')
    part = parts['test']
    x = design(part, config, spatial['test'])
    predicted = forecast(payload['model'], x)
    save_predictions(folder / 'test_predictions.npz', part, x, predicted)
    metrics = score(part.target, predicted, part.history[:, -1, 0])
    save_json(folder / 'test_report.json', {'name': args.config, 'config': config, 'test': metrics,
        'selected_budget': selected, 'test_windows': len(x), 'fingerprint': expected['fingerprint'],
        'selected_by_validation': lock['selected_by_validation'], 'selection_frozen_before_test': True})
    print(f"{args.config} TEST MSE={metrics['mse']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-dir', type=Path, default=ROOT)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'tmp/_project/latest-model-runs')
    parser.add_argument('--models', nargs='+', help='Discovered model names; default: all')
    parser.add_argument('--scopes', nargs='+', choices=('east', 'five'), default=['east', 'five'])
    parser.add_argument('--config', help='Run one candidate in a validation/test stage')
    parser.add_argument('--stage', choices=('all', 'validation', 'test', 'report'), default='all')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--list-models', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--sample-count', type=int, default=10)
    args = parser.parse_args()
    if min(args.threads, args.checkpoint_every, args.sample_count) < 1 or args.folds < 2:
        parser.error('threads, checkpoint interval and samples must be positive; folds must be >= 2')
    if args.stage == 'report':
        from source.reporting import generate_report
        generate_report(args.output_dir.resolve(), args.sample_count)
        return
    models = discover_models()
    if args.models and set(args.models) - set(models):
        parser.error(f'Unknown models: {set(args.models) - set(models)}')
    configs = {k: c for k, c in discover_configs(args.seed, tuple(dict.fromkeys(args.scopes))).items()
               if not args.models or c['algorithm'] in args.models}
    if args.config and args.config not in configs:
        parser.error(f'Unknown candidate: {args.config}')
    if args.config and args.stage == 'all':
        parser.error('--config is for individual validation/test stages; use --models for a complete run')
    if args.list_models or args.dry_run:
        print(json.dumps({'models': {n: f'{c.__module__}.{c.__name__}' for n, c in models.items()},
            'candidates': configs, 'stages': ['validation_all', 'lock', 'test_all', 'export_and_plot']}, indent=2))
        return
    from source.east_weather_data import load_windows
    args.project_dir, args.output_dir = args.project_dir.resolve(), args.output_dir.resolve()
    data = args.project_dir / 'tmp/_project/prepared-data/weather_2024'
    required = [data / f'{split}{suffix}' for split in ('train', 'validation', 'test')
                for suffix in ('.pkl', '_windows_24h.npz')]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in required}
    parts = load_windows(data, 6)
    spatial, audit = spatial_histories(data, parts)
    if args.check:
        shapes = {name: {split: list(design(p, c, spatial[split]).shape) for split, p in parts.items()}
                  for name, c in configs.items()}
        for c in configs.values():
            if getattr(new_model(c, args.threads), 'budget', 0):
                raise ValueError('Constructors must not fit models')
        save_json(args.output_dir / 'structure_checks.json', {'status': 'passed', 'fit_called': False,
            'design_shapes': shapes, 'source_sha256': hashes, 'input_audit': audit})
        print('Discovered model interfaces and input designs passed; no models fitted.')
        return
    freeze_run(args, configs, hashes)
    with threadpool_limits(limits=args.threads):
        if args.stage in ('all', 'validation'):
            requested = args.config
            for name, c in configs.items():
                if requested and name != requested:
                    continue
                args.config = name
                validation_stage(args, parts, spatial, audit, c, hashes, configs=configs)
            args.config = requested
        if args.stage in ('all', 'test'):
            requested = args.config
            for name, c in configs.items():
                if requested and name != requested:
                    continue
                args.config = name
                test_stage(args, parts, spatial, c, hashes, configs=configs)
            args.config = requested
    if args.stage == 'all':
        from source.reporting import generate_report
        generate_report(args.output_dir, args.sample_count)


if __name__ == '__main__':
    main()
