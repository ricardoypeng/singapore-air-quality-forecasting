"""Export metrics, common samples and figures from saved predictions, without fitting."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import numpy as np
from source.forecasting import score_forecast


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_report(root: Path, sample_count: int = 10):
    root = Path(root).resolve()
    if sample_count < 1:
        raise ValueError('sample_count must be positive')
    lock = json.loads((root / 'locked_selection.json').read_text())
    names = list(lock['candidates'])
    if not names:
        raise ValueError('No locked candidates to report')
    predictions, validation, summaries, horizon_rows = {}, {}, [], []
    reference = None
    for name in names:
        folder = root / name
        vr = json.loads((folder / 'validation_report.json').read_text())
        tr = json.loads((folder / 'test_report.json').read_text())
        expected = lock['candidates'][name]
        if any(vr[k] != expected[k] for k in ('config', 'selected_budget', 'fingerprint')) or tr['fingerprint'] != expected['fingerprint']:
            raise ValueError(f'{name}: report does not match the validation lock')
        with np.load(folder / 'test_predictions.npz') as saved:
            data = {k: saved[k] for k in saved.files}
        with np.load(folder / 'validation_predictions.npz') as saved:
            validation[name] = {k: saved[k] for k in saved.files}
        if reference is None:
            reference = data
        else:
            for key in ('origins', 'target_times', 'actual', 'current', 'input_history'):
                np.testing.assert_array_equal(data[key], reference[key], err_msg=f'{name}: inconsistent common test {key}')
        vd = validation[name]
        vm = score_forecast(vd['actual'], vd['predicted'], vd['current'])
        np.testing.assert_allclose(vm['mse'], vr['validation']['mse'], rtol=1e-12, atol=1e-12)
        metrics = score_forecast(data['actual'], data['predicted'], data['current'])
        np.testing.assert_allclose(metrics['mse'], tr['test']['mse'], rtol=1e-12, atol=1e-12)
        predictions[name] = data
        summaries.append({'model': name, 'selected_by_validation': name == lock['selected_by_validation'],
            'budget': vr['selected_budget'], 'cv_mse': vr['cv_mean_mse'], 'train_mse': vr['train']['mse'],
            'validation_mse': vr['validation']['mse'], 'test_mse': metrics['mse'],
            'test_mae': metrics['mae'], 'test_mean_horizon_rmse': metrics['mean_horizon_rmse'],
            'direction_accuracy': metrics['net6_direction_accuracy']})
        for split, part in [('validation', validation[name]), ('test', data)]:
            errors = part['predicted'] - part['actual']
            for h in range(errors.shape[1]):
                horizon_rows.append({'model': name, 'split': split, 'horizon': h + 1,
                    'mse': float(np.mean(errors[:, h] ** 2)), 'rmse': float(np.sqrt(np.mean(errors[:, h] ** 2))),
                    'mae': float(np.mean(np.abs(errors[:, h])))})
    if reference is None:
        raise ValueError('No prediction arrays loaded')
    exported = root / 'exports'
    figures = root / 'figures'
    exported.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    write_csv(exported / 'comparison.csv', list(summaries[0]), summaries)
    write_csv(exported / 'per_horizon.csv', list(horizon_rows[0]), horizon_rows)

    def prediction_rows():
        for name in names:
            for split, data in [('validation', validation[name]), ('test', predictions[name])]:
                for i, origin in enumerate(data['origins']):
                    for h in range(data['actual'].shape[1]):
                        actual, pred = float(data['actual'][i, h]), float(data['predicted'][i, h])
                        yield {'model': name, 'split': split, 'origin': str(origin),
                            'target_time': str(data['target_times'][i, h]), 'horizon': h + 1,
                            'actual': actual, 'predicted': pred, 'squared_error': (pred - actual) ** 2}
    write_csv(exported / 'predictions.csv', ['model', 'split', 'origin', 'target_time', 'horizon',
              'actual', 'predicted', 'squared_error'], prediction_rows())
    # Equally spaced chronological indices are shared by all models, never chosen by error.
    indices = np.unique(np.linspace(0, len(reference['origins']) - 1,
                                   min(sample_count, len(reference['origins'])), dtype=int))
    samples = [{'index': int(i), 'origin': str(reference['origins'][i]),
        'current': float(reference['current'][i]), 'input_history': reference['input_history'][i].tolist(),
        'target_times': [str(t) for t in reference['target_times'][i]], 'actual': reference['actual'][i].tolist(),
        'models': {name: {'predicted': predictions[name]['predicted'][i].tolist(),
                          'features': predictions[name]['features'][i].tolist()} for name in names}}
        for i in indices]
    (exported / 'samples.json').write_text(json.dumps(samples, indent=2, allow_nan=False))
    (exported / 'summary.json').write_text(json.dumps({'selected_by_validation': lock['selected_by_validation'],
        'models': summaries, 'sample_policy': 'equally spaced chronological test origins, shared by every model',
        'sample_indices': indices.tolist()}, indent=2, allow_nan=False))

    cache = root / '.mpl-cache'
    own_cache = not cache.exists()
    os.environ['MPLCONFIGDIR'] = str(cache)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        x = np.arange(len(names)); width = 0.25
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 1.2), 5))
        for offset, field, label in [(-1, 'train_mse', 'Train'), (0, 'validation_mse', 'Validation'), (1, 'test_mse', 'Test')]:
            ax.bar(x + offset * width, [r[field] for r in summaries], width, label=label)
        ax.set(xticks=x, xticklabels=names, ylabel='MSE', title='Model comparison; selection uses validation')
        ax.tick_params(axis='x', labelrotation=35); ax.legend(); fig.tight_layout()
        fig.savefig(figures / 'mse_comparison.png', dpi=140); plt.close(fig)

        fig, ax = plt.subplots(figsize=(9, 5))
        for name in names:
            rows = [r for r in horizon_rows if r['model'] == name and r['split'] == 'test']
            ax.plot([r['horizon'] for r in rows], [r['mse'] for r in rows], marker='o', label=name)
        baseline = ((reference['actual'] - reference['current'][:, None]) ** 2).mean(axis=0)
        ax.plot(np.arange(1, len(baseline) + 1), baseline, '--', color='black', label='Persistence')
        ax.set(xlabel='Hours ahead', ylabel='Test MSE', title='Test error by horizon'); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(figures / 'per_horizon.png', dpi=140); plt.close(fig)

        columns = min(2, len(names)); rows = math.ceil(len(names) / columns)
        for kind in ('learning_curves', 'cv_budgets'):
            fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 3 * rows), squeeze=False)
            for ax, name in zip(axes.flat, names):
                folder = root / name
                if kind == 'learning_curves':
                    history = json.loads((folder / 'history.json').read_text())
                    for metric in ('train_mse', 'validation_mse'):
                        ax.plot([r['round'] for r in history], [r[metric] for r in history], marker='.', label=metric)
                else:
                    cv = json.loads((folder / 'cv.json').read_text())['summaries']
                    ax.errorbar([r['budget'] for r in cv], [r['mean_mse'] for r in cv],
                        yerr=[float(np.std(r['fold_mse'])) for r in cv], marker='o', label='CV mean +/- fold std')
                ax.set(title=name, xlabel='Budget', ylabel='MSE'); ax.legend(fontsize=8)
            for ax in list(axes.flat)[len(names):]: ax.set_visible(False)
            fig.tight_layout(); fig.savefig(figures / f'{kind}.png', dpi=140); plt.close(fig)

        columns = min(2, len(samples)); rows = math.ceil(len(samples) / columns)
        fig, axes = plt.subplots(rows, columns, figsize=(7 * columns, 3 * rows), squeeze=False)
        for ax, sample in zip(axes.flat, samples):
            history = np.asarray(sample['input_history'])[:, 0]
            future = np.arange(1, len(sample['actual']) + 1)
            ax.plot(np.arange(-len(history) + 1, 1), history, color='black', label='Observed history')
            ax.plot(np.r_[0, future], [sample['current'], *sample['actual']], color='black', linestyle='--', label='Actual future')
            for name in names:
                ax.plot(np.r_[0, future], [sample['current'], *sample['models'][name]['predicted']], label=name)
            ax.axvline(0, color='gray', linewidth=0.6)
            ax.set(title=sample['origin'], xlabel='Hours relative to origin', ylabel='PM2.5')
        for ax in list(axes.flat)[len(samples):]: ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper center', ncol=min(4, len(labels)), fontsize=8)
        fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(figures / 'common_samples.png', dpi=140); plt.close(fig)
    finally:
        plt.close('all')
        if own_cache:
            shutil.rmtree(cache, ignore_errors=True)
    print(f'Exported {len(names)} models and {len(samples)} common samples to {exported}; figures: {figures}')
    return {'models': len(names), 'samples': len(samples), 'exports': str(exported), 'figures': str(figures)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sample-count', type=int, default=10)
    args = parser.parse_args()
    generate_report(args.output_dir, args.sample_count)


if __name__ == '__main__':
    main()
