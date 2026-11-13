"""Contract tests with a discovered third model; never fit the real estimators."""
import copy
import importlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
import source.models as package
from source.models import discover_models
from source.workflow import (ROOT, discover_configs, freeze_run, validation_stage, test_stage)
from source.reporting import generate_report

PLUGIN = '''from source.models.base import ForecastModel
import numpy as np
class Probe(ForecastModel):
    name = 'probe'
    @classmethod
    def variants(cls):
        return {'default': {'parameters': {}, 'budgets': [1, 3], 'training_cap': 3}}
    def __init__(self, seed=42, threads=4):
        self.budget = 0
    def fit(self, x, y, budget):
        self.budget = budget
        self.horizons = y.shape[1]
        return self
    def predict(self, x, budget=None):
        end = self.budget if budget is None else budget
        return x[:, :1] + np.arange(1, self.horizons + 1)[None, :] + end / 10
    def model_budget(self):
        return {'budget': self.budget}
'''


def make_part(start, n):
    origins = np.datetime64('2020-01-01T00', 'h') + np.arange(start, start + n).astype('timedelta64[h]')
    history = np.repeat((np.arange(start, start + n)[:, None] + np.arange(-23, 1)[None, :])[:, :, None], 6, axis=2).astype(float)
    return SimpleNamespace(origins=origins, history=history,
        target=history[:, -1, :1] + np.arange(1, 7)[None, :],
        target_times=origins[:, None] + np.arange(1, 7).astype('timedelta64[h]'))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        task = ROOT / 'tmp/_project/generic-model-pipeline/unit-tests'
        task.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=task)
        self.root = Path(self.tmp.name)
        plugins = self.root / 'plugins'; plugins.mkdir()
        (plugins / 'probe.py').write_text(PLUGIN)
        self.plugins = plugins
        package.__path__.append(str(plugins)); importlib.invalidate_caches()
        self.parts = {s: make_part(start, n) for s, start, n in [('train', 0, 90), ('validation', 100, 20), ('test', 130, 20)]}
        self.spatial = {s: np.zeros((len(p.origins), 24, 4)) for s, p in self.parts.items()}
        self.configs = {k: c for k, c in discover_configs().items() if c['algorithm'] == 'probe'}
        self.args = SimpleNamespace(output_dir=self.root / 'run', config=None,
            folds=2, threads=1, checkpoint_every=2)
        self.hashes = {'synthetic': 'fixed'}
        freeze_run(self.args, self.configs, self.hashes)

    def tearDown(self):
        package.__path__.remove(str(self.plugins))
        for name in ['probe', 'duplicate', 'bad']:
            sys.modules.pop('source.models.' + name, None)
        importlib.invalidate_caches(); self.tmp.cleanup()

    def train_all(self):
        # Validation is intentionally given no test data; touching test would fail.
        parts = {s: self.parts[s] for s in ('train', 'validation')}
        for name, cfg in self.configs.items():
            self.args.config = name
            validation_stage(self.args, parts, self.spatial, {}, cfg, self.hashes, configs=self.configs)

    def evaluate_all(self):
        for name, cfg in self.configs.items():
            self.args.config = name
            test_stage(self.args, self.parts, self.spatial, cfg, self.hashes, configs=self.configs)

    def test_new_model_is_discovered_and_completes_exports_and_figures(self):
        self.assertIn('probe', discover_models())
        self.train_all(); self.evaluate_all()
        # Reporting must work without discovering models or calling fit/predict.
        from unittest.mock import patch
        with patch('source.workflow.new_model', side_effect=AssertionError('No model access in reporting')):
            result = generate_report(self.args.output_dir, 10)
        self.assertEqual(result['samples'], 10)
        self.assertEqual(result['models'], 2)
        exports = self.args.output_dir / 'exports'
        samples = json.loads((exports / 'samples.json').read_text())
        self.assertEqual([s['index'] for s in samples], np.linspace(0, 19, 10, dtype=int).tolist())
        for sample in samples:
            self.assertEqual(set(sample['models']), set(self.configs))
        for name in ['comparison.csv', 'per_horizon.csv', 'predictions.csv', 'summary.json']:
            self.assertTrue((exports / name).is_file())
        for name in ['mse_comparison.png', 'per_horizon.png', 'learning_curves.png', 'cv_budgets.png', 'common_samples.png']:
            self.assertGreater((self.args.output_dir / 'figures' / name).stat().st_size, 1000)

    def test_cli_discovers_and_runs_the_complete_third_model_pipeline(self):
        from unittest.mock import patch
        from source.workflow import main
        output = self.root / 'cli-run'
        project = self.root / 'project'
        prepared = project / 'tmp/_project/prepared-data/weather_2024'
        prepared.mkdir(parents=True)
        # CLI hashes these inputs; the patched loader supplies synthetic windows.
        # Keep the test independent of any locally prepared real observations.
        for split in self.parts:
            for suffix in ('.pkl', '_windows_24h.npz'):
                (prepared / f'{split}{suffix}').write_bytes(f'{split}{suffix}'.encode())
        argv = ['main.py', '--models', 'probe', '--project-dir', str(project), '--output-dir', str(output),
                '--folds', '2', '--threads', '1', '--checkpoint-every', '2']
        with patch.object(sys, 'argv', argv), \
             patch('source.east_weather_data.load_windows', return_value=self.parts), \
             patch('source.workflow.spatial_histories', return_value=(self.spatial, {})):
            main()
        self.assertTrue((output / 'locked_selection.json').is_file())
        self.assertTrue((output / 'exports/samples.json').is_file())
        self.assertTrue((output / 'figures/common_samples.png').is_file())

    def test_test_is_blocked_until_all_candidates_finish_validation(self):
        name = next(iter(self.configs)); self.args.config = name
        validation_stage(self.args, self.parts, self.spatial, {}, self.configs[name], self.hashes, configs=self.configs)
        self.assertFalse((self.args.output_dir / 'locked_selection.json').exists())
        with self.assertRaises(FileNotFoundError):
            test_stage(self.args, self.parts, self.spatial, self.configs[name], self.hashes, configs=self.configs)

    def test_purged_cv_and_non_multiple_checkpoint_budget(self):
        self.train_all()
        for name in self.configs:
            folder = self.args.output_dir / name
            cv = json.loads((folder / 'cv.json').read_text())
            for fold in cv['folds']:
                self.assertLess(np.datetime64(fold['training_last_label']), np.datetime64(fold['validation_first_origin']))
            budget = cv['selected_budget']
            self.assertTrue((folder / 'checkpoints' / f'checkpoint_{budget:03d}.joblib').exists())

    def test_frozen_candidate_recipe_cannot_change(self):
        changed = copy.deepcopy(self.configs)
        changed[next(iter(changed))]['seed'] = 123
        with self.assertRaises(ValueError):
            freeze_run(self.args, changed, self.hashes)
        with self.assertRaises(ValueError):
            freeze_run(self.args, self.configs, {'synthetic': 'changed-window-or-data'})

    def test_validation_and_test_cannot_be_overwritten(self):
        self.train_all(); self.evaluate_all()
        name = next(iter(self.configs));self.args.config = name
        with self.assertRaises(FileExistsError):
            validation_stage(self.args, self.parts, self.spatial, {}, self.configs[name], self.hashes, configs=self.configs)
        with self.assertRaises(FileExistsError):
            test_stage(self.args, self.parts, self.spatial, self.configs[name], self.hashes, configs=self.configs)

    def test_duplicate_name_and_invalid_budget_are_rejected(self):
        (self.plugins / 'duplicate.py').write_text(PLUGIN.replace('class Probe', 'class Duplicate'))
        importlib.invalidate_caches()
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            discover_models()
        (self.plugins / 'duplicate.py').unlink();sys.modules.pop('source.models.duplicate', None)
        (self.plugins / 'bad.py').write_text(PLUGIN.replace('class Probe', 'class Bad').replace("name = 'probe'", "name = 'bad'").replace('[1, 3]', '[3, 1]'))
        importlib.invalidate_caches()
        with self.assertRaisesRegex(ValueError, 'budgets'):
            discover_models()

    def test_reporting_rejects_misaligned_common_test_samples(self):
        self.train_all();self.evaluate_all()
        folder = self.args.output_dir / next(iter(self.configs))
        with np.load(folder / 'test_predictions.npz') as z:
            data = {k: z[k] for k in z.files}
        data['origins'] = data['origins'] + np.timedelta64(1, 'h')
        np.savez_compressed(folder / 'test_predictions.npz', **data)
        with self.assertRaises(AssertionError):
            generate_report(self.args.output_dir)


if __name__ == '__main__':
    unittest.main()
