"""Regression tests for aligned inputs and the notebook checkpoint contract."""

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from source import build_experiment_matrices, comparison
from source.experiment_schema import BASE_COLUMNS, RECENT_COLUMNS
from source.workflow import discover_configs, main


class ComparisonTests(unittest.TestCase):
    def test_final_capacity_refits_in_same_order_and_scores_snapshot(self):
        x = np.arange(6).reshape(3, 2)
        y = np.repeat(x[:, :1], 6, axis=1)
        config = {
            "input_profile": "east",
            "expanded_training": False,
            "shuffle": True,
            "seed": 42,
            "training_cap": 3,
            "budgets": [1, 3],
        }
        search, final = Mock(), Mock()
        search.predict.side_effect = lambda x, budget: np.full((len(x), 6), budget)
        search.training_capacity.return_value = 1
        final.snapshot.return_value.predict.return_value = np.full((1, 6), 2)
        valid = Mock(actual=np.ones((1, 6)), current=np.zeros(1))
        designs = {("east", False): (x, y, x[:1], None)}

        with patch.object(comparison, "new_model", side_effect=[search, final]):
            # The public fit contract returns the fitted model.
            final.fit.return_value = final
            payload = comparison.train_candidate(config, designs, valid, threads=1)

        search.training_capacity.assert_called_once_with(1, 3)
        self.assertEqual(final.fit.call_args.args[2], 1)
        for search_data, final_data in zip(search.fit.call_args.args[:2],
                                           final.fit.call_args.args[:2]):
            np.testing.assert_array_equal(search_data, final_data)
        final.snapshot.assert_called_once_with(1)
        self.assertEqual(payload["validation"]["mse"], 1)

    def test_matrix_loader_rejects_permuted_weather_names(self):
        archive = io.BytesIO()
        names = list(RECENT_COLUMNS)
        names[0], names[1] = names[1], names[0]
        np.savez(archive, base_names=BASE_COLUMNS, recent_names=names)
        archive.seek(0)
        with self.assertRaisesRegex(ValueError, "out of order: recent_names"):
            comparison.load_cohort(archive, "test")

    def test_matrix_builder_rejects_permuted_weather_names(self):
        names = list(RECENT_COLUMNS)
        names[0], names[1] = names[1], names[0]
        archive = io.BytesIO()
        np.savez(
            archive,
            times=build_experiment_matrices.TIME_AXIS,
            legacy_names=build_experiment_matrices.EAST_CHANNELS[1:],
            legacy=np.zeros((len(build_experiment_matrices.TIME_AXIS), 5)),
            recent=np.zeros((len(build_experiment_matrices.TIME_AXIS), len(names))),
            recent_names=names,
        )
        archive.seek(0)
        with (patch.object(build_experiment_matrices, "read_pm", return_value=(None, "")),
              patch.object(build_experiment_matrices, "read_historical", return_value=(None, None))):
            with self.assertRaisesRegex(ValueError, "Recent weather columns.*out of order"):
                build_experiment_matrices.build(archive)

    def test_prediction_rejects_changed_implementation_before_loading_data(self):
        manifest = {
            "schema_version": comparison.CHECKPOINT_SCHEMA,
            "seed": 42,
            "matrix_sha256": "unchanged",
            "implementation_sha256": {"source/experiment_features.py": "before"},
            "candidates": {"candidate": {}},
        }
        with (
            patch.object(Path, "read_text", return_value=json.dumps(manifest)),
            patch.object(comparison, "file_sha256", return_value="unchanged"),
            patch.object(comparison, "comparison_configs", return_value={"candidate": {}}),
            patch.object(comparison, "implementation_hashes",
                         return_value={"source/experiment_features.py": "after"}),
            patch.object(comparison, "load_cohort") as load,
        ):
            with self.assertRaisesRegex(ValueError, "Model or feature code changed"):
                comparison.predict_comparison("matrix.npz", "run")
        load.assert_not_called()

    def test_cli_rejects_notebook_only_models_in_preview_and_run(self):
        for flags in ([], ["--dry-run"], ["--list-models"]):
            with self.subTest(flags=flags):
                with patch("sys.argv", ["main.py", "--models", "analog", *flags]):
                    with redirect_stderr(io.StringIO()) as error:
                        with self.assertRaises(SystemExit) as raised:
                            main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn("Use the training notebook", error.getvalue())

    def test_scoped_configs_are_independent(self):
        configs = discover_configs()
        east = configs["catboost_east_depth4"]
        five = configs["catboost_five_depth4"]
        east["budgets"].append(1)
        east["model_parameters"]["depth"] = 6
        self.assertNotIn(1, five["budgets"])
        self.assertEqual(five["model_parameters"]["depth"], 4)
