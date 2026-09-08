from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from kimore_ml_data import EXERCISES
    from kimore_ml_model import TemporalScoreModel, trainable_parameter_count
    from train_ml_baseline import (
        denormalize_targets,
        fit_target_standardizer,
        normalize_targets,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch ML extras are not installed")
class MLModelTests(unittest.TestCase):
    def test_training_partial_resume_and_verified_aggregation(self) -> None:
        import tempfile
        import contextlib
        import io
        from train_ml_baseline import main
        from analyze_ml_results import load_run
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rng = np.random.default_rng(12)
            data = root / 'data.npz'
            np.savez(data, features=rng.normal(size=(25, 8, 9, 3)).astype(np.float32),
                     frame_mask=np.ones((25, 8), dtype=bool),
                     component_observed_mask=np.ones((25, 8, 9), dtype=bool),
                     targets=np.arange(25, dtype=np.float32),
                     exercise_indices=np.tile(np.arange(5), 5),
                     fold_numbers=np.repeat(np.arange(1, 6), 5),
                     sample_ids=np.asarray([f'S{i}_Es{e}' for i in range(5) for e in range(5)]),
                     subject_ids=np.repeat([f'S{i}' for i in range(5)], 5),
                     cohorts=np.repeat('synthetic', 25), quality_statuses=np.repeat('pass', 25),
                     retained_fractions=np.ones(25, dtype=np.float32))
            args = ['--data', str(data), '--output-dir', str(root / 'run'), '--epochs', '1',
                    '--channels', '8', '--gru-hidden', '4', '--device', 'cpu', '--no-amp']
            old_threads = torch.get_num_threads()
            torch.set_num_threads(1)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    main([*args, '--fold', '1'])
                    main([*args, '--resume'])
                    # Requesting one completed fold must preserve the full OOF summary.
                    main([*args, '--resume', '--fold', '1'])
                _, summary, rows = load_run(root / 'run')
                self.assertEqual(len(rows), 25)
                self.assertEqual(summary['completed_folds'], [1, 2, 3, 4, 5])
                import csv
                with (root / 'run/fold_1/history.csv').open(newline='') as handle:
                    history = list(csv.DictReader(handle))
                self.assertTrue(all(f'validation_mae_Es{number}' in history[0] for number in range(1, 6)))
                self.assertEqual(sum(int(history[0][f'validation_samples_Es{number}']) for number in range(1, 6)), 5)
                with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                    main([*args, '--resume', '--seed', '123'])
            finally:
                torch.set_num_threads(old_threads)

    def test_model_returns_one_score_and_normalized_attention_per_sample(self) -> None:
        model = TemporalScoreModel(channels=16, gru_hidden=8, dropout=0.0)
        vectors = torch.randn(3, 12, 9, 3)
        frame_mask = torch.ones(3, 12, dtype=torch.bool)
        frame_mask[0, -2:] = False
        observed = frame_mask.unsqueeze(2).expand(-1, -1, 9).clone()
        exercise = torch.tensor([0, 2, 4])

        predictions, attention = model(
            vectors, frame_mask, observed, exercise, return_attention=True
        )

        self.assertEqual(tuple(predictions.shape), (3,))
        self.assertEqual(tuple(attention.shape), (3, 12))
        torch.testing.assert_close(attention.sum(dim=1), torch.ones(3))
        self.assertTrue(torch.all(attention[0, -2:] == 0))
        self.assertGreater(trainable_parameter_count(model), 0)

    def test_target_standardization_is_per_exercise_and_reversible(self) -> None:
        targets = np.asarray([10.0, 20.0, 30.0, 50.0], dtype=np.float32)
        exercises = np.asarray([0, 0, 1, 1], dtype=np.int64)
        # Duplicate both pairs for the otherwise absent exercise heads.
        targets = np.concatenate((targets, [1.0, 2.0, 3.0]))
        exercises = np.concatenate((exercises, [2, 3, 4]))
        train_indices = np.arange(len(targets))

        standardizer = fit_target_standardizer(
            targets, exercises, train_indices
        )
        normalized = normalize_targets(targets, exercises, standardizer)
        restored = denormalize_targets(normalized, exercises, standardizer)

        np.testing.assert_allclose(restored, targets)
        self.assertEqual(len(standardizer.mean), len(EXERCISES))


if __name__ == "__main__":
    unittest.main()
