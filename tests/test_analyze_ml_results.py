from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analyze_ml_results import main  # noqa: E402


class MultiSeedAnalysisTests(unittest.TestCase):
    def write_run(
        self, root: Path, name: str, seed: int, predictions: list[float]
    ) -> Path:
        run = root / name
        run.mkdir()
        rows = []
        for index, prediction in enumerate(predictions, start=1):
            rows.append(
                {
                    "fold": index,
                    "validation_fold": index % 5 + 1,
                    "sample_id": f"S{index}_Es1",
                    "subject_id": f"S{index}",
                    "cohort": "synthetic",
                    "exercise": "Es1",
                    "actual_ts": float(index * 10),
                    "predicted_ts": prediction,
                    "training_exercise_mean_ts": 30.0,
                    "absolute_error": abs(prediction - index * 10),
                    "quality_status": "pass",
                    "retained_fraction": 1.0,
                }
            )
        with (run / "oof_predictions.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "completed_folds": [1, 2, 3, 4, 5],
                    "configuration": {"seed": seed},
                    "oof_samples": len(rows),
                }
            ),
            encoding="utf-8",
        )
        return run

    def write_dtw(self, root: Path) -> Path:
        path = root / "dtw.csv"
        rows = [
            {
                "sample_id": f"S{index}_Es1",
                "subject_id": f"S{index}",
                "exercise": "Es1",
                "fold": index,
                "actual_ts": float(index * 10),
                "qc_predicted_ts": 30.0,
            }
            for index in range(1, 6)
        ]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return path

    def test_writes_mean_ensemble_and_subject_bootstrap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_run(root, "first", 1, [11, 19, 31, 39, 49])
            second = self.write_run(root, "second", 2, [9, 21, 29, 41, 51])
            output = root / "analysis"
            result = main(
                [
                    "--run-dir",
                    str(first),
                    "--run-dir",
                    str(second),
                    "--dtw-predictions",
                    str(self.write_dtw(root)),
                    "--output-dir",
                    str(output),
                    "--bootstrap-resamples",
                    "20",
                ]
            )

            self.assertEqual(result, 0)
            summary = json.loads(
                (output / "multiseed_summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["seeds"], [1, 2])
            self.assertAlmostEqual(summary["ensemble_metrics"]["overall"]["mae"], 0.0)
            self.assertEqual(summary["population"]["qc_usable_samples"], 5)
            self.assertIsNone(summary["population"]["full_samples"])
            with (output / "ensemble_oof_predictions.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(float(rows[0]["ensemble_predicted_ts"]), 10.0)
            self.assertTrue((output / "bootstrap_intervals.csv").is_file())

    def test_rejects_duplicate_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_run(root, "first", 1, [10, 20, 30, 40, 50])
            second = self.write_run(root, "second", 1, [10, 20, 30, 40, 50])
            with self.assertRaisesRegex(ValueError, "distinct random seed"):
                main(
                    [
                        "--run-dir",
                        str(first),
                        "--run-dir",
                        str(second),
                        "--dtw-predictions",
                        str(self.write_dtw(root)),
                    ]
                )

    def test_rejects_mismatched_dtw_fold(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self.write_run(root, "first", 1, [10, 20, 30, 40, 50])
            second = self.write_run(root, "second", 2, [10, 20, 30, 40, 50])
            dtw = self.write_dtw(root)
            with dtw.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            rows[0]["fold"] = "5"
            with dtw.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            with self.assertRaisesRegex(ValueError, "QC-DTW field 'fold'"):
                main(
                    [
                        "--run-dir",
                        str(first),
                        "--run-dir",
                        str(second),
                        "--dtw-predictions",
                        str(dtw),
                        "--bootstrap-resamples",
                        "5",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
