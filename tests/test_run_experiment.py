from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from run_experiment import (  # noqa: E402
    _combine_csv_files,
    _fold_payload,
    _run_root,
    parse_args,
)


class RunExperimentTests(unittest.TestCase):
    def test_all_exercises_is_available_for_interpretable_method(self) -> None:
        args = parse_args(
            ["--exercise", "all", "--method", "interpretable_dtw"]
        )

        self.assertEqual(args.exercise, "all")

    def test_all_exercises_rejects_es3_only_plain_method(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--exercise", "all", "--method", "plain_dtw"])

    def test_all_exercises_rejects_strict_yu_xiong_method(self) -> None:
        with self.assertRaises(SystemExit):
            parse_args(["--exercise", "all", "--method", "yu_xiong_dtw"])

    def test_default_batch_root_isolated_from_single_exercise_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            batch = _run_root(None, root / "interpretable_dtw", "all")
            single = _run_root(None, root / "interpretable_dtw", "Es3")

            self.assertEqual(
                batch,
                (root / "interpretable_dtw" / "all_exercises").resolve(),
            )
            self.assertEqual(
                single,
                (root / "interpretable_dtw" / "Es3").resolve(),
            )
            self.assertNotEqual(batch / "Es3", single)

    def test_custom_batch_root_keeps_the_requested_location(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            requested = Path(directory) / "custom"

            resolved = _run_root(
                requested,
                Path(directory) / "unused",
                "all",
            )

            self.assertEqual(resolved, requested.resolve())

    def test_combined_csv_retains_explicit_exercise(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs = []
            for exercise in ("Es1", "Es2"):
                path = root / f"{exercise}.csv"
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.DictWriter(
                        handle, fieldnames=["exercise", "sample_id"]
                    )
                    writer.writeheader()
                    writer.writerow(
                        {"exercise": exercise, "sample_id": f"A_{exercise}"}
                    )
                inputs.append(path)

            output = root / "combined.csv"
            rows = _combine_csv_files(inputs, output)

            self.assertEqual(rows, 2)
            with output.open(encoding="utf-8", newline="") as handle:
                combined = list(csv.DictReader(handle))
            self.assertEqual(
                [row["exercise"] for row in combined],
                ["Es1", "Es2"],
            )

    def test_persisted_fold_numbers_are_one_based(self) -> None:
        payload = _fold_payload({"A": 0, "B": 1, "C": 0})

        self.assertEqual(payload["fold_numbering"], "one_based")
        self.assertEqual(payload["folds"][0], {"fold": 1, "subjects": ["A", "C"]})
        self.assertEqual(payload["folds"][1], {"fold": 2, "subjects": ["B"]})


if __name__ == "__main__":
    unittest.main()
