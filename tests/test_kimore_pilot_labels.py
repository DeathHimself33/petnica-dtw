from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kimore_apply_pilot_labels import merge_labels, second_review_rows  # noqa: E402


class PilotLabelTest(unittest.TestCase):
    def read_rows(self, path: Path) -> list[dict[str, str]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_tracked_labels_cover_the_generated_pilot_contract(self) -> None:
        labels = self.read_rows(ROOT / "annotations" / "kimore_es3_pilot_labels.csv")
        queue = [
            {
                "sample_id": row["sample_id"],
                "candidate_rank": row["candidate_rank"],
                "cohort": "test",
                "component_name": "test_component",
                **{
                    field: ""
                    for field in (
                        "review_status",
                        "execution_label",
                        "error_type",
                        "severity",
                        "reviewer_confidence",
                        "annotator",
                        "review_notes",
                    )
                },
            }
            for row in labels
        ]
        merged = merge_labels(queue, labels)
        self.assertEqual(len(merged), 100)
        self.assertEqual(
            {row["execution_label"] for row in merged},
            {"correct", "error", "uncertain"},
        )

    def test_second_review_is_one_row_per_recording_and_rank_balanced(self) -> None:
        labels = self.read_rows(ROOT / "annotations" / "kimore_es3_pilot_labels.csv")
        queue = [
            {
                **row,
                "execution_label": "",
                "review_status": "",
                "error_type": "",
                "severity": "",
                "reviewer_confidence": "",
                "annotator": "",
                "review_notes": "",
            }
            for row in labels
        ]
        selected = second_review_rows(queue)
        self.assertEqual(len(selected), 20)
        self.assertEqual(len({row["sample_id"] for row in selected}), 20)
        rank_counts = {
            rank: sum(row["candidate_rank"] == rank for row in selected)
            for rank in ("1", "2", "3", "4", "5")
        }
        self.assertEqual(rank_counts, {rank: 4 for rank in rank_counts})
        self.assertTrue(all(not row["execution_label"] for row in selected))


if __name__ == "__main__":
    unittest.main()
