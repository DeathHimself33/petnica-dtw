from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from openpyxl import Workbook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_dataset_audit import read_subject_scores, write_summary  # noqa: E402


def write_score_workbook(path: Path, ts_values: list[float | None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook = Workbook()
    sheet = workbook.active
    sheet.append([f"Clinical TS Ex #{number}" for number in range(1, 6)])
    sheet.append(ts_values)
    workbook.save(str(path))
    workbook.close()


class ClinicalWorkbookAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.subject_dir = Path(self.temporary.name) / "E_ID3"

    def test_misfiled_other_subject_workbook_is_ignored_and_warned(self) -> None:
        write_score_workbook(
            self.subject_dir / "Es1" / "Label" / "ClinicalAssessment_E_ID3.xlsx",
            [41.0, 42.0, 43.0, 44.0, 45.0],
        )
        write_score_workbook(
            self.subject_dir / "Es3" / "Label" / "ClinicalAssessment_E_ID1.xlsx",
            [1.0, 2.0, 3.0, 4.0, 5.0],
        )

        scores, issues, warnings = read_subject_scores(self.subject_dir)

        self.assertEqual(scores[("ts", 3)], 43.0)
        self.assertFalse(any(issues.values()))
        self.assertIn("ClinicalAssessment_E_ID1.xlsx", warnings[0])

    def test_conflicting_correctly_named_copies_are_rejected(self) -> None:
        for exercise, suffix, third_score in (
            ("Es1", "", 43.0),
            ("Es2", "(1)", 13.0),
        ):
            write_score_workbook(
                self.subject_dir
                / exercise
                / "Label"
                / f"ClinicalAssessment_E_ID3{suffix}.xlsx",
                [41.0, 42.0, third_score, 44.0, 45.0],
            )

        scores, issues, _ = read_subject_scores(self.subject_dir)

        self.assertEqual(scores, {})
        self.assertTrue(
            all("conflicting correctly named" in messages[0] for messages in issues.values())
        )

    def test_missing_score_problem_stays_with_its_exercise(self) -> None:
        write_score_workbook(
            self.subject_dir / "Es1" / "Label" / "ClinicalAssessment_E_ID3.xlsx",
            [41.0, 42.0, None, 44.0, 45.0],
        )

        scores, issues, _ = read_subject_scores(self.subject_dir)

        self.assertEqual(scores[("ts", 2)], 42.0)
        self.assertEqual(issues[1], [])
        self.assertTrue(any("Es3" in message for message in issues[3]))


class AuditSummaryTests(unittest.TestCase):
    def test_summary_separates_full_audit_status_from_model_usability(self) -> None:
        rows = [
            {
                "subject_id": "A",
                "sample_id": "A_Es3",
                "cohort": "synthetic",
                "exercise": "Es3",
                "status": "problem",
                "position_target_usable": "true",
                "clinical_ts": 30.0,
                "issues": "missing timestamp",
                "warnings": "",
            }
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "summary.txt"
            write_summary(rows, output)
            text = output.read_text(encoding="utf-8")

        self.assertIn("Rows fully OK: 0", text)
        self.assertIn("Position+target usable recordings by exercise:", text)
        self.assertIn("  Es3: 1", text)


if __name__ == "__main__":
    unittest.main()
