from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from openpyxl import Workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kimore_source_selection import select_identical_bundle
from kimore_dataset_audit import read_subject_scores
from kimore_coverage import coverage_rows


class SourceSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.folder = self.root / 'E_ID14/Es4/Raw'
        self.folder.mkdir(parents=True)
        position = np.zeros((3, 25, 4))
        position[:, :, 3] = 2
        for stream, values in [('JointPosition', position.reshape(3, 100)),
                               ('JointOrientation', np.ones((3, 100))),
                               ('TimeStamp', np.arange(3)[:, None])]:
            for suffix in ('', '(1)'):
                np.savetxt(self.folder / f'{stream}123{suffix}.csv', values, delimiter=',')

    def test_identical_complete_bundle_selects_canonical_without_deleting(self):
        result = select_identical_bundle(self.folder)
        self.assertEqual(result['frames'], 3)
        self.assertTrue(result['selected']['JointPosition'].endswith('JointPosition123.csv'))
        self.assertEqual(len(list(self.folder.iterdir())), 6)

    def test_different_orientation_copy_is_not_collapsed(self):
        np.savetxt(self.folder / 'JointOrientation123(1).csv', np.full((3, 100), 2), delimiter=',')
        with self.assertRaisesRegex(ValueError, 'copies differ'):
            select_identical_bundle(self.folder)

    def test_matching_copies_with_bad_timing_are_rejected(self):
        for suffix in ('', '(1)'):
            np.savetxt(self.folder / f'TimeStamp123{suffix}.csv', np.array([[1], [1], [2]]), delimiter=',')
        with self.assertRaisesRegex(ValueError, 'increase strictly'):
            select_identical_bundle(self.folder)

    def test_mismatched_frame_counts_are_rejected(self):
        for suffix in ('', '(1)'):
            np.savetxt(self.folder / f'TimeStamp123{suffix}.csv', np.array([[1], [2]]), delimiter=',')
        with self.assertRaisesRegex(ValueError, 'frame counts differ'):
            select_identical_bundle(self.folder)


class SourceIdentityTests(unittest.TestCase):
    def write(self, subject, exercise, internal_id, score):
        path = subject / exercise / 'Label' / f'ClinicalAssessment_{subject.name}.xlsx'
        path.parent.mkdir(parents=True)
        workbook = Workbook()
        workbook.active.append(['Subject ID', 'Clinical TS Ex #1', 'Clinical PO Ex #1', 'Clinical CF Ex #1'])
        workbook.active.append([internal_id, score, 15, 29])
        workbook.save(path)
        workbook.close()

    def test_mismatched_copy_is_ignored_in_favor_of_identity_consistent_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            subject = Path(temporary) / 'E_ID16'
            self.write(subject, 'Es1', 'E_ID17', 20)
            self.write(subject, 'Es2', 'E_ID16', 44)
            scores, _, warnings = read_subject_scores(subject)
            self.assertEqual(scores['ts', 1], 44)
            self.assertTrue(any('E_ID17' in value for value in warnings))

    def test_all_mismatched_copies_exclude_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            subject = Path(temporary) / 'NE_ID2'
            self.write(subject, 'Es1', 'E_ID1', 44)
            scores, issues, _ = read_subject_scores(subject)
            self.assertFalse(scores)
            self.assertTrue(all('unresolved clinical Subject ID' in values[0] for values in issues.values()))

    def test_sum_discrepancy_preserves_identity_consistent_source_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            subject = Path(temporary) / 'E_ID3'
            self.write(subject, 'Es1', 'E_ID3', 44.6667)
            scores, _, warnings = read_subject_scores(subject)
            self.assertEqual(scores['ts', 1], 44.6667)
            self.assertTrue(any('PO+CF' in value for value in warnings))


class CoverageTests(unittest.TestCase):
    def test_reports_manifest_and_processing_exclusions_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'manifest.csv'
            rows = [dict(sample_id=str(i), exercise='Es3', cohort='stroke', clinical_ts='40',
                         position_frames='3', position_columns='100', position_path='source.csv', issues='')
                    for i in range(3)]
            rows[0]['clinical_ts'] = ''
            with path.open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            result = coverage_rows(path, ['2'], {'0': 'missing target', '1': 'QC failure'})[0]
            self.assertEqual(result['total'], 3)
            self.assertEqual(result['source_excluded'], 1)
            self.assertEqual(result['processing_excluded'], 1)
            self.assertEqual(result['coverage_fraction'], 1/3)
            with self.assertRaisesRegex(ValueError, 'partition'):
                coverage_rows(path, ['2'], {'0': 'missing target'})


class FoldReferenceTests(unittest.TestCase):
    def test_complete_json_preserves_qc_excluded_subjects(self):
        import json
        from kimore_grouping import load_subject_fold_assignments
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'subject_folds.json'
            payload = {'fold_numbering': 'one_based', 'folds': [
                {'fold': number, 'subjects': [f'S{number}']} for number in range(1, 6)]}
            payload['folds'][0]['subjects'].append('QC_EXCLUDED')
            path.write_text(json.dumps(payload))
            result = load_subject_fold_assignments(path)
            self.assertEqual(result['QC_EXCLUDED'], 0)
            payload['folds'][1]['subjects'].append('QC_EXCLUDED')
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, 'repeats subjects'):
                load_subject_fold_assignments(path)

    def test_npz_reference_rejects_cross_fold_subject(self):
        from kimore_grouping import load_subject_fold_assignments
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'folds.npz'
            np.savez(path, subject_ids=['S1', 'S1', 'S3', 'S4', 'S5'], fold_numbers=[1, 2, 3, 4, 5])
            with self.assertRaisesRegex(ValueError, 'multiple folds'):
                load_subject_fold_assignments(path)


if __name__ == '__main__':
    unittest.main()
