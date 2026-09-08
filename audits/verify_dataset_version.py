"""Verify the corrected export against the frozen baseline on common samples."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kimore_run_provenance import sha256, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, default=ROOT / 'results/ml_data/kimore_all_exercises_128.npz')
    parser.add_argument('--version-dir', type=Path, default=ROOT / 'results/dataset_v2_verified')
    args = parser.parse_args()
    new_path = args.version_dir / 'kimore_all_exercises_128.npz'
    metadata = json.loads(new_path.with_suffix('.json').read_text())
    assert metadata['npz_sha256'] == sha256(new_path)
    assert metadata['manifest_sha256'] == sha256(args.version_dir / 'kimore_manifest.csv')
    with np.load(args.baseline, allow_pickle=False) as old, np.load(new_path, allow_pickle=False) as new:
        old_index = {str(value): index for index, value in enumerate(old['sample_ids'])}
        new_index = {str(value): index for index, value in enumerate(new['sample_ids'])}
        common = sorted(old_index.keys() & new_index.keys())
        checked = []
        for name in new.files:
            left = old[name][[old_index[sample] for sample in common]]
            right = new[name][[new_index[sample] for sample in common]]
            np.testing.assert_array_equal(left, right, err_msg=f'Common sample array changed: {name}')
            checked.append(name)
        report = {'baseline_samples': len(old_index), 'new_samples': len(new_index),
                  'common_samples': len(common), 'identical_arrays_on_common_samples': checked,
                  'added_samples': sorted(new_index.keys() - old_index.keys()),
                  'excluded_since_baseline': sorted(old_index.keys() - new_index.keys()),
                  'baseline_npz_sha256': sha256(args.baseline), 'new_npz_sha256': sha256(new_path)}
    coverage = metadata['coverage_by_exercise_cohort']
    assert sum(row['total'] for row in coverage) == 390
    assert sum(row['retained'] for row in coverage) == report['new_samples']
    report['coverage'] = coverage
    atomic_json(args.version_dir / 'verification.json', report)
    print(json.dumps({key: value for key, value in report.items() if key != 'coverage'}, indent=2))


if __name__ == '__main__':
    main()
