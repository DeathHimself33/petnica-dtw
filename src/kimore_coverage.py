"""Report evaluation denominators and exclusion stages by exercise and cohort."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from kimore_dataset import explain_position_exclusion


def coverage_rows(manifest_path: Path, retained_ids, exclusions: dict[str, str], exercises=None) -> list[dict]:
    with manifest_path.open(encoding='utf-8-sig', newline='') as handle:
        manifest = list(csv.DictReader(handle))
    if exercises is not None:
        manifest = [row for row in manifest if row['exercise'] in exercises]
    ids = [row['sample_id'] for row in manifest]
    retained = list(retained_ids)
    if len(set(ids)) != len(ids) or len(set(retained)) != len(retained):
        raise ValueError('Coverage requires unique sample IDs')
    retained = set(retained)
    if retained & set(exclusions) or retained | set(exclusions) != set(ids):
        raise ValueError('Retained and excluded samples must partition the manifest exactly')
    groups = {}
    for row in manifest:
        key = row['exercise'], row['cohort']
        group = groups.setdefault(key, {'total': 0, 'retained': 0, 'source_excluded': 0,
                                       'processing_excluded': 0, 'reasons': Counter()})
        group['total'] += 1
        if row['sample_id'] in retained:
            if explain_position_exclusion(row):
                raise ValueError('Retained sample is not source-eligible')
            group['retained'] += 1
        else:
            stage = 'source_excluded' if explain_position_exclusion(row) else 'processing_excluded'
            group[stage] += 1
            group['reasons'][exclusions[row['sample_id']]] += 1
    return [{'exercise': exercise, 'cohort': cohort, **group,
             'coverage_fraction': group['retained'] / group['total'],
             'reasons': dict(sorted(group['reasons'].items()))}
            for (exercise, cohort), group in sorted(groups.items())]
