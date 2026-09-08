"""Create a sidecar inventory of unresolved source anomalies; never edit raw data."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from kimore_run_provenance import sha256
from kimore_dataset_audit import workbook_provenance_warnings, audit_numeric_csv


def main():
    raw = ROOT / 'data/raw/KIMORE'
    workbook_entries = []
    for path in sorted(raw.glob('**/Label/ClinicalAssessment_*.xlsx')):
        subject = path.parents[2].name
        warnings = workbook_provenance_warnings(path, subject)
        if warnings:
            workbook_entries.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                                     'warnings': warnings, 'decision': 'unresolved; preserve source labels'})
    groups = {}
    for path in sorted(raw.glob('**/Raw/JointPosition*.csv')):
        groups.setdefault(sha256(path), []).append(path)
    duplicates = []
    for digest, paths in groups.items():
        if len(paths) < 2:
            continue
        canonical = min(paths, key=lambda p: ('(' in p.name, p.name))
        companions = []
        for path in sorted(canonical.parent.iterdir()):
            if path.suffix.lower() != '.csv':
                continue
            audit = audit_numeric_csv(path)
            companions.append({'path': str(path.relative_to(ROOT)), 'sha256': sha256(path),
                               'rows': audit.rows, 'inconsistent_rows': audit.inconsistent_rows,
                               'nonnumeric_rows': audit.nonnumeric_rows})
        rgb = [{'path': str(p.relative_to(ROOT)), 'sha256': sha256(p)}
               for p in sorted(canonical.parent.parent.glob('rgb/*.mp4'))]
        duplicates.append({'position_sha256': digest,
                           'copies': [str(p.relative_to(ROOT)) for p in paths],
                           'proposed_canonical': str(canonical.relative_to(ROOT)),
                           'companions': companions, 'rgb': rgb,
                           'decision': 'pending timestamp/orientation/RGB alignment and QC; not admitted to evaluation'})
    target = ROOT / 'annotations/source_anomaly_registry.json'
    target.write_text(json.dumps({'version': 1, 'workbooks': workbook_entries, 'duplicate_positions': duplicates},
                                indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'Recorded {len(workbook_entries)} workbook anomalies and {len(duplicates)} duplicate groups in {target}')


if __name__ == '__main__':
    main()
