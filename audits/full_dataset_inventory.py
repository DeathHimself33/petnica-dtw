"""Read-only whole-dataset inventory, including excluded recordings."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import re
from collections import Counter, defaultdict
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from kimore_dataset import load_joint_positions
from kimore_dataset_audit import build_manifest, read_numeric_csv

OUT = ROOT/'results/full_project_audit'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'kimore_audit_output/kimore_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
        old = list(csv.DictReader(f))
    print('Re-auditing all source files and workbooks...', flush=True)
    fresh = build_manifest(ROOT/'data/raw/KIMORE')
    differences = []
    for a, b in zip(old, fresh):
        for key in a:
            if a[key] != ('' if b[key] is None else str(b[key])):
                differences.append(dict(sample=a['sample_id'], field=key, old=a[key], new=b[key]))
    hashes = defaultdict(list)
    records = []
    for i, path in enumerate(sorted((ROOT/'data/raw/KIMORE').rglob('JointPosition*.csv'))):
        exercise_dir = next(p for p in path.parents if re.fullmatch(r'Es\d+', p.name))
        row = dict(path=str(path.relative_to(ROOT)), subject=exercise_dir.parent.name, exercise=exercise_dir.name)
        hashes[hashlib.sha256(path.read_bytes()).hexdigest()].append(row['path'])
        try:
            seq = load_joint_positions(path)
            row.update(frames=len(seq.positions), inferred_fraction=float((seq.tracking_states == 1).mean()), untracked_fraction=float((seq.tracking_states == 0).mean()), duplicate_adjacent_frames=int(np.all(seq.positions[1:] == seq.positions[:-1], axis=(1, 2)).sum()))
            stamp = path.with_name(path.name.replace('JointPosition', 'TimeStamp'))
            orient = path.with_name(path.name.replace('JointPosition', 'JointOrientation'))
            row['timestamp_exists'] = stamp.exists()
            if stamp.exists():
                ts = read_numeric_csv(stamp)
                row['timestamp_shape'] = list(ts.shape)
                row['timestamp_finite'] = bool(np.isfinite(ts).all())
                dt = np.diff(ts.reshape(-1))
                median = np.median(dt[dt > 0]) if (dt > 0).any() else np.nan
                row.update(timestamp_frame_delta=len(ts)-len(seq.positions), nonincreasing=int((dt <= 0).sum()), gap_over_2x=int((dt > 2*median).sum()), gap_over_5x=int((dt > 5*median).sum()), max_step_ratio=float(dt.max()/median) if len(dt) else None)
            row['orientation_exists'] = orient.exists()
            if orient.exists():
                q = read_numeric_csv(orient)
                row.update(orientation_shape=list(q.shape), orientation_finite=bool(np.isfinite(q).all()), orientation_frame_delta=len(q)-len(seq.positions))
        except Exception as error:
            row['load_error'] = str(error)
        records.append(row)
        if (i+1) % 100 == 0:
            print(f'Inspected {i+1} raw recordings', flush=True)
    summary = dict(manifest_rows_old=len(old), manifest_rows_new=len(fresh), manifest_differences=differences, raw_position_recordings=len(records), duplicate_file_groups=[v for v in hashes.values() if len(v)>1], raw_by_exercise=dict(Counter(r['exercise'] for r in records)), load_errors=[r for r in records if 'load_error' in r], anomalies=[r for r in records if r.get('nonincreasing', 0) or r.get('timestamp_frame_delta', 0) or r.get('orientation_frame_delta', 0) or r.get('duplicate_adjacent_frames', 0)], timestamp_gaps_by_exercise={ex: dict(recordings=sum(r['exercise']==ex for r in records), with_over_2x=sum(r['exercise']==ex and r.get('gap_over_2x', 0)>0 for r in records), with_over_5x=sum(r['exercise']==ex and r.get('gap_over_5x', 0)>0 for r in records)) for ex in ['Es1','Es2','Es3','Es4','Es5']}, records=records)
    (OUT/'dataset_inventory.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['records','anomalies','manifest_differences']}, indent=2))
    print('Manifest differences:', len(differences), 'anomalous raw recordings:', len(summary['anomalies']))


if __name__ == '__main__':
    main()
