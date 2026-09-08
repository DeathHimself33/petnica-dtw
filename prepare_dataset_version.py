"""Build a versioned, source-checked manifest and canonical-copy registry."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))

from kimore_dataset_audit import build_manifest, write_manifest, write_summary
from kimore_source_selection import build_selection_registry
from kimore_run_provenance import atomic_json, sha256


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'data/raw/KIMORE')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory to preserve earlier dataset versions')
    output.mkdir(parents=True, exist_ok=True)
    root = args.root.resolve()
    registry = build_selection_registry(root)
    atomic_json(output / 'source_selection.json', registry)
    rows = build_manifest(root, registry['bundles'])
    manifest = output / 'kimore_manifest.csv'
    write_manifest(rows, manifest)
    write_summary(rows, output / 'kimore_audit_summary.txt')
    sources = {str(Path(row[field]).relative_to(root)): sha256(Path(row[field]))
               for row in rows for field in ('position_path', 'orientation_path', 'timestamp_path')
               if row[field]}
    sources.update({str(p.relative_to(root)): sha256(p) for p in sorted(root.glob('**/Label/*.xlsx'))})
    atomic_json(output / 'dataset_version.json', {
        'version': 2, 'manifest_sha256': sha256(manifest),
        'source_selection_sha256': sha256(output / 'source_selection.json'),
        'policy': 'identity-consistent label copies; verified identical recording bundles; source TS never inferred',
        'sources': sources,
        'code': {str(p.relative_to(ROOT)): sha256(p) for p in
                 [Path(__file__).resolve(), *sorted((ROOT / 'src').glob('*.py'))]},
    })
    print(f'Wrote {len(rows)} manifest rows; {len(registry["bundles"])} canonical bundles selected; '
          f'{len(registry["rejected"])} ambiguous bundles rejected. Output: {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
