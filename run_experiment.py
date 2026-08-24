"""Run a reproducible KIMORE experiment from the project root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_evaluation import run_cross_validated_evaluation  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "kimore_audit_output" / "kimore_manifest.csv",
        help="Audit manifest (default: ./kimore_audit_output/kimore_manifest.csv)",
    )
    parser.add_argument(
        "--exercise",
        default="Es3",
        choices=[f"Es{i}" for i in range(1, 6)],
    )
    parser.add_argument(
        "--method",
        default="plain_dtw",
        choices=["plain_dtw"],
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=5000,
        help="Subject-level bootstrap resamples (default: 5000)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "plain_dtw",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        default=PROJECT_ROOT / "figures" / "plain_dtw",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.method != "plain_dtw":
        raise ValueError(f"Unsupported method: {args.method}")
    run_cross_validated_evaluation(
        manifest_path=args.manifest.expanduser().resolve(),
        exercise=args.exercise,
        output_dir=args.output_dir.expanduser().resolve(),
        figure_dir=args.figure_dir.expanduser().resolve(),
        bootstrap_resamples=args.bootstrap_resamples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
