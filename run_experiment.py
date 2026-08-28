"""Run a reproducible KIMORE experiment from the project root."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from kimore_evaluation import run_cross_validated_evaluation  # noqa: E402
from kimore_yu_xiong_evaluation import run_yu_xiong_evaluation  # noqa: E402


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
        help=(
            "KIMORE exercise; plain_dtw is defined only for Es3, while "
            "yu_xiong_dtw accepts Es1--Es5"
        ),
    )
    parser.add_argument(
        "--method",
        default="plain_dtw",
        choices=["plain_dtw", "yu_xiong_dtw"],
    )
    parser.add_argument(
        "--bootstrap-resamples",
        type=int,
        default=5000,
        help=(
            "Conditional fixed-OOF-prediction subject-bootstrap resamples "
            "(default: 5000)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: ./results/<method>)",
    )
    parser.add_argument(
        "--figure-dir",
        type=Path,
        help="Figure directory (default: ./figures/<method>)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or PROJECT_ROOT / "results" / args.method
    figure_dir = args.figure_dir or PROJECT_ROOT / "figures" / args.method
    evaluator = {
        "plain_dtw": run_cross_validated_evaluation,
        "yu_xiong_dtw": run_yu_xiong_evaluation,
    }[args.method]
    evaluator(
        manifest_path=args.manifest.expanduser().resolve(),
        exercise=args.exercise,
        output_dir=output_dir.expanduser().resolve(),
        figure_dir=figure_dir.expanduser().resolve(),
        bootstrap_resamples=args.bootstrap_resamples,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
