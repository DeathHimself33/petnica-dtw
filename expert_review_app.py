"""Run the local KIMORE blinded expert-review web application."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from kimore_expert_review import create_app  # noqa: E402
from waitress import serve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument(
        "--queue",
        type=Path,
        default=ROOT
        / Path(
            "results/interpretable_dtw/pilot_review/second_review_queue.csv"
        ),
    )
    parser.add_argument(
        "--primary-labels",
        type=Path,
        default=ROOT / "annotations/kimore_es3_pilot_labels.csv",
    )
    parser.add_argument(
        "--sheets",
        type=Path,
        default=ROOT / "results/interpretable_dtw/pilot_review/sheets",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=ROOT
        / Path(
            "results/interpretable_dtw/expert_review/expert_review.sqlite3"
        ),
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--video-manifest", type=Path, help="RGB paths and movement boundaries for validation/reference recordings")
    parser.add_argument("--training-inventory", type=Path, help="Complete sample_id,subject_id CSV used to train the evaluated model")
    parser.add_argument("--legacy-sheets", action="store_true", help="Open the historical image-only pilot, not holdout validation")
    args = parser.parse_args()
    if not args.legacy_sheets and (args.video_manifest is None or args.training_inventory is None):
        parser.error('Video validation requires --video-manifest and --training-inventory; use --legacy-sheets only for the historical pilot')
    if args.legacy_sheets and (args.video_manifest or args.training_inventory):
        parser.error('--legacy-sheets cannot be combined with video validation')
    if args.host not in {'127.0.0.1', 'localhost', '::1'}:
        parser.error('This pilot uses organizer-controlled reviewer IDs, not authentication; bind to localhost only')

    app = create_app(
        queue_path=args.queue,
        primary_labels_path=args.primary_labels,
        sheets_dir=args.sheets,
        database_path=args.database,
        video_manifest_path=args.video_manifest,
        training_inventory_path=args.training_inventory,
    )
    if args.debug:
        app.run(host=args.host, port=args.port, debug=True)
    else:
        print(f"KIMORE Review: http://{args.host}:{args.port}", flush=True)
        serve(app, host=args.host, port=args.port, threads=4)


if __name__ == "__main__":
    main()
