"""Data, validation, persistence, and agreement helpers for expert review."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sqlite3
from dataclasses import dataclass
from contextlib import contextmanager
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from PIL import Image


EXECUTION_LABELS = ("correct", "error", "uncertain", "ungradable")
ERROR_TYPES = (
    "range_of_motion",
    "direction",
    "timing",
    "asymmetry",
    "compensation",
    "posture",
    "other",
)
SEVERITIES = ("mild", "moderate", "severe")
CONFIDENCES = ("low", "medium", "high")
REVIEWER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{1,63}$")
REVIEW_COLUMNS = (
    "review_status",
    "execution_label",
    "error_type",
    "severity",
    "reviewer_confidence",
    "annotator",
    "review_notes",
)


@dataclass(frozen=True)
class Candidate:
    row: dict[str, str]
    sheet_path: Path

    @property
    def key(self) -> tuple[str, int]:
        return self.row["sample_id"], int(self.row["candidate_rank"])

    @property
    def component_label(self) -> str:
        return self.row["component_name"].replace("_", " ")

    @property
    def interval_label(self) -> str:
        start = float(self.row["window_start_percent"])
        end = float(self.row["window_end_percent"])
        return f"{start:g}–{end:g}% snimka"


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        return list(reader.fieldnames), [dict(row) for row in reader]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidates(queue_path: Path, sheets_dir: Path) -> tuple[list[Candidate], str]:
    fields, rows = _read_csv(queue_path)
    required = {
        "sample_id",
        "candidate_rank",
        "component_name",
        "window_start_percent",
        "window_end_percent",
        *REVIEW_COLUMNS,
    }
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"Review queue is missing columns: {sorted(missing)}")
    if not rows:
        raise ValueError("Review queue is empty")

    candidates: list[Candidate] = []
    seen: set[tuple[str, int]] = set()
    for row in rows:
        key = row["sample_id"], int(row["candidate_rank"])
        if key in seen:
            raise ValueError(f"Duplicate review item: {key}")
        seen.add(key)
        sheet_path = sheets_dir / f"{row['sample_id']}.jpg"
        if not sheet_path.is_file():
            raise FileNotFoundError(f"Missing review sheet: {sheet_path}")
        candidates.append(Candidate(row=row, sheet_path=sheet_path))
    return candidates, file_sha256(queue_path)


def load_primary_labels(
    path: Path,
    candidates: Iterable[Candidate],
) -> dict[tuple[str, int], dict[str, str]]:
    fields, rows = _read_csv(path)
    required = {"sample_id", "candidate_rank", *REVIEW_COLUMNS}
    missing = required.difference(fields)
    if missing:
        raise ValueError(f"Primary labels are missing columns: {sorted(missing)}")
    by_key = {
        (row["sample_id"], int(row["candidate_rank"])): row for row in rows
    }
    missing_keys = [candidate.key for candidate in candidates if candidate.key not in by_key]
    if missing_keys:
        raise ValueError(f"Primary labels missing review items: {missing_keys}")
    return {candidate.key: by_key[candidate.key] for candidate in candidates}


def reviewer_order(
    candidates: Iterable[Candidate], reviewer_id: str, queue_hash: str
) -> list[Candidate]:
    def blind_key(candidate: Candidate) -> str:
        value = f"{queue_hash}\0{reviewer_id}\0{candidate.key[0]}\0{candidate.key[1]}"
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    return sorted(candidates, key=blind_key)


def validate_reviewer_id(value: str) -> str:
    reviewer_id = value.strip()
    if not REVIEWER_PATTERN.fullmatch(reviewer_id):
        raise ValueError(
            "Reviewer ID mora imati 2–64 slova, brojeva, tačaka, crtica ili donjih crta."
        )
    return reviewer_id


def validate_review(form: Mapping[str, str]) -> dict[str, str]:
    label = form.get("execution_label", "").strip()
    error_type = form.get("error_type", "").strip()
    severity = form.get("severity", "").strip()
    confidence = form.get("reviewer_confidence", "").strip()
    notes = form.get("review_notes", "").strip()
    if label not in EXECUTION_LABELS:
        raise ValueError("Izaberite oznaku izvođenja.")
    if confidence not in CONFIDENCES:
        raise ValueError("Izaberite sigurnost procene.")
    if label == "error":
        if error_type not in ERROR_TYPES:
            raise ValueError("Izaberite tip greške.")
        if severity not in SEVERITIES:
            raise ValueError("Izaberite težinu greške.")
    elif error_type or severity:
        raise ValueError("Tip i težina važe samo uz oznaku greške.")
    if label in {"uncertain", "ungradable"} and not notes:
        raise ValueError("Dodajte kratku belešku za nesiguran ili neocenjiv dokaz.")
    return {
        "review_status": (
            "adjudication_needed"
            if label in {"uncertain", "ungradable"}
            else "reviewed"
        ),
        "execution_label": label,
        "error_type": error_type,
        "severity": severity,
        "reviewer_confidence": confidence,
        "review_notes": notes,
    }


def validate_adjudication(form: Mapping[str, str]) -> dict[str, str]:
    result = validate_review(form)
    if result["execution_label"] in {"uncertain", "ungradable"}:
        raise ValueError("Finalna odluka mora biti ispravno ili greška.")
    if not result["review_notes"]:
        raise ValueError("Adjudikacija zahteva kratko obrazloženje.")
    result["review_status"] = "reviewed"
    return result


@contextmanager
def connect_database(path: Path) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_database(path: Path, queue_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect_database(path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reviews (
                sample_id TEXT NOT NULL,
                candidate_rank INTEGER NOT NULL,
                reviewer_id TEXT NOT NULL,
                queue_sha256 TEXT NOT NULL,
                review_status TEXT NOT NULL,
                execution_label TEXT NOT NULL,
                error_type TEXT NOT NULL DEFAULT '',
                severity TEXT NOT NULL DEFAULT '',
                reviewer_confidence TEXT NOT NULL,
                review_notes TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (sample_id, candidate_rank, reviewer_id)
            );
            CREATE TABLE IF NOT EXISTS reviewer_state (
                reviewer_id TEXT PRIMARY KEY,
                queue_sha256 TEXT NOT NULL,
                sealed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS adjudications (
                sample_id TEXT NOT NULL,
                candidate_rank INTEGER NOT NULL,
                reviewer_id TEXT NOT NULL,
                review_status TEXT NOT NULL,
                execution_label TEXT NOT NULL,
                error_type TEXT NOT NULL DEFAULT '',
                severity TEXT NOT NULL DEFAULT '',
                reviewer_confidence TEXT NOT NULL,
                review_notes TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (sample_id, candidate_rank)
            );
            CREATE INDEX IF NOT EXISTS idx_reviews_reviewer_status
            ON reviews(reviewer_id, review_status);
            """
        )
        stored = connection.execute(
            "SELECT value FROM metadata WHERE key = 'queue_sha256'"
        ).fetchone()
        if stored is None:
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES('queue_sha256', ?)",
                (queue_hash,),
            )
        elif stored["value"] != queue_hash:
            row_count = connection.execute("SELECT COUNT(*) AS n FROM reviews").fetchone()[
                "n"
            ]
            if row_count:
                raise RuntimeError(
                    "The review queue changed after reviews were saved. Use a new database."
                )
            connection.execute(
                "UPDATE metadata SET value = ? WHERE key = 'queue_sha256'",
                (queue_hash,),
            )
        connection.execute("PRAGMA optimize")


def load_reviews(path: Path, reviewer_id: str) -> dict[tuple[str, int], dict[str, str]]:
    with connect_database(path) as connection:
        rows = connection.execute(
            "SELECT * FROM reviews WHERE reviewer_id = ?", (reviewer_id,)
        ).fetchall()
    return {
        (row["sample_id"], int(row["candidate_rank"])): dict(row) for row in rows
    }


def is_reviewer_sealed(path: Path, reviewer_id: str) -> bool:
    with connect_database(path) as connection:
        row = connection.execute(
            "SELECT 1 FROM reviewer_state WHERE reviewer_id = ?", (reviewer_id,)
        ).fetchone()
    return row is not None


def save_review(
    path: Path,
    reviewer_id: str,
    candidate: Candidate,
    queue_hash: str,
    review: Mapping[str, str],
) -> None:
    now = datetime.now(UTC).isoformat()
    with connect_database(path) as connection:
        sealed = connection.execute(
            "SELECT 1 FROM reviewer_state WHERE reviewer_id = ?", (reviewer_id,)
        ).fetchone()
        if sealed is not None:
            raise RuntimeError("Ovaj pregled je zaključen i više se ne može menjati.")
        connection.execute(
            """
            INSERT INTO reviews(
                sample_id, candidate_rank, reviewer_id, queue_sha256,
                review_status, execution_label, error_type, severity,
                reviewer_confidence, review_notes, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sample_id, candidate_rank, reviewer_id) DO UPDATE SET
                review_status = excluded.review_status,
                execution_label = excluded.execution_label,
                error_type = excluded.error_type,
                severity = excluded.severity,
                reviewer_confidence = excluded.reviewer_confidence,
                review_notes = excluded.review_notes,
                updated_at = excluded.updated_at
            """,
            (
                candidate.key[0],
                candidate.key[1],
                reviewer_id,
                queue_hash,
                review["review_status"],
                review["execution_label"],
                review["error_type"],
                review["severity"],
                review["reviewer_confidence"],
                review["review_notes"],
                now,
                now,
            ),
        )


def seal_reviewer(path: Path, reviewer_id: str, queue_hash: str, expected: int) -> None:
    with connect_database(path) as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS n FROM reviews WHERE reviewer_id = ?",
            (reviewer_id,),
        ).fetchone()["n"]
        if count != expected:
            raise ValueError(f"Popunite svih {expected} stavki pre zaključavanja.")
        connection.execute(
            """
            INSERT INTO reviewer_state(reviewer_id, queue_sha256, sealed_at)
            VALUES (?, ?, ?)
            ON CONFLICT(reviewer_id) DO NOTHING
            """,
            (reviewer_id, queue_hash, datetime.now(UTC).isoformat()),
        )


def save_adjudication(
    path: Path,
    reviewer_id: str,
    candidate: Candidate,
    review: Mapping[str, str],
) -> None:
    with connect_database(path) as connection:
        connection.execute(
            """
            INSERT INTO adjudications(
                sample_id, candidate_rank, reviewer_id, review_status,
                execution_label, error_type, severity, reviewer_confidence,
                review_notes, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sample_id, candidate_rank) DO UPDATE SET
                reviewer_id = excluded.reviewer_id,
                review_status = excluded.review_status,
                execution_label = excluded.execution_label,
                error_type = excluded.error_type,
                severity = excluded.severity,
                reviewer_confidence = excluded.reviewer_confidence,
                review_notes = excluded.review_notes,
                updated_at = excluded.updated_at
            """,
            (
                candidate.key[0],
                candidate.key[1],
                reviewer_id,
                review["review_status"],
                review["execution_label"],
                review["error_type"],
                review["severity"],
                review["reviewer_confidence"],
                review["review_notes"],
                datetime.now(UTC).isoformat(),
            ),
        )


def load_adjudications(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with connect_database(path) as connection:
        rows = connection.execute("SELECT * FROM adjudications").fetchall()
    return {
        (row["sample_id"], int(row["candidate_rank"])): dict(row) for row in rows
    }


def needs_adjudication(primary: Mapping[str, str], second: Mapping[str, str]) -> bool:
    if primary["execution_label"] != second["execution_label"]:
        return True
    if primary["execution_label"] in {"uncertain", "ungradable"}:
        return True
    if primary["review_status"] == "adjudication_needed":
        return True
    if primary["execution_label"] == "error":
        return (
            primary["error_type"] != second["error_type"]
            or primary["severity"] != second["severity"]
        )
    return False


def agreement_summary(
    candidates: Iterable[Candidate],
    primary: Mapping[tuple[str, int], Mapping[str, str]],
    second: Mapping[tuple[str, int], Mapping[str, str]],
) -> dict[str, object]:
    pairs = [
        (primary[candidate.key]["execution_label"], second[candidate.key]["execution_label"])
        for candidate in candidates
    ]
    total = len(pairs)
    exact = sum(first == other for first, other in pairs)
    labels = list(EXECUTION_LABELS)
    first_counts = {label: sum(first == label for first, _ in pairs) for label in labels}
    second_counts = {label: sum(other == label for _, other in pairs) for label in labels}
    expected = sum(first_counts[label] * second_counts[label] for label in labels) / (
        total * total
    )
    observed = exact / total
    kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)
    decided_pairs = [pair for pair in pairs if set(pair) <= {"correct", "error"}]
    decided_agreement = (
        sum(first == other for first, other in decided_pairs) / len(decided_pairs)
        if decided_pairs
        else None
    )
    confusion = {
        first: {other: sum(pair == (first, other) for pair in pairs) for other in labels}
        for first in labels
    }
    return {
        "items": total,
        "exact_agreement": observed,
        "cohen_kappa": kappa,
        "decided_items": len(decided_pairs),
        "decided_agreement": decided_agreement,
        "confusion": confusion,
    }


def agreement_json_bytes(summary: Mapping[str, object]) -> bytes:
    return json.dumps(summary, indent=2, sort_keys=True).encode("utf-8") + b"\n"


@lru_cache(maxsize=64)
def _cropped_evidence_cached(path_text: str, rank: int, modified_ns: int) -> bytes:
    del modified_ns
    path = Path(path_text)
    with Image.open(path) as source:
        image = source.convert("RGB")
    expected_width = 1916
    expected_height = 1320
    if image.width < expected_width or image.height < expected_height:
        raise ValueError(f"Unexpected review-sheet dimensions: {image.size}")

    content_x = 440
    overview_top = 78
    candidate_top = 410 + (rank - 1) * 178
    overview = image.crop((content_x, overview_top, image.width, 410))
    candidate = image.crop((content_x, candidate_top, image.width, candidate_top + 178))
    combined = Image.new(
        "RGB", (overview.width, overview.height + 16 + candidate.height), "white"
    )
    combined.paste(overview, (0, 0))
    combined.paste(candidate, (0, overview.height + 16))
    output = io.BytesIO()
    combined.save(output, format="JPEG", quality=92, optimize=True)
    return output.getvalue()


def cropped_evidence(candidate: Candidate) -> bytes:
    stat = candidate.sheet_path.stat()
    return _cropped_evidence_cached(
        str(candidate.sheet_path.resolve()), candidate.key[1], stat.st_mtime_ns
    )
