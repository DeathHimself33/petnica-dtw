"""Local blinded expert-review web application for the KIMORE pilot."""

from __future__ import annotations

import csv
import io
import os
import secrets
import hashlib
import json
import shutil
from dataclasses import replace
from functools import wraps
from pathlib import Path
from typing import Callable, Mapping

from flask import (
    Flask,
    Response,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    send_file,
    url_for,
)

from .model import (
    CONFIDENCES,
    ERROR_TYPES,
    EXECUTION_LABELS,
    REVIEW_COLUMNS,
    SEVERITIES,
    Candidate,
    file_sha256,
    agreement_json_bytes,
    agreement_summary,
    cropped_evidence,
    initialize_database,
    is_reviewer_sealed,
    load_adjudications,
    load_candidates,
    load_primary_labels,
    load_reviews,
    needs_adjudication,
    reviewer_order,
    save_adjudication,
    save_review,
    seal_reviewer,
    validate_adjudication,
    validate_review,
    validate_reviewer_id,
)


def _load_or_create_secret(database_path: Path) -> str:
    configured = os.environ.get("KIMORE_REVIEW_SECRET")
    if configured:
        return configured
    path = database_path.parent / ".review-secret"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        return path.read_text(encoding="utf-8").strip()
    value = secrets.token_hex(32)
    path.write_text(value, encoding="utf-8")
    return value


def _csv_response(rows: list[dict[str, object]], filename: str) -> Response:
    output = io.StringIO(newline="")
    fields = list(rows[0]) if rows else []
    writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def create_app(
    *,
    queue_path: Path,
    primary_labels_path: Path,
    sheets_dir: Path,
    database_path: Path,
    testing: bool = False,
    secret_key: str | None = None,
    video_manifest_path: Path | None = None,
    training_inventory_path: Path | None = None,
) -> Flask:
    queue_path = Path(queue_path).resolve()
    primary_labels_path = Path(primary_labels_path).resolve()
    sheets_dir = Path(sheets_dir).resolve()
    database_path = Path(database_path).resolve()
    candidates, queue_hash = load_candidates(queue_path, sheets_dir)
    videos, video_audit = {}, None
    if (video_manifest_path is None) != (training_inventory_path is None):
        raise ValueError("Supply both video manifest and training inventory")
    if video_manifest_path is not None:
        from .video import load_video_round
        candidates, videos, video_audit = load_video_round(video_manifest_path, training_inventory_path, candidates)
    primary = load_primary_labels(primary_labels_path, candidates)
    package = {
        'video_round': video_audit,
        'queue_sha256': queue_hash,
        'primary_sha256': file_sha256(primary_labels_path),
        'evidence': {c.sheet_path.name: c.evidence_sha256 for c in candidates},
        'protocol': 'paired-review-v2; uncertain unresolved; ungradable excluded from binary metrics',
        'annotation_guide_sha256': file_sha256(Path(__file__).resolve().parents[2] / 'ANNOTATION_GUIDE.md'),
        'candidate_identity_code_sha256': file_sha256(Path(__file__).resolve().parents[1] / 'kimore_candidate_identity.py'),
        'application': {str(p.relative_to(Path(__file__).parent)): file_sha256(p)
                        for p in sorted(Path(__file__).parent.rglob('*'))
                        if p.is_file() and p.suffix in {'.py', '.html', '.js', '.css'}},
    }
    round_id = hashlib.sha256(json.dumps(package, sort_keys=True).encode()).hexdigest()
    initialize_database(database_path, queue_hash, package)
    frozen = database_path.parent / (database_path.name + '.round')
    frozen.mkdir(exist_ok=True)
    for source, name, expected in [(queue_path, 'queue.csv', queue_hash),
                                   (primary_labels_path, 'primary.csv', package['primary_sha256']),
                                   *[(c.sheet_path, c.sheet_path.name, c.evidence_sha256) for c in candidates]]:
        destination = frozen / name
        if not destination.exists():
            shutil.copyfile(source, destination)
        if file_sha256(destination) != expected:
            raise RuntimeError('Frozen round package is damaged')
    (frozen / 'manifest.json').write_text(json.dumps(package, indent=2, sort_keys=True), encoding='utf-8')
    candidates = [replace(c, sheet_path=frozen / c.sheet_path.name) for c in candidates]
    for video in videos.values():
        destination = frozen / (video['sha256'] + video['path'].suffix.lower())
        if not destination.exists():
            shutil.copyfile(video['path'], destination)
        if file_sha256(destination) != video['sha256']:
            raise RuntimeError('Frozen video is damaged')
        video['path'] = destination

    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=secret_key or _load_or_create_secret(database_path),
        TESTING=testing,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Strict",
        MAX_CONTENT_LENGTH=64 * 1024,
    )
    app.extensions["review_candidates"] = candidates
    app.extensions["review_primary"] = primary
    app.extensions["review_queue_hash"] = queue_hash
    app.extensions["review_database_path"] = database_path
    app.extensions["review_round_id"] = round_id

    def csrf_token() -> str:
        token = session.get("csrf_token")
        if token is None:
            token = secrets.token_urlsafe(32)
            session["csrf_token"] = token
        return token

    def require_csrf() -> None:
        supplied = request.form.get("csrf_token", "")
        expected = session.get("csrf_token", "")
        if not expected or not secrets.compare_digest(supplied, expected):
            abort(400, "Invalid form token")

    def reviewer_required(view: Callable[..., Response | str]):
        @wraps(view)
        def wrapped(*args: object, **kwargs: object):
            if "reviewer_id" not in session:
                return redirect(url_for("index"))
            return view(*args, **kwargs)

        return wrapped

    def current_reviewer() -> str:
        reviewer_id = session.get("reviewer_id")
        if not isinstance(reviewer_id, str):
            raise RuntimeError("Reviewer session is missing")
        return reviewer_id

    def ordered_candidates() -> list[Candidate]:
        return reviewer_order(candidates, current_reviewer(), queue_hash)

    def progress_data() -> dict[str, object]:
        reviewer_id = current_reviewer()
        reviews = load_reviews(database_path, reviewer_id)
        return {
            "reviewer_id": reviewer_id,
            "reviewed": len(reviews),
            "total": len(candidates),
            "remaining": len(candidates) - len(reviews),
            "percent": round(100 * len(reviews) / len(candidates)),
            "sealed": is_reviewer_sealed(database_path, reviewer_id),
        }

    def sealed_data() -> tuple[str, dict[tuple[str, int], dict[str, str]]]:
        reviewer_id = current_reviewer()
        if not is_reviewer_sealed(database_path, reviewer_id):
            abort(403, "Finish and seal the blinded review first.")
        reviews = load_reviews(database_path, reviewer_id)
        if len(reviews) != len(candidates):
            abort(409, "The sealed review is incomplete.")
        return reviewer_id, reviews

    def disagreement_candidates(
        reviews: Mapping[tuple[str, int], Mapping[str, str]],
    ) -> list[Candidate]:
        return [
            candidate
            for candidate in ordered_candidates()
            if needs_adjudication(primary[candidate.key], reviews[candidate.key])
        ]

    @app.context_processor
    def inject_context() -> dict[str, object]:
        active_reviewer = session.get("reviewer_id")
        return {
            "csrf_token": csrf_token,
            "execution_labels": EXECUTION_LABELS,
            "error_types": ERROR_TYPES,
            "severities": SEVERITIES,
            "confidences": CONFIDENCES,
            "active_reviewer": active_reviewer,
            "video_mode": bool(videos),
            "nav_sealed": (
                is_reviewer_sealed(database_path, active_reviewer)
                if isinstance(active_reviewer, str)
                else False
            ),
            "label_names": {
                "correct": "Ispravno",
                "error": "Vidljiva greška",
                "uncertain": "Nesigurno",
                "ungradable": "Nije moguće oceniti",
            },
            "error_type_names": {
                "range_of_motion": "Opseg pokreta",
                "direction": "Smer",
                "timing": "Tajming",
                "asymmetry": "Asimetrija",
                "compensation": "Kompenzacija",
                "posture": "Položaj tela",
                "other": "Drugo",
            },
            "severity_names": {
                "mild": "Blaga",
                "moderate": "Umerena",
                "severe": "Teška",
            },
            "confidence_names": {
                "low": "Niska",
                "medium": "Srednja",
                "high": "Visoka",
            },
        }

    @app.after_request
    def add_security_headers(response: Response) -> Response:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; form-action 'self'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    def index() -> str:
        primary_annotators = sorted(
            {row.get("annotator", "") for row in primary.values() if row.get("annotator")}
        )
        data = progress_data() if session.get("reviewer_id") else None
        return render_template(
            "index.html",
            progress=data,
            primary_annotators=primary_annotators,
            queue_items=len(candidates),
        )

    @app.post("/session")
    def start_session() -> Response:
        require_csrf()
        try:
            reviewer_id = validate_reviewer_id(request.form.get("reviewer_id", ""))
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("index"))
        primary_annotators = {row.get("annotator", "") for row in primary.values()}
        if reviewer_id in primary_annotators:
            flash("Use an ID different from the preliminary reviewer.", "error")
            return redirect(url_for("index"))
        session.clear()
        session["reviewer_id"] = reviewer_id
        session["csrf_token"] = secrets.token_urlsafe(32)
        return redirect(url_for("review_next"))

    @app.post("/session/end")
    @reviewer_required
    def end_session() -> Response:
        require_csrf()
        session.clear()
        return redirect(url_for("index"))

    @app.get("/review")
    @reviewer_required
    def review_next() -> Response:
        reviews = load_reviews(database_path, current_reviewer())
        order = ordered_candidates()
        for position, candidate in enumerate(order, start=1):
            if candidate.key not in reviews:
                return redirect(url_for("review_item", position=position))
        return redirect(url_for("progress"))

    @app.route("/review/<int:position>", methods=["GET", "POST"])
    @reviewer_required
    def review_item(position: int) -> Response | str:
        order = ordered_candidates()
        if position < 1 or position > len(order):
            abort(404)
        reviewer_id = current_reviewer()
        candidate = order[position - 1]
        if videos and session.get('reference_seen') != candidate.row['reference_sample_id']:
            return redirect(url_for('reference_intro', position=position))
        sealed = is_reviewer_sealed(database_path, reviewer_id)
        if request.method == "POST":
            require_csrf()
            if sealed:
                abort(409, "This review has already been finalized.")
            try:
                review = validate_review(request.form)
                save_review(
                    database_path, reviewer_id, candidate, queue_hash, review
                )
            except (RuntimeError, ValueError) as error:
                flash(str(error), "error")
            else:
                flash("Oznaka je sačuvana.", "success")
                return redirect(url_for("review_next"))
        reviews = load_reviews(database_path, reviewer_id)
        return render_template(
            "review.html",
            candidate=candidate,
            position=position,
            total=len(order),
            saved=reviews.get(candidate.key),
            progress=progress_data(),
            sealed=sealed,
            sample_video=videos.get(candidate.row['sample_id']),
            reference_video=videos.get(candidate.row.get('reference_sample_id')),
        )

    @app.route('/reference/<int:position>', methods=['GET', 'POST'])
    @reviewer_required
    def reference_intro(position):
        order = ordered_candidates()
        if not videos or not 1 <= position <= len(order):
            abort(404)
        candidate = order[position - 1]
        if request.method == 'POST':
            require_csrf()
            if request.form.get('understood') != 'yes':
                abort(400, 'Potvrdite da ste pogledali referencu i razumeli zadatak.')
            session['reference_seen'] = candidate.row['reference_sample_id']
            return redirect(url_for('review_item', position=position))
        return render_template('reference.html', candidate=candidate, position=position,
                               video=videos[candidate.row['reference_sample_id']])

    @app.get('/video/<int:position>/<kind>')
    @reviewer_required
    def video_evidence(position, kind):
        order = ordered_candidates()
        if not videos or not 1 <= position <= len(order) or kind not in {'sample', 'reference'}:
            abort(404)
        candidate = order[position - 1]
        if kind == 'sample' and session.get('reference_seen') != candidate.row['reference_sample_id']:
            abort(403)
        sid = candidate.row['sample_id' if kind == 'sample' else 'reference_sample_id']
        return send_file(videos[sid]['path'], conditional=True)

    @app.get("/evidence/<int:position>.jpg")
    @reviewer_required
    def evidence(position: int) -> Response:
        order = ordered_candidates()
        if position < 1 or position > len(order):
            abort(404)
        return Response(cropped_evidence(order[position - 1]), mimetype="image/jpeg")

    @app.get("/progress")
    @reviewer_required
    def progress() -> str:
        reviews = load_reviews(database_path, current_reviewer())
        items = [
            {
                "position": position,
                "candidate": candidate,
                "review": reviews.get(candidate.key),
            }
            for position, candidate in enumerate(ordered_candidates(), start=1)
        ]
        return render_template(
            "progress.html", progress=progress_data(), items=items
        )

    @app.get("/api/progress")
    def progress_api() -> Response:
        if "reviewer_id" not in session:
            return jsonify({"error": "reviewer_session_required"}), 401
        return jsonify(progress_data())

    @app.post("/finish")
    @reviewer_required
    def finish() -> Response:
        require_csrf()
        try:
            seal_reviewer(
                database_path, current_reviewer(), queue_hash, len(candidates)
            )
        except ValueError as error:
            flash(str(error), "error")
            return redirect(url_for("progress"))
        flash(
            "Nezavisni pregled je zaključan. Primarne oznake su sada dostupne za poređenje.",
            "success",
        )
        return redirect(url_for("agreement"))

    @app.get("/agreement")
    @reviewer_required
    def agreement() -> str:
        reviewer_id, reviews = sealed_data()
        summary = agreement_summary(candidates, primary, reviews)
        disagreements = disagreement_candidates(reviews)
        adjudications = load_adjudications(database_path, current_reviewer())
        return render_template(
            "agreement.html",
            reviewer_id=reviewer_id,
            summary=summary,
            disagreements=disagreements,
            adjudicated=sum(candidate.key in adjudications and adjudications[candidate.key]['execution_label'] != 'uncertain' for candidate in disagreements),
        )

    @app.route("/adjudication", methods=["GET", "POST"])
    @reviewer_required
    def adjudication_start() -> Response | str:
        _, reviews = sealed_data()
        disagreements = disagreement_candidates(reviews)
        if request.method == "POST":
            require_csrf()
            try:
                adjudicator_id = validate_reviewer_id(
                    request.form.get("adjudicator_id", "")
                )
            except ValueError as error:
                flash(str(error), "error")
                return redirect(url_for("adjudication_start"))
            if adjudicator_id in {current_reviewer(), *[row.get('annotator', '') for row in primary.values()]}:
                flash("Adjudicator must use a different ID from the second reviewer.", "error")
                return redirect(url_for("adjudication_start"))
            session["adjudicator_id"] = adjudicator_id
            return redirect(url_for("adjudication_item", position=1))
        adjudications = load_adjudications(database_path, current_reviewer())
        return render_template(
            "adjudication_start.html",
            disagreements=disagreements,
            adjudicated=sum(candidate.key in adjudications and adjudications[candidate.key]['execution_label'] != 'uncertain' for candidate in disagreements),
            adjudicator_id=session.get("adjudicator_id", ""),
        )

    @app.route("/adjudication/<int:position>", methods=["GET", "POST"])
    @reviewer_required
    def adjudication_item(position: int) -> Response | str:
        _, reviews = sealed_data()
        adjudicator_id = session.get("adjudicator_id")
        if not isinstance(adjudicator_id, str):
            return redirect(url_for("adjudication_start"))
        disagreements = disagreement_candidates(reviews)
        if position < 1 or position > len(disagreements):
            abort(404)
        candidate = disagreements[position - 1]
        if request.method == "POST":
            require_csrf()
            try:
                result = validate_adjudication(request.form)
                save_adjudication(
                    database_path, adjudicator_id, candidate, result, current_reviewer()
                )
            except ValueError as error:
                flash(str(error), "error")
            else:
                flash("Adjudikacija je sačuvana.", "success")
                next_position = position + 1
                if next_position <= len(disagreements):
                    return redirect(
                        url_for("adjudication_item", position=next_position)
                    )
                return redirect(url_for("agreement"))
        adjudications = load_adjudications(database_path, current_reviewer())
        order = ordered_candidates()
        evidence_position = order.index(candidate) + 1
        return render_template(
            "adjudication.html",
            candidate=candidate,
            position=position,
            total=len(disagreements),
            evidence_position=evidence_position,
            primary=primary[candidate.key],
            second=reviews[candidate.key],
            saved=adjudications.get(candidate.key),
            adjudicator_id=adjudicator_id,
        )

    @app.get("/export/second-review.csv")
    @reviewer_required
    def export_second_review() -> Response:
        reviewer_id, reviews = sealed_data()
        rows: list[dict[str, object]] = []
        for candidate in candidates:
            row: dict[str, object] = dict(candidate.row)
            row['round_id'] = round_id
            review = reviews[candidate.key]
            for field in REVIEW_COLUMNS:
                row[field] = reviewer_id if field == "annotator" else review[field]
            rows.append(row)
        return _csv_response(rows, "second_review_completed.csv")

    @app.get("/export/agreement.json")
    @reviewer_required
    def export_agreement() -> Response:
        _, reviews = sealed_data()
        body = agreement_json_bytes({**agreement_summary(candidates, primary, reviews), 'round_id': round_id, 'binary_coverage_definition': 'decided_items / items; uncertain and ungradable excluded'})
        return Response(
            body,
            mimetype="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="review_agreement.json"'
            },
        )

    @app.get("/export/adjudicated.csv")
    @reviewer_required
    def export_adjudicated() -> Response:
        reviewer_id, reviews = sealed_data()
        adjudications = load_adjudications(database_path, current_reviewer())
        rows: list[dict[str, object]] = []
        for candidate in candidates:
            first = primary[candidate.key]
            second = reviews[candidate.key]
            requires = needs_adjudication(first, second)
            final = adjudications.get(candidate.key)
            if not requires:
                final = dict(second)
                final["reviewer_id"] = "reviewer_consensus"
            row: dict[str, object] = dict(candidate.row)
            row['round_id'] = round_id
            for field in REVIEW_COLUMNS:
                row[f"primary_{field}"] = first.get(field, "")
                row[f"second_{field}"] = (
                    reviewer_id if field == "annotator" else second.get(field, "")
                )
                if final is None:
                    row[f"final_{field}"] = ""
                elif field == "annotator":
                    row[f"final_{field}"] = final.get("reviewer_id", "")
                else:
                    row[f"final_{field}"] = final.get(field, "")
            row["adjudication_required"] = str(requires).lower()
            row['decision_id'] = final.get('decision_id', '') if final else ''
            row['input_sha256'] = final.get('input_sha256', '') if final else ''
            row['binary_metric_eligible'] = str(final is not None and final['execution_label'] in {'correct', 'error'}).lower()
            row["adjudication_complete"] = str(final is not None and final['execution_label'] != 'uncertain').lower()
            rows.append(row)
        return _csv_response(rows, "adjudicated_review.csv")

    return app


__all__ = ["create_app"]
