"""CPU-only book catalog, outline and staged execution CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from clipper.stage_contracts import content_fingerprint

from . import acquisition, evaluation, faithful, ingest, mining, scripting, selection, storyboards
from .contracts import SourceUnresolved
from .orchestrator import BookOrchestrator, BookRunError

DEFAULT_CONFIG = Path("niches/book-lessons.yaml")
IMPORT_REPORT_VERSION = "clipper-book-research-import-report-v1"
REJECTION_REPORT_VERSION = "clipper-book-research-import-rejection-v3"


def configure_books_parser(subparsers: Any) -> None:
    books = subparsers.add_parser("books", help="source-backed book episode stage DAG")
    actions = books.add_subparsers(dest="books_action", required=True)

    discover = actions.add_parser(
        "candidates", help="resolve reviewed title/author/edition without network"
    )
    discover.add_argument("--title", required=True)
    discover.add_argument("--author")
    discover.add_argument("--edition")
    discover.add_argument("--registry", type=Path, default=acquisition.REGISTRY)

    acquire = actions.add_parser("acquire", help="bounded anonymous Archive acquisition; no mining")
    acquire.add_argument("--title", required=True)
    acquire.add_argument("--author")
    acquire.add_argument("--edition")
    acquire.add_argument("--registry", type=Path, default=acquisition.REGISTRY)
    acquire.add_argument("--offline", action="store_true")
    acquire.add_argument(
        "--refresh", action="store_true", help="explicitly revalidate remote source"
    )
    acquire.add_argument("--artifact-root", type=Path, default=Path("artifacts/books"))

    ingest_cmd = actions.add_parser(
        "ingest", help="title-only candidate acquire and faithful extract"
    )
    ingest_cmd.add_argument("--title", required=True)
    ingest_cmd.add_argument("--author")
    ingest_cmd.add_argument("--edition")
    ingest_cmd.add_argument("--source", type=Path, help="explicit owner file (legacy mode)")
    ingest_cmd.add_argument("--registry", type=Path, default=acquisition.REGISTRY)
    ingest_cmd.add_argument("--offline", action="store_true")
    ingest_cmd.add_argument(
        "--refresh", action="store_true", help="explicitly revalidate remote source"
    )
    ingest_cmd.add_argument("--overlay", type=Path, help="private reviewed image sidecar")
    ingest_cmd.add_argument("--ocr", choices=("none", "tesseract"), default="none")
    ingest_cmd.add_argument("--artifact-root", type=Path, default=Path("artifacts/books"))

    private_import = actions.add_parser(
        "overlay-import", help="convert local research observations using pinned offline source"
    )
    private_import.add_argument("--title", required=True)
    private_import.add_argument("--author")
    private_import.add_argument("--edition")
    private_import.add_argument("--registry", type=Path, default=acquisition.REGISTRY)
    private_import.add_argument("--artifact-root", type=Path, default=Path("artifacts/books"))
    private_import.add_argument("--research", type=Path, required=True)
    private_import.add_argument("--output", type=Path, required=True)

    mine_cmd = actions.add_parser(
        "mine", help="mine recorded replies or explicitly opt into local CPU Qwen inference"
    )
    mine_cmd.add_argument(
        "--book-id", help="legacy owner-file book id or candidate ref cross-check"
    )
    mine_cmd.add_argument(
        "--candidate-ref",
        type=Path,
        help="exact immutable title-only ingest analysis ref; never discover or fetch",
    )
    mode = mine_cmd.add_mutually_exclusive_group(required=True)
    mode.add_argument("--responses", type=Path, help="offline recorded response JSON")
    mode.add_argument(
        "--local-cpu",
        action="store_true",
        help="explicit local pinned Qwen CPU inference (may download pinned weights)",
    )
    mine_cmd.add_argument(
        "--max-new-units",
        type=int,
        help="local CPU only; maximum uncached model attempts this invocation (default 8)",
    )
    mine_cmd.add_argument("--artifact-root", type=Path, default=Path("artifacts/books"))

    evaluate = actions.add_parser(
        "evaluate", help="private source-bound mined-to-text-shots evaluation; no video"
    )
    evaluate.add_argument("--candidate-ref", type=Path, required=True)
    evaluate.add_argument(
        "--progress", type=Path, help="immutable private whole-book mining progress"
    )
    evaluate.add_argument("--review-file", type=Path, help="private human source-support verdicts")
    evaluate.add_argument(
        "--run-manifest", type=Path,
        help="separate explicit private CPU permissions; omitted means model-free",
    )
    evaluate.add_argument("--responses", type=Path, help="model-free recorded mining responses")
    evaluate.add_argument("--artifact-root", type=Path, default=Path("artifacts/books"))

    validate = actions.add_parser("validate", help="validate catalog and niche configuration")
    validate.add_argument("--catalog", type=Path, default=selection.CATALOG_PATH)
    validate.add_argument("--config", type=Path, default=DEFAULT_CONFIG)

    plan = actions.add_parser("plan", help="select sourced lesson and emit original JSON files")
    plan.add_argument("--catalog", type=Path, default=selection.CATALOG_PATH)
    plan.add_argument("--history", type=Path)
    plan.add_argument("--book-id")
    plan.add_argument("--lesson-id")
    plan.add_argument("--angle-id")
    plan.add_argument("--pillar")
    plan.add_argument("--storyboard-template", type=Path)
    plan.add_argument("--artifact-root", type=Path, default=Path("artifacts/books"))
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--niche-config-output", type=Path)
    plan.add_argument("--storyboard-output", type=Path)

    run = actions.add_parser("run", help="execute book DAG; unported stages always fail closed")
    run.add_argument("--catalog", type=Path, default=selection.CATALOG_PATH)
    run.add_argument("--history", type=Path)
    run.add_argument("--book-id")
    run.add_argument("--lesson-id")
    run.add_argument("--angle-id")
    run.add_argument("--pillar")
    run.add_argument("--storyboard-template", type=Path)
    run.add_argument("--artifact-root", type=Path, default=Path("artifacts/books"))
    run.add_argument(
        "--script-brief",
        type=Path,
        help="recorded, validated script; avoids invoking the pinned Qwen provider",
    )
    run.add_argument(
        "--narration-dir",
        type=Path,
        help="recorded scene-NN.wav files; no live Kokoro inference or model download",
    )


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def dispatch_books(args: argparse.Namespace) -> int:
    if args.books_action == "candidates":
        resolved = acquisition.discover(
            args.title,
            acquisition.load_registry(args.registry),
            author=args.author,
            edition=args.edition,
        )
        print(json.dumps(asdict(resolved), ensure_ascii=False))
        return 1 if isinstance(resolved, SourceUnresolved) else 0

    if args.books_action == "overlay-import":
        runner = BookOrchestrator(args.artifact_root)
        recovered = runner.ingest_candidate(
            args.title,
            author=args.author,
            edition=args.edition,
            registry=acquisition.load_registry(args.registry),
            offline=True,
        )
        if isinstance(recovered, SourceUnresolved):
            print(json.dumps(asdict(recovered), ensure_ascii=False))
            return 1
        research_bytes = faithful.read_evidence(args.research)
        research_digest = hashlib.sha256(research_bytes).hexdigest()
        import_root = args.artifact_root / "book-candidates" / "research-import"
        acquisition._private_directory(
            import_root, private_root=args.artifact_root / "book-candidates"
        )
        raw_research_path = import_root / "source" / f"{research_digest}.json"
        acquisition._private_write(
            raw_research_path,
            research_bytes,
            private_root=args.artifact_root / "book-candidates",
        )
        try:
            overlay_bytes, report = faithful.convert_research_observations(
                recovered.book, research_bytes
            )
        except faithful.ExtractionError as exc:
            rejection = {
                "schema": REJECTION_REPORT_VERSION,
                "research_sha256": research_digest,
                "source_sha256": recovered.book.content_sha256,
                "candidate_fingerprint": recovered.candidate.fingerprint,
                "status": "rejected",
                "reason": str(exc),
            }
            diagnostic_identity = content_fingerprint(rejection)
            rejection["diagnostic_identity"] = diagnostic_identity
            rejection_root = import_root / "rejections"
            acquisition._private_directory(
                rejection_root, private_root=args.artifact_root / "book-candidates"
            )
            rejection_path = (
                rejection_root
                / recovered.candidate.fingerprint
                / f"diagnostic-{diagnostic_identity}.json"
            )
            acquisition._private_write(
                rejection_path,
                (json.dumps(rejection, sort_keys=True, indent=2) + "\n").encode(),
                private_root=args.artifact_root / "book-candidates",
            )
            raise
        report_material = {
            "schema": IMPORT_REPORT_VERSION,
            "reconciliation_schema": report["schema"],
            "parser_version": report["parser_version"],
            "candidate_fingerprint": recovered.candidate.fingerprint,
            "source_sha256": recovered.book.content_sha256,
            "research_sha256": research_digest,
            "overlay_sha256": hashlib.sha256(overlay_bytes).hexdigest(),
        }
        report_identity = content_fingerprint(report_material)
        report["import_report_identity"] = report_identity
        report["import_report_material"] = report_material
        report_path = import_root / recovered.candidate.book_id / f"report-{report_identity}.json"
        acquisition._private_write(
            report_path,
            (json.dumps(report, sort_keys=True, indent=2) + "\n").encode(),
            private_root=args.artifact_root / "book-candidates",
        )
        acquisition._private_write(args.output, overlay_bytes)
        print(
            json.dumps(
                {
                    "overlay_path": str(args.output),
                    "report_path": str(report_path),
                    "unreviewed_assets": report["unreviewed_assets"],
                    "normalized_only_occurrences": report["normalized_only_occurrences"],
                    "linked_occurrences": report["linked_occurrences"],
                }
            )
        )
        return 2 if report["unreviewed_assets"] else 0

    if args.books_action in {"acquire", "ingest"}:
        runner = BookOrchestrator(args.artifact_root)
        if args.books_action == "ingest" and args.source is not None:
            if args.author is None:
                raise BookRunError("explicit owner-file ingest requires --author")
            book = runner.ingest(args.title, args.author, ingest.LocalFileResolver(args.source))
            if isinstance(book, SourceUnresolved):
                print(json.dumps(asdict(book), ensure_ascii=False))
                return 1
            print(
                json.dumps(
                    {
                        "book_id": "-".join(re.findall(r"[a-z0-9]+", args.title.casefold())),
                        "source_fingerprint": book.source_fingerprint,
                        "content_sha256": book.content_sha256,
                        "total_words": book.total_word_count,
                        "authorization_status": book.source.authorization_status,
                    }
                )
            )
            return 0

        reviewed = acquisition.load_registry(args.registry)
        if args.books_action == "acquire":
            acquired = runner.acquire_candidate(
                args.title,
                author=args.author,
                edition=args.edition,
                registry=reviewed,
                offline=args.offline,
                refresh=args.refresh,
            )
            if isinstance(acquired, SourceUnresolved):
                print(json.dumps(asdict(acquired), ensure_ascii=False))
                return 1
            print(
                json.dumps(
                    {
                        "book_id": acquired.candidate.book_id,
                        "record_url": acquired.record_url,
                        "actual_asset_url": acquired.asset_url,
                        "raw_sha256": acquired.observed_sha256,
                        "raw_bytes": acquired.observed_bytes,
                        "http_status": acquired.http_status,
                        "cache_hit": acquired.cache_hit,
                        "authorization_status": acquired.source.authorization_status,
                        "raw_path": str(acquired.raw_path),
                    }
                )
            )
            return 0

        recovered = runner.ingest_candidate(
            args.title,
            author=args.author,
            edition=args.edition,
            registry=reviewed,
            offline=args.offline,
            refresh=args.refresh,
            overlay_path=args.overlay,
            ocr=faithful.cpu_tesseract if args.ocr == "tesseract" else None,
            ocr_identity=faithful.tesseract_identity() if args.ocr == "tesseract" else None,
        )
        if isinstance(recovered, SourceUnresolved):
            print(json.dumps(asdict(recovered), ensure_ascii=False))
            return 1
        dossier = json.loads(recovered.dossier_path.read_text(encoding="utf-8"))
        print(
            json.dumps(
                {
                    "book_id": recovered.candidate.book_id,
                    "raw_sha256": recovered.acquired.observed_sha256,
                    "stage_key": recovered.stage_key,
                    "analysis_ref_path": str(recovered.analysis_ref_path),
                    "recovery_status": recovered.book.recovery_status,
                    "dossier_path": str(recovered.dossier_path),
                    "unique_images": dossier["unique_images"],
                    "image_occurrences": dossier["image_occurrences"],
                    "excluded_pages": dossier["excluded_pages"],
                    "warnings": dossier["warnings"],
                    "authorization_status": recovered.book.source.authorization_status,
                }
            )
        )
        return 2 if recovered.book.recovery_status == "incomplete_recovery" else 0
    if args.books_action == "mine":
        runner = BookOrchestrator(args.artifact_root)
        if args.candidate_ref is None and args.book_id is None:
            raise BookRunError("books mine requires --book-id or --candidate-ref")
        if args.candidate_ref is not None and args.local_cpu:
            result = runner.mine_candidate(
                args.candidate_ref,
                book_id=args.book_id,
                local_cpu=True,
                max_new_units=args.max_new_units,
            )
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] == "complete" else 2
        if args.local_cpu or args.max_new_units is not None:
            raise BookRunError("--local-cpu and --max-new-units require --candidate-ref")
        if args.candidate_ref is not None:
            # Bounded input bytes; recorded replies cannot be an unbounded private dump.
            data = acquisition._read_bounded_file(args.responses, 64_000_001)
            if len(data) > 64_000_000:
                raise mining.BookMiningError("recorded response mapping exceeds bounded input")
            replies: object = json.loads(data)
            if not isinstance(replies, dict) or not all(
                isinstance(k, str) and re.fullmatch(r"[0-9a-f]{64}", k) for k in replies
            ):
                raise mining.BookMiningError("recorded response mapping keys must be SHA-256")
            result = runner.mine_candidate(
                args.candidate_ref,
                responses=replies,
                response_file_sha256=hashlib.sha256(data).hexdigest(),
                book_id=args.book_id,
            )
            # Never print source quotations or private image/research observations.
            print(json.dumps(result, sort_keys=True))
            return 0 if result["status"] == "complete" else 2
        book = runner.load_text(args.book_id)
        data = args.responses.read_bytes()
        replies = json.loads(data)
        if not isinstance(replies, dict):
            raise mining.BookMiningError("recorded mining responses must be a JSON mapping")

        def recorded(request: dict[str, object]) -> str:
            from clipper.stage_contracts import content_fingerprint

            key = content_fingerprint(request)
            answer = replies.get(key)
            if not isinstance(answer, dict):
                raise mining.BookMiningError(f"recorded completion missing for {key}")
            return json.dumps(answer, ensure_ascii=False, sort_keys=True)

        mined = runner.mine(
            book, completion=recorded, response_sha256=hashlib.sha256(data).hexdigest()
        )
        print(
            json.dumps(
                {
                    "book_id": args.book_id,
                    "candidate_count": len(mined["books"][0]["lessons"]),
                    "review_status": mined["review_status"],
                }
            )
        )
        return 0
    if args.books_action == "evaluate":
        permissions = (
            evaluation.EvaluationOptions.from_manifest(
                json.loads(evaluation._read(args.run_manifest, 16_384))
            )
            if args.run_manifest is not None
            else evaluation.EvaluationOptions()
        )
        replies: dict[str, object] | None = None
        response_sha: str | None = None
        if args.responses is not None:
            response_bytes = evaluation._read(args.responses, 64_000_000)
            parsed: object = json.loads(response_bytes)
            if not isinstance(parsed, dict) or any(
                not isinstance(k, str) or re.fullmatch(r"[0-9a-f]{64}", k) is None
                for k in parsed
            ):
                raise evaluation.EvaluationStop("recorded responses require SHA-256 keys")
            replies = parsed
            response_sha = hashlib.sha256(response_bytes).hexdigest()
        runner = BookOrchestrator(args.artifact_root)
        result = runner.evaluate_candidate(
            args.candidate_ref,
            progress_path=args.progress,
            review_file=args.review_file,
            options=permissions,
            recorded_responses=replies,
            response_file_sha256=response_sha,
        )
        # Public stdout never includes private quotes, titles, paths or slug IDs.
        safe = {key: result[key] for key in (
            "schema", "status", "progress_sha256", "review_sha256",
            "candidate_count", "accepted_lessons", "script_scene_count",
            "script_attempts", "direction_attempts", "text_shot_count",
            "narration_included", "measured_duration_seconds", "timing_basis",
            "publish_eligible", "source_authorization", "recovery_status",
            "execution", "model_calls_authorized", "visual_generation_calls"
        ) if key in result}
        print(json.dumps(safe, sort_keys=True))
        return 0 if result["status"] == "PREVISUAL_TEXT_READY_UNREVIEWED" else 2
    catalog = _json(args.catalog)
    if not isinstance(catalog, dict):
        raise selection.BookSelectionError("unknown editorial book catalog schema")
    selection._validate_catalog(catalog)

    if args.books_action == "validate":
        config = scripting.NicheConfig.load(args.config)
        if config.format != "book_lesson":
            raise scripting.NicheVideoError("book pipeline requires format: book_lesson")
        for book in catalog["books"]:
            selected = selection.choose_lesson(catalog, book_id=book["id"])
            storyboards.resolve_template(catalog, selected, catalog_dir=args.catalog.parent)
        print(json.dumps({"valid": True, "catalog": str(args.catalog), "config": str(args.config)}))
        return 0

    history = _json(args.history) if args.history else []
    if not isinstance(history, list):
        raise selection.BookSelectionError("history requires a list of selection entries")
    orchestrator = BookOrchestrator(args.artifact_root)
    kwargs = {
        "book_id": args.book_id,
        "lesson_id": args.lesson_id,
        "angle_id": args.angle_id,
        "pillar": args.pillar,
    }
    if args.books_action == "plan":
        planned = orchestrator.plan(
            catalog,
            history=history,
            catalog_dir=args.catalog.parent,
            storyboard_path=args.storyboard_template,
            **kwargs,
        )
        _write(args.output, planned.selection_json)
        if args.niche_config_output:
            _write(args.niche_config_output, planned.niche_config_json)
        if args.storyboard_output:
            _write(args.storyboard_output, planned.outline_json)
        selected = json.loads(planned.selection_json)
        print(
            json.dumps(
                {
                    "book_id": selected["book"]["id"],
                    "lesson_id": selected["lesson"]["id"],
                    "mode": selected["mode"],
                    "fingerprint": selected["fingerprint"],
                }
            )
        )
        return 0

    if args.books_action == "run":
        if args.narration_dir is None:
            raise BookRunError(
                "books run requires --narration-dir for CPU-only recorded narration; "
                "implicit live Kokoro synthesis is not authorized by this CLI"
            )
        orchestrator.run(
            catalog,
            history=history,
            catalog_dir=args.catalog.parent,
            storyboard_path=args.storyboard_template,
            source=args.script_brief,
            narration_dir=args.narration_dir,
            **kwargs,
        )
        raise AssertionError("book run unexpectedly completed unported stages")
    raise ValueError("unsupported book subcommand")
