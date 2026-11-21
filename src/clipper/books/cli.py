"""CPU-only book catalog, outline and staged execution CLI."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import acquisition, faithful, ingest, mining, scripting, selection, storyboards
from .contracts import SourceUnresolved
from .orchestrator import BookOrchestrator, BookRunError

DEFAULT_CONFIG = Path("niches/book-lessons.yaml")


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
    ingest_cmd.add_argument("--overlay", type=Path, help="private reviewed image sidecar")
    ingest_cmd.add_argument("--ocr", choices=("none", "tesseract"), default="none")
    ingest_cmd.add_argument("--artifact-root", type=Path, default=Path("artifacts/books"))

    mine_cmd = actions.add_parser("mine", help="mine recorded responses, never infer live")
    mine_cmd.add_argument("--book-id", required=True)
    mine_cmd.add_argument("--responses", type=Path, required=True)
    mine_cmd.add_argument("--artifact-root", type=Path, default=Path("artifacts/books"))

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
            overlay_path=args.overlay,
            ocr=faithful.cpu_tesseract if args.ocr == "tesseract" else None,
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
        book = runner.load_text(args.book_id)
        data = args.responses.read_bytes()
        replies: object = json.loads(data)
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
