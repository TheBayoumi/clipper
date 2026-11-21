"""CPU-only book catalog, outline and staged execution CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import scripting, selection
from .orchestrator import BookOrchestrator


DEFAULT_BOARD = Path("niches/atomic-habits-approved-storyboard.json")
DEFAULT_CONFIG = Path("niches/book-lessons.yaml")


def configure_books_parser(subparsers: Any) -> None:
    books = subparsers.add_parser("books", help="source-backed book episode stage DAG")
    actions = books.add_subparsers(dest="books_action", required=True)

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
    plan.add_argument("--storyboard-template", type=Path, default=DEFAULT_BOARD)
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
    run.add_argument("--storyboard-template", type=Path, default=DEFAULT_BOARD)
    run.add_argument("--artifact-root", type=Path, default=Path("artifacts/books"))
    run.add_argument(
        "--script-brief",
        type=Path,
        help="recorded, validated script; avoids invoking the pinned Qwen provider",
    )


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def dispatch_books(args: argparse.Namespace) -> int:
    catalog = _json(args.catalog)
    if not isinstance(catalog, dict):
        raise selection.BookSelectionError("unknown editorial book catalog schema")
    selection._validate_catalog(catalog)

    if args.books_action == "validate":
        config = scripting.NicheConfig.load(args.config)
        if config.format != "book_lesson":
            raise scripting.NicheVideoError("book pipeline requires format: book_lesson")
        print(json.dumps({"valid": True, "catalog": str(args.catalog), "config": str(args.config)}))
        return 0

    history = _json(args.history) if args.history else []
    if not isinstance(history, list):
        raise selection.BookSelectionError("history requires a list of selection entries")
    board = _json(args.storyboard_template)
    if not isinstance(board, dict):
        raise selection.BookSelectionError("unsupported 30-60 second production storyboard")
    orchestrator = BookOrchestrator(args.artifact_root)
    kwargs = {
        "book_id": args.book_id,
        "lesson_id": args.lesson_id,
        "angle_id": args.angle_id,
        "pillar": args.pillar,
    }
    if args.books_action == "plan":
        planned = orchestrator.plan(catalog, board, history, **kwargs)
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
        orchestrator.run(
            catalog,
            board,
            history,
            source=args.script_brief,
            **kwargs,
        )
        raise AssertionError("book run unexpectedly completed unported stages")
    raise ValueError("unsupported book subcommand")
