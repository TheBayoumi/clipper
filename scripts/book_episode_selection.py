"""Choose source-backed original book lessons before writing a Clipper episode.

Selection is deterministic, keeps audited provenance, avoids recent repetition,
and supports an explicit user override. It does not invent a full-book summary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "clipper-book-selection-v1"
CATALOG_PATH = Path("niches/book-editorial-catalog.json")
RECENT_BOOK_WINDOW = 2
RECENT_LESSON_WINDOW = 6


class BookSelectionError(ValueError):
    """Candidate book/lesson cannot be selected with source-backed evidence."""


def _digest(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _validate_catalog(data: dict[str, Any]) -> list[dict[str, Any]]:
    if data.get("schema") != "clipper-book-editorial-catalog-v1":
        raise BookSelectionError("unknown editorial book catalog schema")
    books = data.get("books")
    if not isinstance(books, list) or not books:
        raise BookSelectionError("empty book library")
    ids: set[str] = set()
    lessons: set[str] = set()
    for book in books:
        if not isinstance(book, dict):
            raise BookSelectionError("invalid book entry")
        fields = ("id", "title", "author", "pillar", "source_url")
        if any(not isinstance(book.get(k), str) or not book[k].strip() for k in fields):
            raise BookSelectionError("unattributed book or missing source")
        if book["id"] in ids or book.get("status") != "source_verified_candidate":
            raise BookSelectionError("duplicate or unverified book")
        ids.add(book["id"])
        if not book["source_url"].startswith("https://"):
            raise BookSelectionError("book source URL must be HTTPS")
        if not isinstance(book.get("lessons"), list) or not book["lessons"]:
            raise BookSelectionError("book has no eligible lesson")
        for idea in book["lessons"]:
            required = ("id", "topic", "claim", "claim_source_url", "visual_action", "hook_angle")
            if any(not isinstance(idea.get(k), str) or not idea[k].strip() for k in required):
                raise BookSelectionError("idea missing evidence, hook, or visual action")
            if not idea["claim_source_url"].startswith("https://"):
                raise BookSelectionError("idea claim source must be HTTPS")
            key = book["id"] + "/" + idea["id"]
            if key in lessons:
                raise BookSelectionError("duplicate lesson")
            lessons.add(key)
    return books


def choose_lesson(
    catalog: dict[str, Any],
    history: list[dict[str, str]] | None = None,
    *,
    book_id: str | None = None,
    lesson_id: str | None = None,
    pillar: str | None = None,
) -> dict[str, Any]:
    """Select a distinct evidence-backed lesson, never synthesize missing facts."""
    books = _validate_catalog(catalog)
    history = [] if history is None else history
    if not isinstance(history, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("book_id"), str)
        or not isinstance(item.get("lesson_id"), str)
        for item in history
    ):
        raise BookSelectionError("history requires book_id and lesson_id entries")
    if lesson_id is not None and book_id is None:
        raise BookSelectionError("lesson override also requires book_id")
    if book_id is not None and not any(book["id"] == book_id for book in books):
        raise BookSelectionError("requested book is not in source-backed catalog")
    candidates: list[tuple[int, int, dict[str, Any], dict[str, Any]]] = []
    recent_books = {item["book_id"] for item in history[-RECENT_BOOK_WINDOW:]}
    recent_lessons = {
        item["book_id"] + "/" + item["lesson_id"] for item in history[-RECENT_LESSON_WINDOW:]
    }
    for i, book in enumerate(books):
        if book_id is not None and book["id"] != book_id:
            continue
        if pillar is not None and book["pillar"] != pillar:
            continue
        for k, lesson in enumerate(book["lessons"]):
            if lesson_id is not None and lesson["id"] != lesson_id:
                continue
            identity = book["id"] + "/" + lesson["id"]
            # Rotate books first, then ideas. Reuse only once catalog is exhausted.
            score = (
                100
                - (65 if book["id"] in recent_books else 0)
                - (85 if identity in recent_lessons else 0)
            )
            candidates.append((score, -(i * 100 + k), book, lesson))
    if not candidates:
        raise BookSelectionError("no verified book/lesson matches selection filters")
    _score, _order, selected, idea = max(candidates, key=lambda item: (item[0], item[1]))
    payload = {
        "schema": SCHEMA,
        "mode": "manual_override" if book_id is not None else "automatic_catalog_rotation",
        "book": {
            "id": selected["id"],
            "title": selected["title"],
            "author": selected["author"],
            "pillar": selected["pillar"],
            "source_url": selected["source_url"],
        },
        "lesson": idea,
        "approved_facts": [idea["claim"]],
        "sources": [selected["source_url"], idea["claim_source_url"]],
        "history_sha256": _digest(history),
        "status": "SOURCE_BACKED_EDITORIAL_CANDIDATE_NEEDS_SCRIPT",
        "min_duration_seconds": 30,
        "max_duration_seconds": 60,
        "publish_eligible": False,
    }
    payload["fingerprint"] = _digest(payload)
    return payload


def to_niche_config(selection: dict[str, Any]) -> dict[str, Any]:
    """Bridge the selected book into the existing cached Qwen3/Kokoro script runtime."""
    if selection.get("schema") != SCHEMA or selection.get("publish_eligible") is not False:
        raise BookSelectionError("cannot use unqualified selection for Qwen narration")
    book = selection["book"]
    idea = selection["lesson"]
    return {
        "format": "book_lesson",
        "niche": "Animated original lessons from books on growth, psychology, and business",
        "audience": "Curious adults looking for practical source-backed lessons",
        "brand": "PAGE TO LIFE",
        "voice": "af_heart",
        "pillar": book["pillar"],
        "book_title": book["title"],
        "book_author": book["author"],
        "source_url": idea["claim_source_url"],
        "topic": idea["topic"],
        "facts": selection["approved_facts"],
    }


def to_storyboard_outline(
    selection: dict[str, Any], production_spec: dict[str, Any]
) -> dict[str, Any]:
    """Deliver narrative roles, not arbitrary durations or invented AI video footage."""
    spec = production_spec.get("production_storyboard")
    if not isinstance(spec, dict) or spec.get("schema") != "clipper-production-storyboard-v2":
        raise BookSelectionError("unsupported 30-60 second production storyboard")
    roles = spec.get("editorial_sequence")
    if not isinstance(roles, list) or not roles:
        raise BookSelectionError("missing narrative roles")
    result = {
        "schema": "clipper-book-story-outline-v1",
        "selection_fingerprint": selection["fingerprint"],
        "book": selection["book"],
        "lesson": selection["lesson"],
        "selected_hook_angle": selection["lesson"]["hook_angle"],
        "selected_concrete_visual_action": selection["lesson"]["visual_action"],
        "source_links": selection["sources"],
        "duration_range_seconds": [30, 60],
        "narrative_roles": [
            {
                "role": role["role"],
                "objective": role["objective"],
                "direction": "AI_DIRECTOR_TO_AUTHOR_MEANINGFUL_ACTION",
            }
            for role in roles
        ],
        "timing": "WAIT_FOR_KOKORO_AUDIO_THEN_DIRECT_CUTS_AT_IDEA_BOUNDARIES",
        "stage": "SOURCED_OUTLINE_REQUIRES_ORIGINAL_SCRIPT_AND_AI_DIRECTION",
        "publish_eligible": False,
    }
    result["fingerprint"] = _digest(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument("--history", type=Path)
    parser.add_argument("--book-id")
    parser.add_argument("--lesson-id")
    parser.add_argument("--pillar")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--niche-config-output", type=Path)
    parser.add_argument("--storyboard-template", type=Path)
    parser.add_argument("--storyboard-output", type=Path)
    args = parser.parse_args()
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    history = json.loads(args.history.read_text(encoding="utf-8")) if args.history else []
    result = choose_lesson(
        catalog,
        history,
        book_id=args.book_id,
        lesson_id=args.lesson_id,
        pillar=args.pillar,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    if args.niche_config_output:
        args.niche_config_output.parent.mkdir(parents=True, exist_ok=True)
        # JSON is valid YAML; pass directly to existing NicheConfig.load().
        args.niche_config_output.write_text(
            json.dumps(to_niche_config(result), indent=2, ensure_ascii=False) + "\n"
        )
    if args.storyboard_output:
        if not args.storyboard_template:
            raise BookSelectionError("storyboard output requires a production template")
        template = json.loads(args.storyboard_template.read_text(encoding="utf-8"))
        outline = to_storyboard_outline(result, template)
        args.storyboard_output.parent.mkdir(parents=True, exist_ok=True)
        args.storyboard_output.write_text(json.dumps(outline, indent=2, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "book_id": result["book"]["id"],
                "lesson_id": result["lesson"]["id"],
                "mode": result["mode"],
                "fingerprint": result["fingerprint"],
            }
        )
    )


if __name__ == "__main__":
    main()
