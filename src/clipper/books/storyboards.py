"""Catalog-bound book-specific storyboard templates; never substitute another book."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .selection import BookSelectionError

SCHEMA = "clipper-production-storyboard-v2"
REQUIRED_ROLES = (
    "pattern_interrupt_hook",
    "relatable_problem",
    "book_attribution",
    "one_mechanism",
    "physical_before_after",
    "meaningful_payoff",
    "one_step_takeaway",
)


@dataclass(frozen=True, slots=True)
class StoryboardProvenance:
    book_id: str
    template_path: str
    template_sha256: str
    schema: str


def declared_filename(catalog: dict[str, Any], book_id: str) -> str:
    for book in catalog.get("books", []):
        if isinstance(book, dict) and book.get("id") == book_id:
            actual = book.get("storyboard_template")
            expected = f"{book_id}-approved-storyboard.json"
            if not isinstance(actual, str) or actual != expected:
                raise BookSelectionError(
                    f"book {book_id} requires declared book-specific storyboard_template"
                )
            return actual
    raise BookSelectionError(f"no storyboard binding for selected book {book_id}")


def validate_template(board: dict[str, Any], selected: dict[str, Any]) -> None:
    if board.get("schema_version") != "storyboard-production-v1":
        raise BookSelectionError("unsupported production storyboard version")
    book = board.get("book")
    selected_book = selected["book"]
    if (
        not isinstance(book, dict)
        or book.get("title") != selected_book["title"]
        or book.get("author") != selected_book["author"]
        or book.get("factual_source") != selected_book["source_url"]
    ):
        raise BookSelectionError(f"storyboard template book mismatch: {selected['book_id']}")
    spec = board.get("production_storyboard")
    if not isinstance(spec, dict) or spec.get("schema") != SCHEMA:
        raise BookSelectionError("unsupported 30-60 second production storyboard")
    context = spec.get("source_context")
    if not isinstance(context, dict) or context.get("book_id") != selected["book_id"]:
        raise BookSelectionError(f"storyboard source_context book mismatch: {selected['book_id']}")
    delivery = spec.get("delivery")
    if not isinstance(delivery, dict):
        raise BookSelectionError("storyboard delivery contract missing")
    duration = delivery.get("duration_seconds")
    if (
        duration != {"min": 30, "max": 60, "preferred": 45}
        or delivery.get("aspect_ratio") != "9:16"
        or delivery.get("resolution") != [1080, 1920]
        or delivery.get("fps") != 24
        or not isinstance(delivery.get("shot_count"), str)
        or not delivery["shot_count"].strip()
    ):
        raise BookSelectionError("invalid storyboard delivery contract")
    roles = spec.get("editorial_sequence")
    if not isinstance(roles, list) or len(roles) != len(REQUIRED_ROLES):
        raise BookSelectionError("storyboard missing required narrative roles")
    if any(
        not isinstance(role, dict)
        or role.get("role") != name
        or not isinstance(role.get("objective"), str)
        or not role["objective"].strip()
        for role, name in zip(roles, REQUIRED_ROLES, strict=True)
    ):
        raise BookSelectionError("invalid storyboard narrative roles or ordering")
    direction = spec.get("shot_direction_requirements")
    if (
        not isinstance(direction, dict)
        or not isinstance(direction.get("required"), list)
        or not direction["required"]
        or not isinstance(direction.get("renderers"), dict)
        or not direction["renderers"]
    ):
        raise BookSelectionError("storyboard missing shot-direction contract")
    acceptance = board.get("acceptance")
    if (
        not isinstance(acceptance, dict)
        or acceptance.get("video_publish_eligible") is not False
        or acceptance.get("tech_result_is_creative_approval") is not False
        or acceptance.get("requires_exact_mp4_sha256_human_approval") is not True
        or spec.get("publish_eligible") is not False
        or spec.get("status") != "PRODUCTION_STORYBOARD_SPEC_ONLY_AWAITING_DYNAMIC_AI_DIRECTOR"
        or not isinstance(board.get("status"), str)
        or not board["status"].startswith("BLOCKED")
    ):
        raise BookSelectionError("storyboard unapproved or missing human-review blocks")


def resolve_template(
    catalog: dict[str, Any],
    selected: dict[str, Any],
    *,
    catalog_dir: Path,
    override: Path | None = None,
) -> tuple[dict[str, Any], StoryboardProvenance]:
    filename = declared_filename(catalog, selected["book_id"])
    path = catalog_dir / filename if override is None else override
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise BookSelectionError(
            f"approved storyboard missing or unreadable for {selected['book_id']}: {path}"
        ) from exc
    try:
        raw: object = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BookSelectionError(f"invalid storyboard JSON: {path}") from exc
    if not isinstance(raw, dict):
        raise BookSelectionError(f"invalid storyboard object: {path}")
    validate_template(raw, selected)
    return raw, StoryboardProvenance(
        book_id=selected["book_id"],
        template_path=str(path),
        template_sha256=hashlib.sha256(data).hexdigest(),
        schema=SCHEMA,
    )
