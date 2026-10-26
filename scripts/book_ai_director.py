"""Validate model-authored cinematic shots before any paid rendering.

A Qwen director supplies word-aligned semantic shots; their timestamp boundaries
must come from actual Kokoro audio later. Never render or publish here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUEST_SCHEMA = "clipper-ai-director-request-v1"
PLAN_SCHEMA = "clipper-ai-director-plan-v1"
ROLES = frozenset({"hook", "problem", "insight", "demonstration", "payoff"})
RENDERERS = frozenset({"s2v", "i2v"})


class DirectorContractError(ValueError):
    """Reject ungrounded or incomplete AI-generated scene direction."""


def _digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_request(selection: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    """Bind an approved book claim and exact Qwen narration to a director request."""
    if not isinstance(selection, dict) or selection.get("schema") != "clipper-book-selection-v1":
        raise DirectorContractError("unsupported book-selection schema")
    if selection.get("publish_eligible") is not False:
        raise DirectorContractError("selection crossed publication boundary")
    if selection.get("status") != "SOURCE_BACKED_EDITORIAL_CANDIDATE_NEEDS_SCRIPT":
        raise DirectorContractError("selection lacks the source-backed gate")
    original = {key: value for key, value in selection.items() if key != "fingerprint"}
    if selection.get("fingerprint") != _digest(original):
        raise DirectorContractError("selection fingerprint mismatch")
    book, lesson = selection.get("book"), selection.get("lesson")
    if not isinstance(book, dict) or not isinstance(lesson, dict):
        raise DirectorContractError("selection is missing a book or lesson")
    if (
        selection.get("book_id") != book.get("id")
        or selection.get("lesson_id") != lesson.get("id")
        or selection.get("episode_id")
        != f"{book['id']}--{lesson['id']}--{selection.get('angle_id')}"
    ):
        raise DirectorContractError("selection identity mismatch")
    claim = lesson.get("claim")
    claim_source = lesson.get("claim_source_url")
    if (
        not isinstance(claim, str)
        or not claim.strip()
        or not isinstance(claim_source, str)
        or not claim_source.startswith("https://")
        or selection.get("approved_facts") != [claim]
        or claim_source not in selection.get("sources", [])
    ):
        raise DirectorContractError("selection is missing approved claim evidence")
    if not isinstance(episode, dict) or set(episode) != {"title", "scenes"}:
        raise DirectorContractError("director requires the canonical Qwen episode")
    scenes = episode["scenes"]
    if not isinstance(episode["title"], str) or not episode["title"].strip():
        raise DirectorContractError("script title missing")
    if not isinstance(scenes, list) or not 3 <= len(scenes) <= 5:
        raise DirectorContractError("script requires 3-5 narrative sections")
    sections: list[dict[str, Any]] = []
    offset = 0
    for index, scene in enumerate(scenes):
        if not isinstance(scene, dict) or set(scene) != {"headline", "narration"}:
            raise DirectorContractError("script scene schema mismatch")
        if not all(isinstance(value, str) and value.strip() for value in scene.values()):
            raise DirectorContractError("script scene text missing")
        word_count = len(scene["narration"].split())
        sections.append(
            {
                "index": index,
                "headline": scene["headline"],
                "narration": scene["narration"],
                "start_word": offset,
                "end_word": offset + word_count,
            }
        )
        offset += word_count
    if not 45 <= offset <= 125:
        raise DirectorContractError("script narration outside supported word budget")
    request = {
        "schema": REQUEST_SCHEMA,
        "selection_fingerprint": selection["fingerprint"],
        "book_id": selection["book_id"],
        "lesson_id": selection["lesson_id"],
        "angle_id": selection["angle_id"],
        "episode_id": selection["episode_id"],
        "script_version": selection["script_version"],
        "book_title": book["title"],
        "book_author": book["author"],
        "script_sha256": _digest(episode),
        "narrative_sections": sections,
        "total_words": offset,
        "approved_claims": [{"claim": claim, "source_url": claim_source}],
        "director_instructions": (
            "Direct 3-18 cinematic shots with contiguous 0-based end-exclusive word spans. "
            "Preserve exact narration. Start with a hook, demonstrate a physical idea, "
            "and finish with payoff. Use s2v for presenter, i2v for justified inserts. "
            "No invented claims, subtitles, book-title overlays, or guessed seconds. "
            "Give each shot camera, setting, action, emotion, props and creative QA."
        ),
        "timing": "AWAITING_MEASURED_FULL_AUDIO_WORD_ALIGNMENT",
        "publish_eligible": False,
    }
    request["fingerprint"] = _digest(request)
    return request


def validate_plan(request: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """Fail closed on gaps, source drift, or missing cinematic intent."""
    if not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA:
        raise DirectorContractError("invalid director request schema")
    original = {key: value for key, value in request.items() if key != "fingerprint"}
    if request.get("fingerprint") != _digest(original):
        raise DirectorContractError("director request fingerprint mismatch")
    if request.get("publish_eligible") is not False:
        raise DirectorContractError("director crossed publication boundary")
    if not isinstance(candidate, dict) or set(candidate) != {
        "schema",
        "request_fingerprint",
        "shots",
    }:
        raise DirectorContractError("invalid candidate plan schema")
    if (
        candidate["schema"] != PLAN_SCHEMA
        or candidate["request_fingerprint"] != request["fingerprint"]
    ):
        raise DirectorContractError("candidate belongs to a different request")
    shots = candidate["shots"]
    if not isinstance(shots, list) or not 3 <= len(shots) <= 18:
        raise DirectorContractError("director requires 3-18 meaningful shots")
    sections = request["narrative_sections"]
    if not isinstance(sections, list) or not sections:
        raise DirectorContractError("director request missing script sections")
    words = [word for section in sections for word in section["narration"].split()]
    if len(words) != request["total_words"]:
        raise DirectorContractError("narration word count mismatch")
    required = {
        "id",
        "role",
        "scene_index",
        "start_word",
        "end_word",
        "narration_text",
        "renderer",
        "character_action",
        "camera",
        "setting",
        "emotion",
        "required_objects",
        "source_evidence",
        "qa_criteria",
    }
    allowed_claims = {(item["claim"], item["source_url"]) for item in request["approved_claims"]}
    seen: set[str] = set()
    roles: list[str] = []
    covered = 0
    has_presenter = False
    source_citations = 0
    for shot in shots:
        if not isinstance(shot, dict) or set(shot) != required:
            raise DirectorContractError("shot missing required cinematic direction")
        shot_id = shot["id"]
        if not isinstance(shot_id, str) or not shot_id.strip() or shot_id in seen:
            raise DirectorContractError("duplicate or missing shot ID")
        seen.add(shot_id)
        role = shot["role"]
        if role not in ROLES:
            raise DirectorContractError("unsupported shot role")
        roles.append(role)
        index, start, end = shot["scene_index"], shot["start_word"], shot["end_word"]
        if (
            type(index) is not int
            or not 0 <= index < len(sections)
            or type(start) is not int
            or type(end) is not int
            or start != covered
            or not start < end <= len(words)
            or start < sections[index]["start_word"]
            or end > sections[index]["end_word"]
        ):
            raise DirectorContractError("shot word spans skip, overlap, or cross scenes")
        if shot["narration_text"] != " ".join(words[start:end]):
            raise DirectorContractError("shot narration differs from authoritative script")
        covered = end
        if shot["renderer"] not in RENDERERS:
            raise DirectorContractError("unsupported renderer")
        has_presenter |= shot["renderer"] == "s2v"
        for key in ("character_action", "camera", "setting", "emotion"):
            if not isinstance(shot[key], str) or not shot[key].strip():
                raise DirectorContractError(f"missing {key} direction")
        for key in ("required_objects", "qa_criteria"):
            value = shot[key]
            if not isinstance(value, list) or not all(
                isinstance(item, str) and item.strip() for item in value
            ):
                raise DirectorContractError(f"invalid {key}")
        if not shot["qa_criteria"]:
            raise DirectorContractError("shot requires concrete QA criteria")
        citations = shot["source_evidence"]
        if not isinstance(citations, list):
            raise DirectorContractError("shot source evidence must be a list")
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) != {"claim", "source_url"}:
                raise DirectorContractError("invalid source citation schema")
            if (citation["claim"], citation["source_url"]) not in allowed_claims:
                raise DirectorContractError("director invented source evidence")
            source_citations += 1
    if covered != len(words):
        raise DirectorContractError("director omitted narration ending")
    if roles[0] != "hook" or roles[-1] != "payoff" or "demonstration" not in roles:
        raise DirectorContractError("missing hook, physical demonstration, or payoff")
    if not has_presenter:
        raise DirectorContractError("continuous presenter backbone is required")
    if not source_citations:
        raise DirectorContractError("director omitted approved source evidence")
    plan = {
        "schema": PLAN_SCHEMA,
        "request_fingerprint": request["fingerprint"],
        "selection_fingerprint": request["selection_fingerprint"],
        "episode_id": request["episode_id"],
        "script_version": request["script_version"],
        "script_sha256": request["script_sha256"],
        "shot_count": len(shots),
        "shots": shots,
        "timing": "AWAITING_LOCKED_KOKORO_AUDIO_ALIGNMENT",
        "stage": "AI_DIRECTOR_VALIDATED_NEEDS_AUDIO_TIMING_AND_CREATIVE_REVIEW",
        "creative_review": "PENDING",
        "publish_eligible": False,
        "published": False,
    }
    plan["fingerprint"] = _digest(plan)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--request-output", type=Path, required=True)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--plan-output", type=Path)
    args = parser.parse_args()
    if (args.candidate is None) != (args.plan_output is None):
        parser.error("--candidate and --plan-output must be supplied together")
    request = build_request(
        json.loads(args.selection.read_text(encoding="utf-8")),
        json.loads(args.script.read_text(encoding="utf-8")),
    )
    args.request_output.parent.mkdir(parents=True, exist_ok=True)
    args.request_output.write_text(
        json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.candidate is not None and args.plan_output is not None:
        plan = validate_plan(request, json.loads(args.candidate.read_text(encoding="utf-8")))
        args.plan_output.parent.mkdir(parents=True, exist_ok=True)
        args.plan_output.write_text(
            json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
