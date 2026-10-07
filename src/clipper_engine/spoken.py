from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import speech
from .gameplay import contract, semantics

_WORD_RE = re.compile(r"[a-z0-9]+")


def _normalize(text: str) -> str:
    return " ".join(_WORD_RE.findall(text.lower()))


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _query_score(text: str, queries: list[str]) -> float:
    normalized = _normalize(text)
    haystack = _tokens(text)
    best = 0.0
    for query in queries:
        query_norm = _normalize(query)
        wanted = _tokens(query)
        if not wanted:
            continue
        if query_norm and query_norm in normalized:
            best = max(best, 1.0)
            continue
        best = max(best, len(haystack & wanted) / len(wanted))
    return best


def _transcript_text(
    segments: tuple[speech.TranscriptSegment, ...], start: float, end: float
) -> str:
    return " ".join(
        item.text.strip()
        for item in segments
        if item.end > start and item.start < end and item.text.strip()
    ).strip()


def _mean_word_probability(
    segments: tuple[speech.TranscriptSegment, ...], start: float, end: float
) -> float:
    values = [
        word.probability
        for item in segments
        if item.end > start and item.start < end
        for word in item.words
        if word.end > start and word.start < end
    ]
    return sum(values) / len(values) if values else 0.75


def _window_variants(
    transcript: speech.Transcript,
    anchor_index: int,
    config: dict[str, Any],
) -> list[tuple[float, float, str, float]]:
    editor = config["semantic_editor"]
    minimum = float(editor.get("minimum_output_seconds", 10.0))
    maximum = float(editor.get("maximum_output_seconds", 12.0))
    preferred = float(editor.get("preferred_output_seconds", 11.0))
    segments = transcript.segments
    variants: list[tuple[float, float, str, float]] = []

    left_floor = max(0, anchor_index - 5)
    right_ceiling = min(len(segments) - 1, anchor_index + 5)
    for left in range(left_floor, anchor_index + 1):
        for right in range(anchor_index, right_ceiling + 1):
            start = float(segments[left].start)
            end = float(segments[right].end)
            duration = end - start
            if duration < minimum - 1e-3 or duration > maximum + 1e-3:
                continue
            text = _transcript_text(segments, start, end)
            if not text:
                continue
            boundary = max(0.0, 1.0 - abs(duration - preferred) / max(1.0, maximum - minimum))
            probability = _mean_word_probability(segments, start, end)
            score = 0.65 * boundary + 0.35 * probability
            variants.append((round(start, 3), round(end, 3), text, score))

    if variants:
        return sorted(variants, key=lambda item: (-item[3], item[0], item[1]))

    anchor = segments[anchor_index]
    center = (float(anchor.start) + float(anchor.end)) / 2.0
    start = max(0.0, center - preferred / 2.0)
    end = min(transcript.duration, start + preferred)
    start = max(0.0, end - preferred)
    text = _transcript_text(segments, start, end)
    return [(round(start, 3), round(end, 3), text, 0.5)]


def _caption(topic: dict[str, Any], config: dict[str, Any]) -> str:
    publishing = dict(config.get("publishing") or {})
    body = str(topic.get("caption") or "").strip()
    disclosure = str(publishing.get("disclosure") or "#Ad").strip()
    account = str(publishing.get("account_tag") or "@callofduty").strip()
    extra = [str(item).strip() for item in publishing.get("hashtags") or [] if str(item).strip()]
    lines = [body, disclosure, account, *extra]
    return "\n".join(line for line in lines if line)


def _content_type(config: dict[str, Any], source_key: str) -> str:
    roles = dict(config.get("spoken_content", {}).get("source_roles") or {})
    content_type = str(roles.get(source_key) or "")
    if not content_type:
        raise RuntimeError(f"spoken-content profile has no source role for {source_key}")
    return content_type


def _topics(config: dict[str, Any], content_type: str) -> list[dict[str, Any]]:
    items = config.get("spoken_content", {}).get("topics", {}).get(content_type) or []
    topics = [dict(item) for item in items if isinstance(item, dict)]
    if not topics:
        raise RuntimeError(f"spoken-content profile has no topics for content_type={content_type}")
    ids = [str(item.get("id") or "") for item in topics]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise RuntimeError(f"spoken-content topics for {content_type} require unique ids")
    return topics


def _anchor_matches(
    transcript: speech.Transcript,
    topic: dict[str, Any],
    config: dict[str, Any],
) -> list[tuple[int, float]]:
    queries = [str(item) for item in topic.get("queries") or [] if str(item).strip()]
    if not queries:
        raise RuntimeError(f"spoken topic {topic.get('id')} has no queries")
    minimum = float(config.get("spoken_content", {}).get("minimum_query_score", 0.60))
    matches: list[tuple[int, float]] = []
    for index, segment in enumerate(transcript.segments):
        direct_score = _query_score(segment.text, queries)
        if direct_score >= minimum:
            matches.append((index, direct_score))
            continue

        left = max(0, index - 1)
        right = min(len(transcript.segments), index + 2)
        context = " ".join(item.text for item in transcript.segments[left:right])
        context_score = _query_score(context, queries)
        if context_score >= minimum and direct_score >= minimum * 0.5:
            matches.append((index, (context_score + direct_score) / 2.0))
    return sorted(matches, key=lambda item: (-item[1], item[0]))


def build_candidate_plans(
    transcript: speech.Transcript,
    source_key: str,
    config: dict[str, Any],
) -> tuple[list[semantics.SemanticPlan], list[dict[str, Any]], dict[str, Any]]:
    content_type = _content_type(config, source_key)
    topics = _topics(config, content_type)
    max_variants = int(config.get("spoken_content", {}).get("variants_per_topic", 4))
    plans: list[semantics.SemanticPlan] = []
    required: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"content_type": content_type, "topics": []}

    for topic in topics:
        topic_id = str(topic["id"])
        anchor_id = f"{content_type}:{topic_id}"
        headline = str(topic.get("headline") or "").strip()
        if not headline:
            raise RuntimeError(f"spoken topic {anchor_id} has no headline")
        matches = _anchor_matches(transcript, topic, config)
        topic_diag: dict[str, Any] = {
            "anchor_id": anchor_id,
            "query_count": len(topic.get("queries") or []),
            "matches": [],
            "candidate_count": 0,
        }
        if not matches:
            topic_diag["disposition"] = "no_transcript_match"
            diagnostics["topics"].append(topic_diag)
            required.append(
                {
                    "id": anchor_id,
                    "content_type": content_type,
                    "topic_id": topic_id,
                    "headline": headline,
                    "anchor_time": None,
                    "disposition": "no_transcript_match",
                }
            )
            continue

        best_anchor_index, _best_query_score = matches[0]
        best_anchor = transcript.segments[best_anchor_index]
        anchor_time = round((best_anchor.start + best_anchor.end) / 2.0, 3)
        required.append(
            {
                "id": anchor_id,
                "content_type": content_type,
                "topic_id": topic_id,
                "headline": headline,
                "anchor_time": anchor_time,
                "disposition": "search_required",
            }
        )

        emitted: set[tuple[float, float]] = set()
        for anchor_index, query_score in matches[:3]:
            for start, end, text, boundary_score in _window_variants(
                transcript, anchor_index, config
            ):
                if len(emitted) >= max_variants:
                    break
                key = (start, end)
                if key in emitted:
                    continue
                emitted.add(key)
                probability = _mean_word_probability(transcript.segments, start, end)
                word_count = max(1, len(_WORD_RE.findall(text)))
                density = min(1.0, word_count / max(1.0, (end - start) * 2.0))
                semantic_score = max(
                    0.0, min(1.0, 0.55 * query_score + 0.25 * boundary_score + 0.20 * probability)
                )
                plan = semantics.SemanticPlan(
                    start=start,
                    end=end,
                    raw_duration=round(end - start, 6),
                    output_duration=round(end - start, 6),
                    score=round(semantic_score, 6),
                    retention_quality=round(max(0.45, 0.65 * density + 0.25), 6),
                    payoff_quality=round(max(0.45, query_score), 6),
                    opening_quality=round(max(0.45, boundary_score), 6),
                    ending_quality=round(max(0.45, boundary_score), 6),
                    story_coherence=round(max(0.60, query_score), 6),
                    weakest_quarter_interest=0.55,
                    low_interest_fraction=0.10,
                    max_unexplained_low_interest_run_seconds=0.0,
                    story_type=f"{content_type}_statement",
                    effect_profile="spoken_content_text_overlay",
                    segments=(semantics.EditSegment(start, end, 1.0, f"spoken_topic:{topic_id}"),),
                    effect_events=(),
                    engagements=(),
                    finishing_move=None,
                    editorial_reasons=(
                        f"semantic_anchor={anchor_id}",
                        f"query_score={query_score:.4f}",
                        f"asr_word_probability={probability:.4f}",
                    ),
                    proposal_anchor_time=anchor_time,
                    quality_diagnostics={
                        "spoken": {
                            "query_score": round(query_score, 6),
                            "boundary_score": round(boundary_score, 6),
                            "asr_word_probability": round(probability, 6),
                        }
                    },
                    content_type=content_type,
                    topic_id=topic_id,
                    headline=headline,
                    caption=_caption(topic, config),
                    transcript_text=text,
                    covered_semantic_anchor_ids=(anchor_id,),
                )
                plans.append(plan)
                topic_diag["matches"].append(
                    {
                        "anchor_time": round(
                            (
                                transcript.segments[anchor_index].start
                                + transcript.segments[anchor_index].end
                            )
                            / 2.0,
                            3,
                        ),
                        "query_score": round(query_score, 6),
                        "candidate_start": start,
                        "candidate_end": end,
                    }
                )
            if len(emitted) >= max_variants:
                break
        topic_diag["candidate_count"] = len(emitted)
        topic_diag["disposition"] = "candidates_emitted" if emitted else "no_legal_window"
        diagnostics["topics"].append(topic_diag)

    plans.sort(
        key=lambda item: (
            item.content_type,
            item.topic_id,
            -item.score,
            item.start,
            item.end,
        )
    )
    return plans, required, diagnostics


def analyze_source_file(
    source_key: str,
    source: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    transcript = speech.transcribe(source, config)
    speech.write_transcript(output_dir / f"{source_key}_transcript.json", transcript)
    plans, required, diagnostics = build_candidate_plans(transcript, source_key, config)
    strategy = dict(config.get("content_strategy") or {})
    payload = {
        "schema_version": 2,
        "mode": "analysis",
        "source_key": source_key,
        "semantic_engine": str(strategy.get("semantic_engine") or "transcript-semantic"),
        "editorial_planner": str(strategy.get("editorial_planner") or "spoken-statement-editor"),
        "candidate_mode": str(strategy.get("candidate_mode") or "spoken_topic_coverage"),
        "content_type": _content_type(config, source_key),
        "transcription": {
            "language": transcript.language,
            "duration": transcript.duration,
            "model": transcript.model,
            "settings": transcript.settings,
            "segment_count": len(transcript.segments),
        },
        "diagnostics": {"spoken_content": diagnostics},
        "required_semantic_anchors": required,
        "candidate_count_after_semantic_gates": len(plans),
        "candidate_pool": [],
        "verified_finishing_move_count": 0,
        "verified_finishing_moves": [],
        "automatic_finishing_move_candidates": [],
        "automatic_finishing_move_candidates_are_discovery_only": True,
        "failure": None,
    }
    for plan in plans:
        item = asdict(plan)
        item["plan_key"] = contract.plan_key(item)
        item["covered_outcome_anchor_times"] = []
        item["covered_payoff_anchor_times"] = []
        payload["candidate_pool"].append(item)

    path = output_dir / f"{source_key}_analysis.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source": source_key,
                "mode": "spoken_content_analysis",
                "content_type": payload["content_type"],
                "required_semantic_anchors": len(required),
                "candidates": len(plans),
            }
        )
    )
    return payload


def validate_configuration(config: dict[str, Any]) -> None:
    capability = config.get("capabilities", {}).get("spoken_content_detection", {})
    if capability.get("enabled") is not True:
        raise RuntimeError("spoken content strategy requires spoken_content_detection capability")
    if not config.get("spoken_content", {}).get("source_roles"):
        raise RuntimeError("spoken content strategy requires source_roles")
    if not config.get("spoken_content", {}).get("topics"):
        raise RuntimeError("spoken content strategy requires topics")
    publishing = dict(config.get("publishing") or {})
    if publishing.get("on_screen_text_required") is not True:
        raise RuntimeError("spoken content campaign requires on-screen text")
    disclosure = str(publishing.get("disclosure") or "")
    if disclosure not in {"#Ad", "#Advertisement", "#Sponsored"}:
        raise RuntimeError("publishing disclosure must be #Ad, #Advertisement, or #Sponsored")
    if not str(publishing.get("account_tag") or "").startswith("@"):
        raise RuntimeError("publishing account_tag is required")


def self_test() -> None:
    sample = "Everything you have can be taken out and lost."
    if _query_score(sample, ["everything persistent", "taken out and lost"]) < 0.99:
        raise AssertionError("spoken query matcher failed exact phrase coverage")
    if _query_score(sample, ["persistent weather"]) >= 0.60:
        raise AssertionError("spoken query matcher accepted unrelated query")
    print("spoken-content semantic matcher self-test: PASS")
