from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import speech, spoken_semantics
from .gameplay import contract, semantics


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


def _mean_word_probability(
    segments: tuple[speech.TranscriptSegment, ...],
    start: float,
    end: float,
) -> float:
    values = [
        word.probability
        for item in segments
        if item.end > start and item.start < end
        for word in item.words
        if word.end > start and word.start < end
    ]
    return sum(values) / len(values) if values else 0.75


def build_candidate_plans(
    transcript: speech.Transcript,
    source_key: str,
    config: dict[str, Any],
    *,
    embedder: spoken_semantics.EmbeddingFn | None = None,
) -> tuple[list[semantics.SemanticPlan], list[dict[str, Any]], dict[str, Any]]:
    content_type = _content_type(config, source_key)
    topics = _topics(config, content_type)
    semantic_cfg = dict(config.get("spoken_content", {}).get("semantic") or {})
    backend = embedder or spoken_semantics.FastEmbedder(
        str(semantic_cfg.get("model") or spoken_semantics.DEFAULT_SEMANTIC_MODEL)
    )
    windows, discovery = spoken_semantics.discover_windows(
        transcript,
        config,
        backend,
    )
    topic_matches, topic_summary = spoken_semantics.topic_matches(
        windows,
        topics,
        backend,
        config,
    )

    plans: list[semantics.SemanticPlan] = []
    for topic in topics:
        topic_id = str(topic["id"])
        for match in topic_matches.get(topic_id, []):
            window = windows[match.window_index]
            probability = _mean_word_probability(
                transcript.segments,
                window.start,
                window.end,
            )
            topic_quality = max(0.0, min(1.0, match.score))
            score = max(
                0.0,
                min(
                    1.0,
                    0.75 * window.score + 0.25 * topic_quality,
                ),
            )
            weakest = min(
                window.opening_quality,
                window.story_quality,
                window.ending_quality,
            )
            anchor_id = f"{content_type}:{topic_id}"
            plans.append(
                semantics.SemanticPlan(
                    start=window.start,
                    end=window.end,
                    raw_duration=round(window.duration, 6),
                    output_duration=round(window.duration, 6),
                    score=round(score, 6),
                    retention_quality=round(
                        (window.ending_quality + window.coherence) / 2.0,
                        6,
                    ),
                    payoff_quality=round(
                        min(
                            1.0,
                            (window.event_similarity + window.ending_quality) / 2.0,
                        ),
                        6,
                    ),
                    opening_quality=window.opening_quality,
                    ending_quality=window.ending_quality,
                    story_coherence=window.story_quality,
                    weakest_quarter_interest=round(weakest, 6),
                    low_interest_fraction=0.0 if weakest >= 0.55 else 0.10,
                    max_unexplained_low_interest_run_seconds=0.0,
                    story_type=f"{content_type}_semantic_moment",
                    effect_profile="spoken_content_text_overlay",
                    segments=(
                        semantics.EditSegment(
                            window.start,
                            window.end,
                            1.0,
                            f"semantic_event:{window.event_label}",
                        ),
                    ),
                    effect_events=(),
                    engagements=(),
                    finishing_move=None,
                    editorial_reasons=(
                        "candidate_origin=embedding_semantic_discovery",
                        "campaign_topic_search=independent",
                        f"semantic_event={window.event_label}",
                        (f"event_similarity={window.event_similarity:.6f}"),
                        f"topic_similarity={match.similarity:.6f}",
                        (f"topic_runner_up_similarity={match.runner_up_similarity:.6f}"),
                        f"topic_margin={match.margin:.6f}",
                        f"asr_word_probability={probability:.4f}",
                        "campaign_keyword_gate=false",
                        ("story_admissibility=opening_story_ending_complete"),
                    ),
                    proposal_anchor_time=window.anchor_time,
                    quality_diagnostics={
                        "spoken": {
                            "event_similarity": window.event_similarity,
                            "topic_similarity": match.similarity,
                            ("topic_runner_up_similarity"): match.runner_up_similarity,
                            "topic_margin": match.margin,
                            "topic_match_score": match.score,
                            "opening_quality": window.opening_quality,
                            "story_quality": window.story_quality,
                            "ending_quality": window.ending_quality,
                            "coherence": window.coherence,
                            "asr_word_probability": round(
                                probability,
                                6,
                            ),
                        }
                    },
                    content_type=content_type,
                    topic_id=topic_id,
                    headline=str(topic["headline"]).strip(),
                    caption=_caption(topic, config),
                    transcript_text=window.text,
                    covered_semantic_anchor_ids=(anchor_id,),
                )
            )

    required: list[dict[str, Any]] = []
    for topic in topics:
        topic_id = str(topic["id"])
        anchor_id = f"{content_type}:{topic_id}"
        matching = [plan for plan in plans if anchor_id in plan.covered_semantic_anchor_ids]
        summary = dict(topic_summary.get(topic_id) or {})
        best_plan = max(
            matching,
            key=lambda plan: plan.score,
            default=None,
        )
        required.append(
            {
                "id": anchor_id,
                "content_type": content_type,
                "topic_id": topic_id,
                "headline": str(topic["headline"]).strip(),
                ("anchor_time"): best_plan.proposal_anchor_time if best_plan else None,
                ("disposition"): "search_required" if best_plan else "no_admissible_candidate",
                ("coverage_mode"): "independent_post_discovery_semantic_search",
                "best_similarity": summary.get("best_similarity"),
                ("best_runner_up_similarity"): summary.get("best_runner_up_similarity"),
                "best_margin": summary.get("best_margin"),
                ("minimum_similarity"): summary.get("minimum_similarity"),
                ("maximum_runner_up_gap"): summary.get("maximum_runner_up_gap"),
                "legal_candidate_count": len(matching),
            }
        )

    diagnostics: dict[str, Any] = {
        "content_type": content_type,
        "discovery": discovery,
        "topic_coverage": topic_summary,
        "candidate_count_after_topic_coverage": len(plans),
        "campaign_queries_used_as_primary_detector": False,
        "independent_topic_search": True,
        "maximum_topics_per_clip": 1,
    }
    plans.sort(
        key=lambda item: (
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
    speech.write_transcript(
        output_dir / f"{source_key}_transcript.json",
        transcript,
    )
    plans, required, diagnostics = build_candidate_plans(
        transcript,
        source_key,
        config,
    )
    strategy = dict(config.get("content_strategy") or {})
    payload = {
        "schema_version": 3,
        "mode": "analysis",
        "source_key": source_key,
        "semantic_engine": str(strategy.get("semantic_engine") or "embedding-semantic"),
        "editorial_planner": str(strategy.get("editorial_planner") or "spoken-context-editor"),
        "candidate_mode": str(strategy.get("candidate_mode") or "semantic-moment-topic-coverage"),
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
                "semantic_engine": payload["semantic_engine"],
            }
        )
    )
    return payload


def validate_configuration(config: dict[str, Any]) -> None:
    capability = config.get(
        "capabilities",
        {},
    ).get("spoken_content_detection", {})
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
    if disclosure not in {
        "#Ad",
        "#Advertisement",
        "#Sponsored",
    }:
        raise RuntimeError("publishing disclosure must be #Ad, #Advertisement, or #Sponsored")
    if not str(publishing.get("account_tag") or "").startswith("@"):
        raise RuntimeError("publishing account_tag is required")


def self_test() -> None:
    if (
        spoken_semantics.cosine(
            [1.0, 0.0],
            [1.0, 0.0],
        )
        < 0.999
    ):
        raise AssertionError("spoken semantic cosine self-test failed identical vectors")
    if abs(spoken_semantics.cosine([1.0, 0.0], [0.0, 1.0])) > 1e-6:
        raise AssertionError("spoken semantic cosine self-test failed orthogonal vectors")
    print("spoken-content semantic discovery self-test: PASS")
