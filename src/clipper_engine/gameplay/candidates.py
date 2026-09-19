from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from . import analysis, features

ENGINE = "deterministic-gameplay"
EDITOR = "semantic-editor"
CANDIDATE_MODE = "engagement_driven_hardened_source_integrity"

def _plan_key_from_dict(plan: dict[str, Any]) -> str:
    finishing = plan.get("finishing_move")
    payload = {
        "story_type": str(plan.get("story_type", "")),
        "effect_profile": str(plan.get("effect_profile", "")),
        "segments": [
            [
                round(float(segment["start"]), 3),
                round(float(segment["end"]), 3),
                round(float(segment.get("speed", 1.0)), 3),
                str(segment.get("reason", "")),
            ]
            for segment in (plan.get("segments") or [])
        ],
        "finishing_move": None
        if finishing is None
        else [
            round(float(finishing["start"]), 3),
            round(float(finishing["payoff"]), 3),
            round(float(finishing["end"]), 3),
        ],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def plan_key(plan: analysis.SemanticPlan) -> str:
    return _plan_key_from_dict(asdict(plan))


def _automatic_finishing_candidates(
    timeline: analysis.SemanticTimeline, config: dict[str, Any]
) -> list[dict[str, Any]]:
    cfg = config.get("finishing_move_detector", {})
    if not bool(cfg.get("automatic_discovery_enabled", True)):
        return []
    review_threshold = float(cfg.get("automatic_review_confidence", 0.72))
    heuristic = analysis.core.discover_finishing_moves(
        timeline.base, timeline.shots, timeline.engagements, config
    )
    return [asdict(span) for span in heuristic if float(span.confidence) >= review_threshold]


def _candidate_anchor_time(payload: dict[str, Any]) -> float | None:
    finishing = payload.get("finishing_move")
    if finishing is not None:
        return float(finishing.get("payoff", finishing["start"]))

    marker = "verified payoff anchor "
    for reason in payload.get("editorial_reasons") or []:
        text = str(reason)
        if marker not in text:
            continue
        suffix = text.split(marker, 1)[1]
        token = suffix.split("s", 1)[0]
        try:
            return float(token)
        except ValueError:
            continue

    anchors: list[float] = []
    for engagement in payload.get("engagements") or []:
        for event in engagement.get("events") or []:
            if set(event.get("kinds") or []).intersection({"outcome_like", "impact"}):
                anchors.append(float(event["time"]))
    return anchors[-1] if anchors else None


def _covered_anchor_times(payload: dict[str, Any], kinds: set[str]) -> list[float]:
    anchors: set[float] = set()
    for event in payload.get("effect_events") or []:
        if str(event.get("kind", "")) in kinds and "time" in event:
            anchors.add(round(float(event["time"]), 3))
    for engagement in payload.get("engagements") or []:
        for event in engagement.get("events") or []:
            event_kinds = {str(item) for item in (event.get("kinds") or [])}
            if event_kinds.intersection(kinds) and "time" in event:
                anchors.add(round(float(event["time"]), 3))
    return sorted(anchors)


def _candidate_dict(plan: analysis.SemanticPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["plan_key"] = _plan_key_from_dict(payload)
    anchor_time = plan.proposal_anchor_time
    if anchor_time is None:
        anchor_time = _candidate_anchor_time(payload)
    payload["proposal_anchor_time"] = (
        round(float(anchor_time), 3) if anchor_time is not None else None
    )
    payload["covered_outcome_anchor_times"] = _covered_anchor_times(payload, {"outcome_like"})
    payload["covered_payoff_anchor_times"] = _covered_anchor_times(
        payload, {"outcome_like", "impact"}
    )
    return payload


def _semantic_event_from_dict(payload: dict[str, Any]) -> features.SemanticEvent:
    return features.SemanticEvent(
        time=float(payload["time"]),
        kind=str(payload["kind"]),
        confidence=float(payload["confidence"]),
        evidence={str(key): float(value) for key, value in (payload.get("evidence") or {}).items()},
    )


def _consolidated_event_from_dict(payload: dict[str, Any]) -> analysis.ConsolidatedEvent:
    return analysis.ConsolidatedEvent(
        time=float(payload["time"]),
        kinds=tuple(str(item) for item in (payload.get("kinds") or [])),
        confidence=float(payload["confidence"]),
        evidence={str(key): float(value) for key, value in (payload.get("evidence") or {}).items()},
    )


def _engagement_from_dict(payload: dict[str, Any]) -> analysis.Engagement:
    return analysis.Engagement(
        start=float(payload["start"]),
        end=float(payload["end"]),
        shot_index=int(payload["shot_index"]),
        confidence=float(payload["confidence"]),
        events=tuple(_consolidated_event_from_dict(item) for item in (payload.get("events") or [])),
    )


def _finishing_move_from_dict(payload: dict[str, Any] | None) -> analysis.FinishingMoveSpan | None:
    if payload is None:
        return None
    return analysis.FinishingMoveSpan(
        start=float(payload["start"]),
        payoff=float(payload["payoff"]),
        end=float(payload["end"]),
        shot_index=int(payload["shot_index"]),
        confidence=float(payload["confidence"]),
        evidence={str(key): float(value) for key, value in (payload.get("evidence") or {}).items()},
    )


def _plan_from_dict(payload: dict[str, Any]) -> analysis.SemanticPlan:
    return analysis.SemanticPlan(
        start=float(payload["start"]),
        end=float(payload["end"]),
        raw_duration=float(payload["raw_duration"]),
        output_duration=float(payload["output_duration"]),
        score=float(payload["score"]),
        retention_quality=float(payload["retention_quality"]),
        payoff_quality=float(payload["payoff_quality"]),
        opening_quality=float(payload["opening_quality"]),
        ending_quality=float(payload["ending_quality"]),
        story_coherence=float(payload["story_coherence"]),
        weakest_quarter_interest=float(payload["weakest_quarter_interest"]),
        low_interest_fraction=float(payload["low_interest_fraction"]),
        max_unexplained_low_interest_run_seconds=float(
            payload["max_unexplained_low_interest_run_seconds"]
        ),
        story_type=str(payload["story_type"]),
        effect_profile=str(payload["effect_profile"]),
        segments=tuple(
            analysis.EditSegment(
                start=float(item["start"]),
                end=float(item["end"]),
                speed=float(item.get("speed", 1.0)),
                reason=str(item.get("reason", "")),
            )
            for item in (payload.get("segments") or [])
        ),
        effect_events=tuple(
            _semantic_event_from_dict(item) for item in (payload.get("effect_events") or [])
        ),
        engagements=tuple(
            _engagement_from_dict(item) for item in (payload.get("engagements") or [])
        ),
        finishing_move=_finishing_move_from_dict(payload.get("finishing_move")),
        editorial_reasons=tuple(str(item) for item in (payload.get("editorial_reasons") or [])),
        proposal_anchor_time=(
            float(payload["proposal_anchor_time"])
            if payload.get("proposal_anchor_time") is not None
            else None
        ),
        quality_diagnostics=(
            dict(payload["quality_diagnostics"])
            if payload.get("quality_diagnostics") is not None
            else None
        ),
    )


def _select_from_allocation(
    source_key: str, allocation_path: Path
) -> tuple[list[analysis.SemanticPlan], dict[str, Any]]:
    allocation = json.loads(allocation_path.read_text(encoding="utf-8"))
    source_allocation = allocation.get("source_allocations", {}).get(source_key)
    if not isinstance(source_allocation, dict):
        raise RuntimeError(f"global allocation has no entry for {source_key}")
    wanted = [str(item) for item in source_allocation.get("plan_keys", [])]
    raw_plans = list(source_allocation.get("plans") or [])
    if len(raw_plans) != len(wanted):
        raise RuntimeError(
            f"{source_key}: allocation plan payload count {len(raw_plans)} "
            f"!= key count {len(wanted)}"
        )
    selected: list[analysis.SemanticPlan] = []
    for expected_key, raw_plan in zip(wanted, raw_plans, strict=False):
        actual_key = _plan_key_from_dict(raw_plan)
        if actual_key != expected_key:
            raise RuntimeError(
                f"{source_key}: allocation plan payload hash {actual_key} "
                f"!= approved key {expected_key}"
            )
        selected.append(_plan_from_dict(raw_plan))
    return selected, allocation


def analyze_source_file(
    source_key: str,
    source: Path,
    config: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    excluded = config.get("excluded_windows", {}).get(source_key, [])
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline = analysis.analyze_source(source, config)
    diagnostics = analysis.diagnose_source(timeline, config, excluded, source_key)
    plans = analysis.build_plans_for_source(timeline, config, excluded, source_key)
    automatic_candidates = _automatic_finishing_candidates(timeline, config)
    payload = {
        "schema_version": 1,
        "mode": "analysis",
        "source_key": source_key,
        "semantic_engine": ENGINE,
        "editorial_planner": EDITOR,
        "candidate_mode": CANDIDATE_MODE,
        "diagnostics": diagnostics,
        "candidate_count_after_semantic_gates": len(plans),
        "candidate_pool": [_candidate_dict(plan) for plan in plans],
        "verified_finishing_move_count": len(timeline.finishing_moves),
        "verified_finishing_moves": [asdict(span) for span in timeline.finishing_moves],
        "automatic_finishing_move_candidates": automatic_candidates,
        "automatic_finishing_move_candidates_are_discovery_only": True,
        "failure": None,
    }
    path = output_dir / f"{source_key}_analysis.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "source": source_key,
                "mode": "analysis",
                "candidates": len(plans),
                "verified_finishing_moves": len(timeline.finishing_moves),
                "automatic_finishing_move_candidates": len(automatic_candidates),
            }
        )
    )
    return payload
