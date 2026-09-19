from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

EXPECTED_ENGINE = "deterministic-gameplay"
EXPECTED_EDITOR = "semantic-editor"
EXPECTED_MODE = "engagement_driven_hardened_source_integrity"
_EPS = 1e-3


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validate_configuration(config: dict[str, Any]) -> None:
    errors: list[str] = []
    if "fallback_policy" in config:
        errors.append("obsolete fallback_policy key must be removed")
    if "semantic_montage" in config.get("semantic_editor", {}):
        errors.append("obsolete semantic_montage configuration must be removed")

    for key in ("count_per_source_max", "minimum_count_per_source"):
        if key in config:
            errors.append(f"static clip-count configuration must be removed: {key}")
    batch = config.get("batch_selection", {})
    for key in (
        "candidate_pool_per_source",
        "maximum_per_source",
        "minimum_total_clips",
        "maximum_total_clips",
    ):
        if key in batch:
            errors.append(f"static batch clip-count configuration must be removed: {key}")

    semantic_editor = config.get("semantic_editor", {})
    minimum_output = float(semantic_editor.get("minimum_output_seconds", 10.0))
    maximum_output = float(semantic_editor.get("maximum_output_seconds", 12.0))
    preferred_output = float(semantic_editor.get("preferred_output_seconds", 11.0))
    if abs(minimum_output - 10.0) > _EPS or abs(maximum_output - 12.0) > _EPS:
        errors.append("gameplay clip duration contract must be exactly 10-12 seconds")
    if not minimum_output - _EPS <= preferred_output <= maximum_output + _EPS:
        errors.append("preferred_output_seconds must remain inside the 10-12 second contract")

    editorial = config.get("editorial", {})
    for key in (
        "finishing_move_allow_semantic_montage_continuation",
        "finishing_move_allow_verified_combat_island_continuation",
        "finishing_move_montage_max_output_seconds",
    ):
        if key in editorial:
            errors.append(f"obsolete editorial key must be removed: {key}")
    first_gap = float(editorial.get("finishing_move_max_continuation_gap_seconds", 0.0))
    body_gap = float(editorial.get("finishing_move_body_hard_cut_max_source_gap_seconds", 0.0))
    if first_gap <= 0.0:
        errors.append("finishing_move_max_continuation_gap_seconds must be positive")
    if body_gap <= 0.0 or body_gap > first_gap + _EPS:
        errors.append(
            "finishing_move_body_hard_cut_max_source_gap_seconds must be positive "
            "and <= first continuation reach"
        )

    if "allow_planned_transition_for_semantic_montage" in config.get("source_integrity", {}):
        errors.append("obsolete semantic-montage source transition key must be removed")
    if "allow_unverified_automatic" in config.get("finishing_move_detector", {}):
        errors.append("automatic Finishing Move acceptance switch must be removed")

    verifier = config.get("combat_state_verifier", {})
    local = verifier.get("local_interaction_verifier", {})
    if verifier.get("enabled") is not True:
        errors.append("combat_state_verifier.enabled must be true")
    if local.get("enabled") is not True:
        errors.append("local_interaction_verifier.enabled must be true")
    continuation = verifier.get("finishing_continuation", {})
    if continuation.get("require_verified_payoff") is not True:
        errors.append("Finishing Move continuation must require verified payoff")
    for key in (
        "minimum_verified_hostile_anchors_without_payoff",
        "minimum_retention_without_payoff",
    ):
        if key in continuation:
            errors.append(f"obsolete no-payoff continuation key must be removed: {key}")

    if errors:
        raise AssertionError(
            "MW4 V3.1 canonical contract configuration violation: " + "; ".join(errors)
        )


def plan_key(plan: dict[str, Any]) -> str:
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


def _source_intervals(plan: dict[str, Any]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for segment in plan.get("segments") or []:
        start = float(segment["start"])
        end = float(segment["end"])
        if end > start:
            intervals.append((start, end))
    return intervals


def _merged_length(intervals: list[tuple[float, float]]) -> float:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def _shared_source_seconds(left: dict[str, Any], right: dict[str, Any]) -> float:
    intersections: list[tuple[float, float]] = []
    for a0, a1 in _source_intervals(left):
        for b0, b1 in _source_intervals(right):
            start, end = max(a0, b0), min(a1, b1)
            if end > start:
                intersections.append((start, end))
    return _merged_length(intersections)


def _semantic_anchor_times(plan: dict[str, Any]) -> list[float]:
    anchors: set[float] = {
        round(float(item), 3) for item in (plan.get("covered_payoff_anchor_times") or [])
    }
    finishing = plan.get("finishing_move")
    if finishing is not None:
        anchors.add(round(float(finishing.get("payoff", finishing["start"])), 3))
    for event in plan.get("effect_events") or []:
        if str(event.get("kind", "")) in {"outcome_like", "impact"} and "time" in event:
            anchors.add(round(float(event["time"]), 3))
    for engagement in plan.get("engagements") or []:
        for event in engagement.get("events") or []:
            kinds = {str(item) for item in (event.get("kinds") or [])}
            if kinds.intersection({"outcome_like", "impact"}) and "time" in event:
                anchors.add(round(float(event["time"]), 3))
    return sorted(anchors)


def _same_finishing_move(left: dict[str, Any], right: dict[str, Any], tolerance: float) -> bool:
    a, b = left.get("finishing_move"), right.get("finishing_move")
    if a is None or b is None:
        return False
    if abs(float(a.get("payoff", a["start"])) - float(b.get("payoff", b["start"]))) <= tolerance:
        return True
    return max(float(a["start"]), float(b["start"])) < min(float(a["end"]), float(b["end"]))


def plans_conflict(left: dict[str, Any], right: dict[str, Any], config: dict[str, Any]) -> bool:
    policy = config.get("duplicate_policy", {})
    anchor_tolerance = float(policy.get("semantic_anchor_tolerance_seconds", 0.35))
    max_context = float(policy.get("max_shared_context_seconds", 2.5))
    substantial_seconds = float(policy.get("substantial_overlap_seconds", 6.0))
    substantial_fraction = float(policy.get("substantial_overlap_fraction_shorter", 0.60))

    if bool(policy.get("finishing_move_exclusive", True)) and _same_finishing_move(
        left, right, anchor_tolerance
    ):
        return True
    if any(
        abs(a - b) <= anchor_tolerance
        for a in _semantic_anchor_times(left)
        for b in _semantic_anchor_times(right)
    ):
        return True

    shared = _shared_source_seconds(left, right)
    if shared <= max_context + 1e-9:
        return False
    shorter = min(_merged_length(_source_intervals(left)), _merged_length(_source_intervals(right)))
    fraction = shared / shorter if shorter > 0 else 0.0
    return shared >= substantial_seconds or fraction >= substantial_fraction


def _ordering_failures(source: str, index: int, plan: dict[str, Any]) -> list[str]:
    segments = list(plan.get("segments") or [])
    if not segments:
        return [f"{source} clip {index}: plan has no source segments"]
    failures: list[str] = []
    start = float(plan.get("start", segments[0]["start"]))
    end = float(plan.get("end", segments[-1]["end"]))
    if abs(start - float(segments[0]["start"])) > _EPS:
        failures.append(f"{source} clip {index}: plan start does not match first segment")
    if abs(end - float(segments[-1]["end"])) > _EPS or end <= start:
        failures.append(f"{source} clip {index}: plan bounds are non-monotonic")
    for pos, segment in enumerate(segments):
        s0 = float(segment["start"])
        s1 = float(segment["end"])
        if s1 <= s0:
            failures.append(f"{source} clip {index}: segment {pos + 1} has invalid bounds")
        if abs(float(segment.get("speed", 1.0)) - 1.0) > 1e-6:
            failures.append(f"{source} clip {index}: segment {pos + 1} is not source-native 1.0x")
        if pos and s0 < float(segments[pos - 1]["end"]) - _EPS:
            failures.append(
                f"{source} clip {index}: source chronology reverses/overlaps at segment {pos + 1}"
            )
    return failures


def _event_has_payoff(event: dict[str, Any]) -> bool:
    return bool(
        {str(item) for item in (event.get("kinds") or [])}.intersection({"outcome_like", "impact"})
    )


def _finishing_plan_failures(
    source: str, index: int, plan: dict[str, Any], config: dict[str, Any]
) -> list[str]:
    finishing = plan.get("finishing_move")
    if finishing is None:
        return []
    failures: list[str] = []
    segments = list(plan.get("segments") or [])
    engagements = list(plan.get("engagements") or [])
    editorial = config["editorial"]

    if (
        str(plan.get("story_type", "")) != "finishing_move_open"
        or str(plan.get("effect_profile", "")) != "finishing_move_hero"
    ):
        failures.append(f"{source} clip {index}: Finishing Move routing is not opening hero")
    if not segments:
        return [*failures, f"{source} clip {index}: Finishing Move has no segments"]

    hero = segments[0]
    if str(hero.get("reason", "")) != "finishing_move_open_hero":
        failures.append(
            f"{source} clip {index}: first segment is not protected Finishing Move hero"
        )
    if not (float(hero["start"]) <= float(finishing["start"]) <= float(hero["end"])):
        failures.append(
            f"{source} clip {index}: verified Finishing Move is not contained in hero segment"
        )
    delay = float(finishing["start"]) - float(hero["start"])
    if delay > 0.48 + _EPS:
        failures.append(f"{source} clip {index}: Finishing Move opens too late ({delay:.3f}s)")

    body = segments[1:]
    if not body:
        failures.append(f"{source} clip {index}: Finishing Move has no verified combat-island body")
        return failures
    if len(body) != len(engagements):
        failures.append(
            f"{source} clip {index}: Finishing Move body/engagement cardinality mismatch"
        )
        return failures

    later_floor = max(float(hero["end"]), float(finishing["end"]) + 0.25)
    max_first_gap = float(editorial["finishing_move_max_continuation_gap_seconds"])
    hard_cut_gap = float(editorial["finishing_move_body_hard_cut_max_source_gap_seconds"])
    if float(body[0]["start"]) < later_floor - _EPS:
        failures.append(
            f"{source} clip {index}: Finishing Move continuation starts before later-content floor"
        )
    if float(body[0]["start"]) - float(hero["end"]) > max_first_gap + _EPS:
        failures.append(f"{source} clip {index}: Finishing Move first continuation exceeds reach")

    for pos, (segment, engagement) in enumerate(zip(body, engagements, strict=False)):
        if str(segment.get("reason", "")) != "verified_combat_island_body":
            failures.append(
                f"{source} clip {index}: body segment {pos + 1} is not a verified combat island"
            )
        if abs(float(segment["start"]) - float(engagement["start"])) > _EPS:
            failures.append(
                f"{source} clip {index}: body segment {pos + 1} does not start "
                "at canonical island boundary"
            )
        expected_end = round(float(engagement["end"]), 3)
        if abs(float(segment["end"]) - expected_end) > _EPS:
            failures.append(
                f"{source} clip {index}: body segment {pos + 1} does not end "
                "at canonical island boundary"
            )
        if not any(_event_has_payoff(event) for event in (engagement.get("events") or [])):
            failures.append(
                f"{source} clip {index}: body island {pos + 1} lacks verified payoff anchor"
            )
        if pos:
            gap = float(segment["start"]) - float(body[pos - 1]["end"])
            if gap < -_EPS:
                failures.append(
                    f"{source} clip {index}: Finishing Move body segments overlap/reverse"
                )
            elif gap > hard_cut_gap + _EPS:
                failures.append(
                    f"{source} clip {index}: Finishing Move body hard-cut source gap "
                    "exceeds contract"
                )
    return failures


def validate_plan(
    source: str,
    index: int,
    plan: dict[str, Any],
    config: dict[str, Any],
) -> list[str]:
    """Validate hard production invariants only.

    Editorial quality is an optimization signal. It does not decide whether verified
    gameplay evidence exists, so quality floors are intentionally not hard contract
    failures for normal candidates.
    """
    validate_configuration(config)
    failures = _ordering_failures(source, index, plan)
    editor = config["semantic_editor"]
    duration = float(plan.get("output_duration", 0.0))
    minimum = float(editor.get("minimum_output_seconds", 10.0))
    maximum = float(editor.get("maximum_output_seconds", 12.0))
    if not (minimum - _EPS <= duration <= maximum + _EPS):
        failures.append(f"{source} clip {index}: duration out of campaign range")

    for field in (
        "opening_quality",
        "ending_quality",
        "retention_quality",
        "payoff_quality",
        "story_coherence",
        "weakest_quarter_interest",
        "low_interest_fraction",
    ):
        value = float(plan.get(field, 0.0))
        if not 0.0 <= value <= 1.0:
            failures.append(f"{source} clip {index}: {field} outside 0..1")

    residual = float(plan.get("max_unexplained_low_interest_run_seconds", 0.0))
    if residual < 0.0:
        failures.append(f"{source} clip {index}: negative unexplained low-interest run")

    failures.extend(_finishing_plan_failures(source, index, plan, config))
    return list(dict.fromkeys(failures))


def _importance_score(plan: dict[str, Any], config: dict[str, Any]) -> float:
    score = float(plan.get("score", 0.0))
    if plan.get("finishing_move") is not None:
        score += float(
            config.get("semantic_editor", {}).get("selection", {}).get("finishing_move_bonus", 0.14)
        )
    return score
