"""Source-verified cinematic moment proposals owned by the Clipper engine.

Unlike first-person hostile evidence, third-person campaign action is verified by
source-grounded editorial annotations. No HUD inference or invented kill outcome is
used. Every verified anchor receives independent source-window search and a
machine-readable disposition before coverage/conflict allocation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from . import anchor_search, quality
from .semantics import EditSegment, SemanticPlan, SemanticTimeline


def _moments(config: dict[str, Any], source_key: str) -> list[dict[str, Any]]:
    settings = config.get("cinematic_moments") or {}
    capability = (config.get("capabilities") or {}).get("cinematic_event_detection") or {}
    if not settings.get("enabled", False) or not capability.get("enabled", False):
        return []
    moments = list((settings.get("verified_anchors") or {}).get(source_key) or [])
    ids: set[str] = set()
    for moment in moments:
        anchor_id = str(moment.get("id") or "")
        if not anchor_id or anchor_id in ids:
            raise ValueError(f"{source_key}: cinematic anchors require unique nonempty ids")
        ids.add(anchor_id)
        if not str(moment.get("verification") or "").strip():
            raise ValueError(f"{source_key}: {anchor_id} lacks source verification evidence")
        if not np.isfinite(float(moment["time"])):
            raise ValueError(f"{source_key}: invalid cinematic anchor time")
    return moments


def _plan(
    timeline: SemanticTimeline,
    source_key: str,
    moment: dict[str, Any],
    start: float,
    end: float,
    config: dict[str, Any],
) -> SemanticPlan:
    fps = float(timeline.fps)
    interest = np.asarray(timeline.signals["interest"], dtype=np.float32)

    def mean(left: float, right: float) -> float:
        a = max(0, min(len(interest), int(left * fps)))
        b = max(a, min(len(interest), int(np.ceil(right * fps))))
        return float(np.mean(interest[a:b])) if b > a else 0.0

    anchor = float(moment["time"])
    opening = mean(start, min(end, start + 1.5))
    ending = mean(max(start, end - 1.0), end)
    retention = mean(start, end)
    payoff = mean(max(start, anchor - 0.5), min(end, anchor + 0.5))
    scores = [
        mean(start + i * (end - start) / 4, start + (i + 1) * (end - start) / 4) for i in range(4)
    ]
    weakest = min(scores)
    coherence = max(0.0, 1.0 - (max(scores) - weakest))
    low_threshold = float(config["semantic_editor"]["dull"].get("low_interest_threshold", 0.30))
    i0 = max(0, min(len(interest), int(start * fps)))
    i1 = max(i0, min(len(interest), int(np.ceil(end * fps))))
    low_fraction = float(np.mean(interest[i0:i1] < low_threshold)) if i1 > i0 else 1.0
    # Human-verified story evidence establishes admissibility, not an invented combat score.
    score = float(np.clip(0.22 * opening + 0.24 * ending + 0.28 * retention + 0.26 * payoff, 0, 1))
    segment = EditSegment(round(start, 3), round(end, 3), 1.0, "verified_cinematic_story")
    anchor_id = f"cinematic:{source_key}:{moment['id']}"
    return SemanticPlan(
        segment.start,
        segment.end,
        round(end - start, 3),
        round(end - start, 3),
        round(score, 5),
        round(retention, 4),
        round(payoff, 4),
        round(opening, 4),
        round(ending, 4),
        round(coherence, 4),
        round(weakest, 4),
        round(low_fraction, 4),
        0.0,
        "cinematic_moment",
        "source_native_full_frame",
        (segment,),
        (),
        (),
        None,
        (
            "source-grounded verified cinematic event; not a hostile kill classification",
            f"independent cinematic anchor search at {anchor:.3f}s",
            str(moment["verification"]),
        ),
        round(anchor, 3),
        {"cinematic": {"source_verified": True, "interest": retention}},
        "cinematic",
        str(moment["id"]),
        str(moment.get("label") or ""),
        "",
        "",
        (anchor_id,),
    )


def analyze(
    timeline: SemanticTimeline,
    config: dict[str, Any],
    excluded: list[list[float]],
    source_key: str,
) -> tuple[list[SemanticPlan], list[dict[str, Any]]]:
    """Independently search every verified anchor before global allocation."""
    plans: list[SemanticPlan] = []
    required: list[dict[str, Any]] = []
    for moment in _moments(config, source_key):
        anchor = float(moment["time"])
        anchor_id = f"cinematic:{source_key}:{moment['id']}"
        region = anchor_search._verified_source_region(timeline, source_key, anchor, config)
        candidates: list[SemanticPlan] = []
        reasons: Counter[str] = Counter()
        attempts = 0
        if region is None:
            reasons["no_verified_source_region"] += 1
        else:
            variants = set(anchor_search._anchor_variants(*region, anchor, config))
            minimum = float(config["semantic_editor"]["minimum_output_seconds"])
            maximum = float(config["semantic_editor"]["maximum_output_seconds"])
            for target in (minimum, 11.0, maximum):
                for lead in (2.0, 4.0, 6.0, 8.0):
                    start, end = round(anchor - lead, 3), round(anchor - lead + target, 3)
                    if region[0] <= start and end <= region[1]:
                        variants.add((start, end))
            for start, end in sorted(variants):
                attempts += 1
                if not (start <= anchor <= end):
                    reasons["anchor_outside_window"] += 1
                    continue
                if any(start < float(cut[1]) and end > float(cut[0]) for cut in excluded):
                    reasons["excluded_window"] += 1
                    continue
                plan = _plan(timeline, source_key, moment, start, end, config)
                # Verified stringout cuts are always fatal. Automatic shot boundaries
                # and combat-specific reload gaps are not cinematic source-integrity walls.
                failures = quality.plan_integrity_violations(plan, timeline, config, source_key)
                hard = [
                    failure
                    for failure in failures
                    if "crosses verified source cut" in failure
                    or "unplanned cross-shot bridge" in failure
                ]
                if hard:
                    reasons["source_integrity:" + hard[0]] += 1
                    continue
                candidates.append(plan)
        candidates.sort(key=lambda p: (-p.score, abs(p.output_duration - 11.0), p.start))
        plans.extend(candidates[:5])
        required.append(
            {
                "id": anchor_id,
                "anchor_time": round(anchor, 3),
                "label": str(moment.get("label") or ""),
                "verification": str(moment["verification"]),
                "coverage_mode": "independent_verified_cinematic_source_search",
                "disposition": "search_required" if candidates else "no_admissible_candidate",
                "attempted_variant_count": attempts,
                "legal_candidate_count": len(candidates),
                "rejections": dict(sorted(reasons.items())),
            }
        )
    return plans, required
