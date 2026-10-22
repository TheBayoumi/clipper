"""Reviewable visual proposals joined to independently verified Clipper evidence.

No neural model claim becomes an admissible payoff, kill, or finishing move here.
Only the canonical detector and manually reviewed campaign evidence can verify
such anchors. This module creates machine-readable dispositions for review.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from . import interaction
from .semantics import SemanticTimeline


def _manual_cinematic_times(config: dict[str, Any], source_key: str) -> dict[str, float]:
    settings = config.get("cinematic_moments") or {}
    verified = (settings.get("verified_anchors") or {}).get(source_key) or []
    result: dict[str, float] = {}
    for item in verified:
        if item.get("verification") and item.get("id") and "time" in item:
            result[str(item["id"])] = float(item["time"])
    return result


def reconcile(
    report: dict[str, Any],
    timeline: SemanticTimeline,
    source_key: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Preserve source-wide observations without treating VLM text as proof.

    A nearby Clipper-verified event is supporting context only. Neural claims
    remain proposals; their independent review is not delegated to a model.
    """
    if report.get("status") != "PROPOSALS_ONLY":
        raise RuntimeError("neural timeline must carry proposals-only evidence")
    identity = report.get("source_identity")
    if not isinstance(identity, dict) or identity.get("source_key") != source_key:
        raise RuntimeError("neural timeline source identity differs from Clipper source")
    raw = report.get("observations")
    coverage = report.get("coverage")
    if (
        not isinstance(raw, list)
        or not isinstance(coverage, dict)
        or not (coverage.get("expected_samples") == coverage.get("observed_samples") == len(raw))
    ):
        raise RuntimeError("neural timeline is missing complete source-wide coverage")
    manual = _manual_cinematic_times(config, source_key)
    joined: list[dict[str, Any]] = []
    observed_times: set[float] = set()
    for item in raw:
        if not isinstance(item, dict) or item.get("verification") != "PROPOSED":
            raise RuntimeError("neural visual model claim escaped proposal quarantine")
        stamp = float(item["timestamp"])
        if (
            not math.isfinite(stamp)
            or stamp in observed_times
            or stamp < 0
            or stamp >= timeline.duration
        ):
            raise RuntimeError("neural observation has an invalid or duplicate source clock")
        observed_times.add(stamp)
        labels = item.get("event_labels")
        if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
            raise RuntimeError("neural observation labels are invalid")
        fingerprint = hashlib.sha256(
            json.dumps(
                {"source": source_key, "time": stamp, "labels": labels},
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]
        corroborated: list[dict[str, Any]] = []
        if labels:
            for event in timeline.consolidated_events:
                if abs(float(event.time) - stamp) > 2.0:
                    continue
                decision = interaction.hostile_decision(event, config)
                if not decision.hostile:
                    continue
                corroborated.append(
                    {
                        "origin": "clipper_direct_interaction_verifier",
                        "verified_anchor_time": round(float(event.time), 3),
                        "event_kinds": list(event.kinds),
                        "existing_outcome": "outcome_like" in event.kinds,
                        "reason": decision.reason,
                    }
                )
            for anchor_id, moment_time in manual.items():
                if abs(moment_time - stamp) <= 2.0:
                    corroborated.append(
                        {
                            "origin": "independently_reviewed_campaign_anchor",
                            "verified_anchor_id": f"cinematic:{source_key}:{anchor_id}",
                            "verified_anchor_time": round(moment_time, 3),
                        }
                    )
        joined.append(
            {
                "neural_proposal_id": f"neural:{source_key}:{fingerprint}",
                "source_time_seconds": stamp,
                "event_labels": labels,
                "summary": item.get("summary", ""),
                "model_confidence": item.get("confidence"),
                "verification": "PROPOSED" if labels else "NO_EVENT_OBSERVED",
                "disposition": (
                    "CORROBORATES_EXISTING_ANCHOR"
                    if corroborated
                    else "REQUIRES_INDEPENDENT_SOURCE_REVIEW"
                    if labels
                    else "NO_CLAIM"
                ),
                "existing_verified_evidence": corroborated,
                "may_create_required_anchor": False,
            }
        )
    return {
        "status": "REVIEW_QUEUE_ONLY",
        "source_key": source_key,
        "source_identity": identity,
        "model": report.get("model"),
        "coverage": {
            "observations": len(raw),
            "review_required": sum(
                x["disposition"] == "REQUIRES_INDEPENDENT_SOURCE_REVIEW" for x in joined
            ),
            "corroborated_existing": sum(
                x["disposition"] == "CORROBORATES_EXISTING_ANCHOR" for x in joined
            ),
        },
        "proposals": joined,
    }
