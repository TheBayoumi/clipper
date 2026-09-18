from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PAYOFF_KINDS = {"outcome_like", "impact"}


@dataclass(frozen=True)
class HostileDecision:
    hostile: bool
    score: float
    reason: str
    actor_state: str = "unknown"


def _f(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def validate_configuration(config: dict[str, Any]) -> None:
    verifier = config.get("combat_state_verifier", {})
    local = verifier.get("local_interaction_verifier", {})
    if not bool(verifier.get("enabled", False)):
        raise RuntimeError("combat_state_verifier.enabled must be true")
    if not bool(local.get("enabled", False)):
        raise RuntimeError("local_interaction_verifier.enabled must be true")


def local_confirmation(event: Any, config: dict[str, Any]) -> tuple[bool, float]:
    cfg = config["combat_state_verifier"]["local_interaction_verifier"]
    evidence = dict(getattr(event, "evidence", {}) or {})
    minimum = _f(cfg.get("minimum_hitmarker_score", 0.34), 0.34)
    attempted = _f(evidence.get("local_refine_attempted")) >= 0.5
    score = _f(evidence.get("local_hitmarker_score"))
    return bool(attempted and score >= minimum), score


def hostile_decision(event: Any, config: dict[str, Any]) -> HostileDecision:
    """Hostility requires local direct player-target interaction in every case."""
    validate_configuration(config)
    local_ok, local_score = local_confirmation(event, config)
    if not local_ok:
        return HostileDecision(False, local_score, "missing_direct_interaction_confirmation")

    cfg = config["combat_state_verifier"]
    kinds = set(getattr(event, "kinds", ()) or ())
    evidence = dict(getattr(event, "evidence", {}) or {})
    confidence = _f(getattr(event, "confidence", 0.0))
    combat = _f(evidence.get("combat"))
    outcome = _f(evidence.get("outcome"))
    impact = _f(evidence.get("impact"))
    audio = _f(evidence.get("audio_transient"))
    center = _f(evidence.get("center_motion"))
    minimum = _f(cfg.get("minimum_hostile_event_score", 0.64), 0.64)

    if (
        "outcome_like" in kinds
        and outcome >= _f(cfg.get("outcome_minimum", 0.52), 0.52)
        and max(combat, impact) >= _f(cfg.get("payoff_combat_minimum", 0.42), 0.42)
    ):
        score = min(
            1.0,
            0.44 * max(confidence, outcome)
            + 0.30 * max(combat, impact)
            + 0.14 * max(audio, center)
            + 0.12 * local_score,
        )
        ok = score >= minimum
        return HostileDecision(
            ok, score, "directly_confirmed_outcome", "hostile" if ok else "unknown"
        )

    if (
        "impact" in kinds
        and impact >= _f(cfg.get("impact_minimum", 0.64), 0.64)
        and combat >= _f(cfg.get("impact_combat_minimum", 0.55), 0.55)
    ):
        score = min(
            1.0,
            0.40 * max(confidence, impact)
            + 0.32 * combat
            + 0.16 * max(audio, center)
            + 0.12 * local_score,
        )
        ok = score >= minimum
        return HostileDecision(
            ok, score, "directly_confirmed_impact", "hostile" if ok else "unknown"
        )

    if (
        "combat_burst" in kinds
        and combat >= _f(cfg.get("strong_combat_minimum", 0.70), 0.70)
        and audio >= _f(cfg.get("strong_combat_audio_transient_minimum", 0.62), 0.62)
        and center >= _f(cfg.get("strong_combat_center_motion_minimum", 0.40), 0.40)
    ):
        score = min(
            1.0, 0.40 * max(confidence, combat) + 0.22 * audio + 0.18 * center + 0.20 * local_score
        )
        ok = score >= minimum
        return HostileDecision(
            ok, score, "directly_confirmed_combat_burst", "hostile" if ok else "unknown"
        )

    return HostileDecision(False, 0.0, "insufficient_hostile_evidence")


def self_test() -> None:
    from types import SimpleNamespace

    config = {
        "combat_state_verifier": {
            "enabled": True,
            "local_interaction_verifier": {"enabled": True, "minimum_hitmarker_score": 0.34},
            "minimum_hostile_event_score": 0.64,
            "outcome_minimum": 0.52,
            "payoff_combat_minimum": 0.42,
            "impact_minimum": 0.64,
            "impact_combat_minimum": 0.55,
            "strong_combat_minimum": 0.70,
            "strong_combat_audio_transient_minimum": 0.62,
            "strong_combat_center_motion_minimum": 0.40,
        }
    }
    good = SimpleNamespace(
        kinds=("combat_burst", "impact", "outcome_like"),
        confidence=0.9,
        evidence={
            "combat": 0.82,
            "impact": 0.82,
            "outcome": 0.8,
            "audio_transient": 0.9,
            "center_motion": 0.9,
            "local_refine_attempted": 1.0,
            "local_hitmarker_score": 0.9,
        },
    )
    if not hostile_decision(good, config).hostile:
        raise AssertionError("directly verified hostile interaction was rejected")
    bad = SimpleNamespace(
        kinds=("combat_burst", "impact", "outcome_like"),
        confidence=0.99,
        evidence={
            "combat": 0.99,
            "impact": 0.99,
            "outcome": 0.99,
            "audio_transient": 0.99,
            "center_motion": 0.99,
            "local_refine_attempted": 1.0,
            "local_hitmarker_score": 0.0,
        },
    )
    if hostile_decision(bad, config).hostile:
        raise AssertionError("coarse evidence bypassed direct-interaction verification")
    contact = SimpleNamespace(
        kinds=("contact",),
        confidence=1.0,
        evidence={"local_refine_attempted": 1.0, "local_hitmarker_score": 1.0},
    )
    if hostile_decision(contact, config).hostile:
        raise AssertionError("contact-only event became hostile")
    print("MW4 canonical hostile verifier self-test: PASS")
