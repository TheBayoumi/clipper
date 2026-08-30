from __future__ import annotations

from typing import Any

from ..stage_contracts import content_fingerprint

# Semantic provider identities only. Exact task contracts are fingerprinted from their
# prompt + JSON Schema and participate in the StageContract cache identity.
EDITORIAL_IDENTITY = "editor"
EDITORIAL_SCHEMA_IDENTITY = "structured-json"

_HAZARD_CLASSIFICATIONS = [
    "editorial_content",
    "advertisement",
    "sponsor_read",
    "promo",
    "intro",
    "outro",
    "housekeeping",
    "graphic_heavy",
    "unknown",
]
_QUALITY_DECISIONS = ["PASS", "REJECT", "ESCALATE"]


def _string(*, nullable: bool = False) -> dict[str, Any]:
    if nullable:
        return {"type": ["string", "null"]}
    return {"type": "string"}


def _confidence() -> dict[str, Any]:
    return {"type": "number", "minimum": 0.0, "maximum": 1.0}


def _string_array(*, max_items: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": {"type": "string"}}
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def _strict_object(
    properties: dict[str, Any],
    *,
    required: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required if required is not None else list(properties),
        "additionalProperties": False,
    }


def editorial_task_family(task: str) -> str:
    if task.startswith("source_hazards:"):
        return "source_hazards"
    if task.startswith("semantic_cores:"):
        return "semantic_cores"
    if task.startswith("narrative_envelope:"):
        return "narrative_envelope"
    if task.startswith("quality_windows:"):
        return "quality_windows"
    raise ValueError(f"unsupported production editorial task: {task!r}")


def editorial_json_schema(task: str) -> dict[str, Any]:
    """Machine-enforced output schema for the adaptive quality graph."""
    family = editorial_task_family(task)

    if family == "source_hazards":
        segment = _strict_object(
            {
                "start_word_id": _string(),
                "end_word_id": _string(),
                "classification": {"type": "string", "enum": _HAZARD_CLASSIFICATIONS},
                "confidence": _confidence(),
                "evidence": _string_array(max_items=8),
            }
        )
        return _strict_object({"segments": {"type": "array", "items": segment, "maxItems": 64}})

    if family == "semantic_cores":
        core = _strict_object(
            {
                "core_id": _string(),
                "start_word_id": _string(),
                "end_word_id": _string(),
                "semantic_summary": _string(),
                "editorial_reason": _string(),
                "confidence": _confidence(),
            }
        )
        return _strict_object({"cores": {"type": "array", "items": core}})

    if family == "narrative_envelope":
        return _strict_object(
            {
                "envelope_id": _string(),
                "core_id": _string(),
                "start_word_id": _string(),
                "end_word_id": _string(),
                "required_prior_context": _string(),
                "required_followup_context": _string(),
                "setup_resolved": {"type": "boolean"},
                "payoff_resolved": {"type": "boolean"},
                "reference_resolution": _string_array(),
                "confidence": _confidence(),
            }
        )

    if family == "quality_windows":
        return _strict_object(
            {
                "core_id": _string(),
                "selected_window_id": _string(nullable=True),
                "decision": {"type": "string", "enum": _QUALITY_DECISIONS},
                "quality_score": _confidence(),
                "opening_strategy": _string(),
                "rationale": _string(),
                "confidence": _confidence(),
            }
        )

    raise AssertionError(family)


def editorial_contract(task: str) -> str:
    """Adaptive semantic contract with no lexical hook vocabulary or fixed hook taxonomy."""
    family = editorial_task_family(task)
    common = (
        "Output exactly one compact JSON object matching the supplied schema, with no markdown "
        "and no extra keys. Ground every claim and range in the supplied timestamped source "
        "evidence. Preserve chronology and copy supplied word_ref values exactly. Never invent "
        "source wording or infer missing facts. Campaign policy and duration bounds are hard "
        "constraints. Evaluate what is interesting, complete, and potentially compelling from "
        "the actual semantic and multimodal evidence; do not use a predeclared vocabulary, topic "
        "list, hook category, emotion keyword, numeric pattern, or domain-specific template. "
        "An opening is valuable only if this source makes it valuable in context. Quality may be "
        "zero; never manufacture moments to satisfy a count. Lower confidence or escalate when "
        "the evidence needed for a decision is insufficient. "
    )

    if family == "source_hazards":
        return common + (
            "Classify the complete supplied interval into exhaustive chronological policy "
            "segments using both speech and multimodal evidence. The allowed classification "
            "labels are policy ontology, not editorial-value keywords. Use unknown when evidence "
            "is insufficient."
        )
    if family == "semantic_cores":
        return common + (
            "Identify every independently worthwhile semantic nucleus supported by this source. "
            "A SemanticCore is the smallest contiguous interval containing the interesting idea, "
            "event, demonstration, reaction, visual development, or other source-grounded reason "
            "a viewer might genuinely care. Infer that reason from the evidence itself. Do not "
            "pad to campaign duration. Return an empty cores array when nothing is worthwhile."
        )
    if family == "narrative_envelope":
        return common + (
            "For the supplied SemanticCore, choose the smallest contiguous source interval that "
            "contains all context needed to understand setup, references, causality, and payoff. "
            "What counts as setup or payoff must be inferred from this source, including visual "
            "or action context when relevant. Do not expand merely to hit delivery duration. Mark "
            "unresolved setup/payoff false when the available evidence cannot form a complete "
            "self-contained narrative."
        )
    if family == "quality_windows":
        return common + (  # noqa: S608 - editorial prose, not SQL
            "All supplied feasible_windows have already been deterministically proven legal. "
            "Judge only those supplied window IDs and never invent timestamps. PASS only when a "
            "window is independently strong enough to publish based on this source and campaign, "
            "not because it resembles a predefined hook pattern. Select the strongest supplied "
            "window when PASS. opening_strategy must briefly describe, in free-form source-specific "
            "language, what makes the actual first moments of that selected window work as an "
            "opening; never choose from or imitate a fixed category list. For REJECT/ESCALATE, "
            "opening_strategy should briefly explain why no reliable opening can be endorsed. Use "
            "REJECT when none deserves a clip and ESCALATE when evidence is insufficient."
        )
    raise AssertionError(family)


def editorial_contract_fingerprint(task: str) -> str:
    """Content-address the exact prompt and schema used by one task family."""
    return content_fingerprint(
        {
            "family": editorial_task_family(task),
            "contract": editorial_contract(task),
            "schema": editorial_json_schema(task),
        }
    )
