from __future__ import annotations

from typing import Any

EDITORIAL_PROMPT_VERSION = "editor-v3"
EDITORIAL_SCHEMA_VERSION = "editorial-json-v2"

_BOUNDARY_STATUSES = ["COMPLETE", "NEEDS_CONTEXT", "INCOMPLETE", "UNCERTAIN"]
_BOUNDARY_FAILURE_REASONS = [
    "start_requires_prior_context",
    "start_fragment",
    "unresolved_reference",
    "end_incomplete",
    "open_question",
    "unresolved_setup",
    "unresolved_payoff",
    "partial_number_or_unit",
    "followup_context_required",
    "boundary_uncertain",
]
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


def editorial_output_budget(payload: dict[str, Any]) -> int:
    """Return the base output budget for one editorial generation attempt."""

    task = str(payload.get("task") or "")
    if task in {"episode_editorial_profile", "global_concept_comparison"}:
        return 1024
    if task.startswith(("story_moments:", "hook_variants:", "boundary_audit:")):
        return 1536
    return 2048


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


def editorial_json_schema(task: str) -> dict[str, Any]:
    """Return the machine-enforced JSON Schema for a V11 editorial task.

    This deliberately covers only the eight V11 task families. Unknown tasks fail
    closed instead of silently falling back to unconstrained text generation.
    """

    if task == "episode_editorial_profile":
        return _strict_object(
            {
                "summary": _string(),
                "valuable_moment_characteristics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 5,
                },
                "avoid_characteristics": _string_array(max_items=4),
                "confidence": _confidence(),
            }
        )

    if task.startswith("story_moments:"):
        moment = _strict_object(
            {
                "moment_id": _string(),
                "start_word_id": _string(),
                "end_word_id": _string(),
                "semantic_summary": _string(),
                "narrative_structure": _string(),
                "required_prior_context": _string(),
                "required_followup_context": _string(),
                "editorial_reason": _string(),
                "confidence": _confidence(),
            }
        )
        return _strict_object(
            {
                "moments": {
                    "type": "array",
                    "items": moment,
                    "maxItems": 8,
                }
            }
        )

    if task == "clip_concepts":
        concept = _strict_object(
            {
                "concept_id": _string(),
                "story_moment_ids": _string_array(max_items=16),
                "start_word_id": _string(),
                "end_word_id": _string(),
                "semantic_summary": _string(),
                "standalone_context": _string(),
                "required_prior_context": _string(),
                "required_followup_context": _string(),
                "narrative_structure": _string(),
                "recommended_duration": {"type": "number", "minimum": 0.0},
                "visual_dependencies": _string_array(max_items=12),
                "confidence": _confidence(),
            }
        )
        return _strict_object(
            {
                "concepts": {
                    "type": "array",
                    "items": concept,
                    "maxItems": 12,
                }
            }
        )

    if task == "global_concept_comparison":
        return _strict_object(
            {
                "concept_ids": _string_array(max_items=12),
            }
        )

    if task.startswith("hook_variants:"):
        variant = _strict_object(
            {
                "variant_id": _string(),
                "strategy_label": _string(),
                "source_start_word_id": _string(),
                "source_end_word_id": _string(),
                "overlay_text": _string(nullable=True),
                "rationale": _string(),
                "confidence": _confidence(),
            }
        )
        return _strict_object(
            {
                "variants": {
                    "type": "array",
                    "items": variant,
                    "maxItems": 4,
                }
            }
        )

    if task.startswith("edit_plans:"):
        plan = _strict_object(
            {
                "plan_id": _string(),
                "video_id": _string(),
                "concept_id": _string(),
                "variant_id": _string(),
                "source_start_word_id": _string(),
                "source_end_word_id": _string(),
                "hook_start_word_id": _string(),
                "hook_end_word_id": _string(),
                "overlay_text": _string(nullable=True),
                "strategy_label": _string(),
                "caption_platform": {"type": "string", "enum": ["tiktok"]},
                "confidence": _confidence(),
            }
        )
        return _strict_object(
            {
                "plans": {
                    "type": "array",
                    "items": plan,
                    "maxItems": 4,
                }
            }
        )

    if task.startswith("source_hazards:"):
        segment = _strict_object(
            {
                "start_word_id": _string(),
                "end_word_id": _string(),
                "classification": {
                    "type": "string",
                    "enum": _HAZARD_CLASSIFICATIONS,
                },
                "confidence": _confidence(),
                "evidence": _string_array(max_items=8),
            }
        )
        return _strict_object(
            {
                "segments": {
                    "type": "array",
                    "items": segment,
                    "maxItems": 64,
                }
            }
        )

    if task.startswith("boundary_audit:"):
        return _strict_object(
            {
                "start_status": {"type": "string", "enum": _BOUNDARY_STATUSES},
                "end_status": {"type": "string", "enum": _BOUNDARY_STATUSES},
                "standalone_status": {"type": "string", "enum": _BOUNDARY_STATUSES},
                "required_prior_context": _string(),
                "required_followup_context": _string(),
                "prior_context_included": {"type": "boolean"},
                "followup_context_included": {"type": "boolean"},
                "setup_resolved": {"type": "boolean"},
                "payoff_resolved": {"type": "boolean"},
                "open_questions": _string_array(max_items=12),
                "open_references": _string_array(max_items=12),
                "narrative_structure": _string(),
                "boundary_confidence": _confidence(),
                "failure_reasons": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": _BOUNDARY_FAILURE_REASONS,
                    },
                    "maxItems": 10,
                    "uniqueItems": True,
                },
                "repair_start_word_id": _string(nullable=True),
                "repair_end_word_id": _string(nullable=True),
            }
        )

    raise ValueError(f"unsupported V11 editorial task for structured generation: {task!r}")


def editorial_contract(task: str) -> str:
    common = (
        "Output exactly one compact JSON object, no markdown and no extra keys. "
        "The runtime constrains generation to the task JSON Schema; satisfy its required fields "
        "without adding commentary outside the object. Keep prose fields concise. For range "
        "fields copy the supplied short word_ref values, never reconstruct or abbreviate word_id "
        "values yourself. Campaign min_clip_seconds is a hard floor and max_clip_seconds is a "
        "hard ceiling for final EditPlan source ranges. Never invent source wording. Preserve "
        "source chronology. The first audible content must be understandable without hidden prior "
        "context, and the ending must resolve the semantic obligation created by the clip. Reject "
        "incomplete stories rather than amputating them. Generated overlays must not falsify "
        "spoken material. Campaign rules are hard constraints; exclude sponsor and promo regions "
        "when policy forbids them. Lower confidence when required evidence is uncertain. "
    )
    if task == "episode_editorial_profile":
        return common + (
            'Schema: {"summary":"<=60 words",'
            '"valuable_moment_characteristics":["3-5 short strings"],'
            '"avoid_characteristics":["0-4 short strings"],"confidence":0.0}. '
        )
    if task.startswith("story_moments:"):
        return common + (
            'Schema: {"moments":[{"moment_id":"unique",'
            '"start_word_id":"first word_ref","end_word_id":"last word_ref",'
            '"semantic_summary":"<=24 words","narrative_structure":"short label",'
            '"required_prior_context":"<=16 words or empty",'
            '"required_followup_context":"<=16 words or empty",'
            '"editorial_reason":"<=20 words","confidence":0.0}]}. '
            "Return at most 8 non-overlapping meaningful moments. Do not copy full word-ID lists. "
            "Story moments are semantic evidence units and may be shorter than the campaign final "
            "clip minimum; do not pad them merely to satisfy duration. "
        )
    if task == "clip_concepts":
        return common + (
            'Schema: {"concepts":[{"concept_id":"unique","story_moment_ids":["ids"],'
            '"start_word_id":"first word_ref","end_word_id":"last word_ref",'
            '"semantic_summary":"<=24 words","standalone_context":"<=16 words or empty",'
            '"required_prior_context":"<=16 words or empty",'
            '"required_followup_context":"<=16 words or empty",'
            '"narrative_structure":"short label","recommended_duration":20.0,'
            '"visual_dependencies":["short labels"],"confidence":0.0}]}. '
            "Return at most 12 materially distinct contiguous concepts. A concept may be a short "
            "semantic core, but recommended_duration is only editorial guidance and is never proof "
            "that its current source range already satisfies campaign duration. Downstream "
            "EditPlans must use timestamped source_context_words to choose the actual final range. "
            "Do not copy full word-ID lists. "
        )
    if task == "global_concept_comparison":
        return common + 'Schema: {"concept_ids":["best-first supplied concept IDs"]}. '
    if task.startswith("hook_variants:"):
        return common + (
            'Schema: {"variants":[{"variant_id":"unique","strategy_label":"<=8 words",'
            '"source_start_word_id":"first word_ref","source_end_word_id":"last word_ref",'
            '"overlay_text":null,"rationale":"<=16 words","confidence":0.0}]}. '
            "Return at most 4 materially different truthful hooks. Do not copy full word-ID lists. "
        )
    if task.startswith("edit_plans:"):
        return common + (
            'Schema: {"plans":[{"plan_id":"unique","video_id":"supplied video ID",'
            '"concept_id":"supplied concept ID","variant_id":"supplied hook ID",'
            '"source_start_word_id":"first edit word_ref",'
            '"source_end_word_id":"last edit word_ref",'
            '"hook_start_word_id":"first hook word_ref",'
            '"hook_end_word_id":"last hook word_ref",'
            '"overlay_text":null,"strategy_label":"<=8 words",'
            '"caption_platform":"tiktok","confidence":0.0}]}. '
            "Return at most 4 contiguous chronological plans. source_context_words is the "
            "authoritative timestamped range evidence. Before emitting each plan, locate its "
            "source_start_word_id and source_end_word_id in source_context_words and calculate "
            "duration = end.source_end - start.source_start. Emit the plan only when that measured "
            "duration is >= campaign.min_clip_seconds and <= campaign.max_clip_seconds. The final "
            "source range may extend before or after the concept start/end when the concept is "
            "shorter than the campaign minimum, but the extension must remain one coherent story, "
            "must preserve chronology, and must contain the spoken hook. Never return only the "
            "hook unless its measured duration already satisfies the campaign bounds. If no "
            "coherent duration-valid source range exists, omit that plan instead of returning an "
            "out-of-bounds range. Do not copy full word-ID lists. "
        )
    if task.startswith("source_hazards:"):
        return common + (
            'Schema: {"segments":[{"start_word_id":"first word_ref",'
            '"end_word_id":"last word_ref","classification":"editorial_content",'
            '"confidence":0.0,"evidence":["short multimodal reason"]}]}. '
            "Use only editorial_content, advertisement, sponsor_read, promo, intro, outro, "
            "housekeeping, graphic_heavy, or unknown. Cover the entire supplied word interval "
            "with exhaustive chronological segments. "
        )
    if task.startswith("boundary_audit:"):
        return common + (
            'Schema: {"start_status":"COMPLETE","end_status":"COMPLETE",'
            '"standalone_status":"COMPLETE","required_prior_context":"",'
            '"required_followup_context":"","prior_context_included":true,'
            '"followup_context_included":true,"setup_resolved":true,'
            '"payoff_resolved":true,"open_questions":[],"open_references":[],'
            '"narrative_structure":"short label","boundary_confidence":0.0,'
            '"failure_reasons":[],"repair_start_word_id":null,'
            '"repair_end_word_id":null}. '
            "Statuses are COMPLETE, NEEDS_CONTEXT, INCOMPLETE, or UNCERTAIN. Failure reasons may "
            "only be start_requires_prior_context, start_fragment, unresolved_reference, "
            "end_incomplete, open_question, unresolved_setup, unresolved_payoff, "
            "partial_number_or_unit, followup_context_required, or boundary_uncertain. "
        )
    raise ValueError(f"unsupported V11 editorial task: {task!r}")
