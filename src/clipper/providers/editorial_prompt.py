from __future__ import annotations

from typing import Any

EDITORIAL_PROMPT_VERSION = "editor-v2"
EDITORIAL_SCHEMA_VERSION = "editorial-json-v2"


def editorial_output_budget(payload: dict[str, Any]) -> int:
    task = str(payload.get("task") or "")
    if task in {"episode_editorial_profile", "global_concept_comparison"}:
        return 768
    if task.startswith(("hook_variants:", "boundary_audit:")):
        return 1024
    return 1536


def editorial_contract(task: str) -> str:
    common = (
        "Output exactly one compact JSON object, no markdown and no extra keys. "
        "Keep prose fields concise. For range fields copy the supplied short word_ref values, "
        "never reconstruct or abbreviate word_id values yourself. "
        "The campaign maximum duration is a ceiling, never a target. Never invent source wording. "
        "Preserve source chronology. The first audible content must be understandable without "
        "hidden prior context, and the ending must resolve the semantic obligation created by the "
        "clip. Reject incomplete stories rather than amputating them. Generated overlays must not "
        "falsify spoken material. Campaign rules are hard constraints; exclude sponsor and promo "
        "regions when policy forbids them. Lower confidence when required evidence is uncertain. "
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
            "Return at most 12 materially distinct contiguous concepts. "
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
            "Return at most 4 contiguous chronological plans. Do not copy full word-ID lists. "
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
    return common + "Follow the task payload exactly."
