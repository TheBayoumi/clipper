from __future__ import annotations

from typing import Any


def editorial_output_budget(payload: dict[str, Any]) -> int:
    task = str(payload.get("task") or "")
    if task in {"episode_editorial_profile", "global_concept_comparison"}:
        return 768
    if task.startswith("hook_variants:"):
        return 1024
    return 1536


def editorial_contract(task: str) -> str:
    common = (
        "Output exactly one compact JSON object, no markdown and no extra keys. "
        "Keep prose fields concise. For range fields copy the supplied short word_ref values, "
        "never reconstruct or abbreviate word_id values yourself. "
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
    return common + "Follow the task payload exactly."
