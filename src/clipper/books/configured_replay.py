"""Private configured native cache replay entrypoint, never an inference host."""

from __future__ import annotations

from pathlib import Path

from . import chapter, pipeline_plan, scripting
from .orchestrator import BookOrchestrator


def replay_configured(
    *,
    config_path: Path,
    selected_source: Path,
    section_manifest: Path,
    artifact_root: Path,
    review_file: Path | None = None,
    expected_brief_identity: str | None = None,
) -> dict[str, object]:
    """Bind actual configured evaluator to retained actual-Qwen provenance only.

    CPU permissions stay false and every computational stage must be disabled.
    The real producer function is an identity binding, not permission to call it.
    Missing, corrupt or different-origin native caches fail closed.
    """
    config = pipeline_plan.validate_config(
        pipeline_plan.strict_json(pipeline_plan._read(config_path, pipeline_plan.MAX_JSON)),
        pipeline_plan.registry(),
    )
    if config["targets"] not in (["brief"], ["script"]):
        raise chapter.ChapterStop("configured replay supports brief or script only")
    for name, entry in config["stages"].items():
        if name not in {"source", "human_review"} and entry["enabled"]:
            raise chapter.ChapterStop("configured replay forbids enabled computational stages")
    result = chapter.evaluate_hybrid_chapter(
        BookOrchestrator(artifact_root),
        selected_source,
        section_manifest,
        pipeline_config=config,
        plan=chapter.ChapterPlan(),
        review_file=review_file,
        expected_brief_identity=expected_brief_identity,
        argument_completion=scripting._ORIGINAL_LOCAL_CHAPTER_COMPLETION,
        audit_completion=scripting._ORIGINAL_LOCAL_CHAPTER_COMPLETION,
        reconcile_completion=scripting._ORIGINAL_LOCAL_CHAPTER_COMPLETION,
        adaptation_completion=scripting._ORIGINAL_LOCAL_CHAPTER_COMPLETION,
        script_completion=scripting._ORIGINAL_LOCAL_CHAPTER_COMPLETION,
    )
    return {
        "schema": "clipper-configured-native-replay-cli-v1",
        "mode": "NATIVE_CACHE_REPLAY_ONLY",
        "status": result["status"],
        "configured_execution_receipt_identity": result["configured_execution_receipt_identity"],
        "configured_stage_results": result["configured_stage_results"],
        "inference_calls": 0,
        "publish_eligible": False,
    }
