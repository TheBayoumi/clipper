from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_RESUME_SCHEMA = "clipper-resume-provenance-v1"
_ARCHITECTURE = "autonomous-multimodal-quality-graph"


def _object(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}


def _read_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{label} is unavailable or invalid: {path}") from exc
    return _object(value, label=label)


def validate_resume_artifact(
    *,
    artifact_root: str | Path,
    requested_run_id: str,
    provenance: dict[str, Any],
    current_brief: dict[str, Any],
    current_source_hashes: dict[str, str],
    current_compute_profile: str,
    current_cache_root: str | Path,
    current_models: dict[str, dict[str, str]],
) -> dict[str, Any]:
    """Bind a resume request to one compatible persisted artifact run.

    The prior run proves provenance. Actual cache reuse remains governed by the
    current content-addressed stage/model/contract keys, never by this run label.
    """
    requested = str(requested_run_id or "").strip()
    if not requested:
        raise RuntimeError("resume provenance requires a requested run ID")
    if str(provenance.get("schema_version") or "") != _RESUME_SCHEMA:
        raise RuntimeError("resume provenance schema is unsupported")
    if str(provenance.get("workflow_run_id") or "") != requested:
        raise RuntimeError("resume provenance workflow run ID does not match the request")

    artifact_path = str(provenance.get("artifact_run_path") or "").strip()
    relative = Path(artifact_path.lstrip("/"))
    if (
        not artifact_path.startswith("/")
        or relative.is_absolute()
        or len(relative.parts) != 1
        or relative.parts[0] in {"", ".", ".."}
    ):
        raise RuntimeError("resume artifact path must identify one direct artifact run directory")

    root = Path(artifact_root)
    run_dir = root / relative
    manifest = _read_object(run_dir / "manifest.json", label="prior resume manifest")
    prior_brief = _read_object(
        run_dir / "brief.normalized.json",
        label="prior normalized brief",
    )

    campaign_id = str(current_brief.get("campaign_id") or "")
    if not campaign_id or str(manifest.get("campaign_id") or "") != campaign_id:
        raise RuntimeError("resume artifact campaign does not match the current campaign")
    if str(provenance.get("campaign_id") or "") != campaign_id:
        raise RuntimeError("resume registry campaign does not match the current campaign")
    if prior_brief != current_brief:
        raise RuntimeError("resume artifact normalized brief is not compatible with the current brief")

    metadata = _object(manifest.get("run_metadata"), label="prior resume run_metadata")
    if str(metadata.get("architecture") or "") != _ARCHITECTURE:
        raise RuntimeError("resume artifact architecture is not compatible")
    if str(metadata.get("compute_profile") or "") != current_compute_profile:
        raise RuntimeError("resume artifact compute profile is not compatible")
    prior_git_sha = str(metadata.get("git_sha") or "").strip().lower()
    if prior_git_sha != str(provenance.get("artifact_origin_head_sha") or "").strip().lower():
        raise RuntimeError("resume artifact source SHA does not match reviewed provenance")

    prior_hashes = _object(metadata.get("source_hashes"), label="prior resume source hashes")
    normalized_prior_hashes = {key: str(value) for key, value in prior_hashes.items()}
    normalized_current_hashes = {
        str(key): str(value) for key, value in current_source_hashes.items()
    }
    registry_hashes = _object(provenance.get("source_hashes"), label="resume registry source hashes")
    normalized_registry_hashes = {key: str(value) for key, value in registry_hashes.items()}
    if normalized_prior_hashes != normalized_current_hashes:
        raise RuntimeError("resume artifact source hashes do not match the current source masters")
    if normalized_registry_hashes != normalized_current_hashes:
        raise RuntimeError("resume registry source hashes do not match the current source masters")

    expected_cache_root = str(Path(current_cache_root))
    if str(provenance.get("cache_root") or "") != expected_cache_root:
        raise RuntimeError("resume registry cache root does not match the active cache root")

    editorial = _object(metadata.get("editorial_inference"), label="prior editorial inference")
    prior_editorial_model = _object(
        editorial.get("model"),
        label="prior editorial model identity",
    )
    current_editorial = current_models.get("editorial")
    if not isinstance(current_editorial, dict) or prior_editorial_model != current_editorial:
        raise RuntimeError("resume artifact editorial model identity is not compatible")

    checked_grounding_models = 0
    grounding = _object(metadata.get("grounding_inference"), label="prior grounding inference")
    grounding_models = grounding.get("models")
    if not isinstance(grounding_models, list):
        raise RuntimeError("prior grounding model evidence must be a list")
    for source_model in grounding_models:
        if not isinstance(source_model, dict):
            raise RuntimeError("prior grounding model evidence contains a non-object entry")
        for stage in ("transcription", "alignment", "diarization"):
            stage_meta = source_model.get(stage)
            if not isinstance(stage_meta, dict):
                continue
            prior_model = stage_meta.get("model")
            if not isinstance(prior_model, dict):
                continue
            current_model = current_models.get(stage)
            if not isinstance(current_model, dict) or prior_model != current_model:
                raise RuntimeError(f"resume artifact {stage} model identity is not compatible")
            checked_grounding_models += 1

    checked_visual_models = 0
    visual = _object(metadata.get("visual_inference"), label="prior visual inference")
    scout_entries = visual.get("scout")
    if not isinstance(scout_entries, list):
        raise RuntimeError("prior visual scout evidence must be a list")
    for scout_entry in scout_entries:
        if not isinstance(scout_entry, dict):
            raise RuntimeError("prior visual scout evidence contains a non-object entry")
        prior_model = scout_entry.get("model")
        if not isinstance(prior_model, dict):
            continue
        current_model = current_models.get("vision_scout")
        if not isinstance(current_model, dict) or prior_model != current_model:
            raise RuntimeError("resume artifact visual scout model identity is not compatible")
        checked_visual_models += 1

    prior_status = str(manifest.get("status") or "")
    if prior_status not in {"FAILED", "SUCCESS", "DEGRADED"}:
        raise RuntimeError("resume artifact has no terminal pipeline status")

    return {
        "status": "PASS",
        "schema_version": _RESUME_SCHEMA,
        "workflow_run_id": requested,
        "artifact_run_path": artifact_path,
        "artifact_origin_workflow_run_id": str(
            provenance.get("artifact_origin_workflow_run_id") or ""
        ),
        "artifact_origin_head_sha": prior_git_sha,
        "campaign_id": campaign_id,
        "source_hashes": normalized_current_hashes,
        "compute_profile": current_compute_profile,
        "cache_root": expected_cache_root,
        "prior_pipeline_status": prior_status,
        "prior_status_reason": str(manifest.get("status_reason") or ""),
        "editorial_model_identity": current_editorial,
        "grounding_model_evidence_checked": checked_grounding_models,
        "visual_model_evidence_checked": checked_visual_models,
        "cache_reuse_policy": "current-content-addressed-stage-model-contract-keys-only",
    }
