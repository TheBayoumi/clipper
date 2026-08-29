import ast
import json
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest
import yaml


def _script_function(name: str) -> Any:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    namespace: dict[str, Any] = {"Any": Any, "json": json}
    exec(  # noqa: S102 - execute one parsed helper from the repository-owned worker script
        compile(ast.Module(body=[function], type_ignores=[]), "<modal-helper>", "exec"), namespace
    )
    return namespace[name]


def _modal_pipeline_helpers(*names: str) -> dict[str, Any]:
    source = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in set(names)
    ]
    namespace: dict[str, Any] = {
        "Any": Any,
        "Path": Path,
        "json": json,
        "parse_qs": parse_qs,
        "urlparse": urlparse,
    }
    exec(  # noqa: S102 - execute repository-owned pure helper functions only
        compile(ast.Module(body=selected, type_ignores=[]), "<modal-pipeline-helpers>", "exec"),
        namespace,
    )
    return {name: namespace[name] for name in names}


def test_public_youtube_acquisition_uses_bgutil_before_optional_cookies() -> None:
    source = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    assert "youtubepot-bgutilscript:" in source
    assert "server_home=/root/bgutil-ytdlp-pot-provider/server" in source
    assert "youtube:player_client=default,mweb" in source
    assert "youtube:player_client=web_embedded,android_vr" in source
    assert '"yt-dlp[default]>=2026.7.4,<2027"' in source
    assert '"--js-runtimes"' in source
    assert '"node"' in source
    assert '"bgutil_default_mweb"' in source
    assert '"cookies_bgutil_default_mweb"' in source
    assert source.index('"bgutil_default_mweb"') < source.index('"cookies_bgutil_default_mweb"')


def test_source_acquisition_is_exact_and_content_addressed() -> None:
    source = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    acquire = source.split("def acquire_source(", 1)[1].split("class VolumeSourceClient", 1)[0]
    assert "source acquisition requires an https video_url" in source
    assert "source acquisition requires a safe video_id" in acquire
    assert '"--no-playlist"' in acquire
    assert '"bestvideo+bestaudio/best"' in acquire
    assert '"quality_policy": "highest_available_no_transcode"' in source
    assert 'volume_path = f"/inputs/{digest}{suffix}"' in acquire
    assert "content-addressed source master hash mismatch" in acquire


def test_source_acquisition_binds_requested_and_extracted_identity() -> None:
    helpers = _modal_pipeline_helpers(
        "_youtube_video_id",
        "_validate_extracted_source_identity",
    )
    youtube_id = helpers["_youtube_video_id"]
    validate = helpers["_validate_extracted_source_identity"]

    assert youtube_id("https://www.youtube.com/watch?v=abc_123") == "abc_123"
    assert youtube_id("https://youtu.be/abc_123") == "abc_123"
    with pytest.raises(ValueError, match="YouTube video URL"):
        youtube_id("https://redirect.example.test/watch?v=abc_123")

    identity = validate(
        expected_video_id="abc_123",
        expected_channel_id="UC_authorized",
        requested_url="https://www.youtube.com/watch?v=abc_123",
        actual_video_id="abc_123",
        actual_channel_id="UC_authorized",
        canonical_url="https://www.youtube.com/watch?v=abc_123",
    )
    assert identity == {
        "video_id": "abc_123",
        "channel_id": "UC_authorized",
        "webpage_url": "https://www.youtube.com/watch?v=abc_123",
    }

    with pytest.raises(RuntimeError, match="URL video ID mismatch"):
        validate(
            expected_video_id="abc_123",
            expected_channel_id="UC_authorized",
            requested_url="https://www.youtube.com/watch?v=other",
            actual_video_id="abc_123",
            actual_channel_id="UC_authorized",
            canonical_url="https://www.youtube.com/watch?v=abc_123",
        )
    with pytest.raises(RuntimeError, match="channel ID"):
        validate(
            expected_video_id="abc_123",
            expected_channel_id="UC_authorized",
            requested_url="https://www.youtube.com/watch?v=abc_123",
            actual_video_id="abc_123",
            actual_channel_id="UC_other",
            canonical_url="https://www.youtube.com/watch?v=abc_123",
        )
    with pytest.raises(RuntimeError, match="canonical URL"):
        validate(
            expected_video_id="abc_123",
            expected_channel_id="UC_authorized",
            requested_url="https://www.youtube.com/watch?v=abc_123",
            actual_video_id="abc_123",
            actual_channel_id="UC_authorized",
            canonical_url="https://www.youtube.com/watch?v=other",
        )


def test_source_cache_and_staging_are_identity_bound_and_concurrency_safe() -> None:
    source = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    existing = source.split("def _existing_source(", 1)[1].split("@app.function(", 1)[0]
    acquire = source.split("def acquire_source(", 1)[1].split("class VolumeSourceClient", 1)[0]
    writer = source.split("def persist_source_index(", 1)[1].split("@app.function(", 1)[0]

    assert "canonical_identity" in existing
    assert 'str(identity.get("channel_id") or "") != channel_id' in existing
    assert "_youtube_video_id(canonical_url) != video_id" in existing
    assert 'Path(MEDIA_ROOT) / "staging" / video_id / uuid.uuid4().hex' in acquire
    assert "shutil.rmtree(staging, ignore_errors=True)" in acquire
    assert "persist_source_index.remote(evidence)" in acquire
    assert "max_containers=1" in source.split("def persist_source_index(", 1)[0][-300:]
    assert "_atomic_write_json(index_path, evidence)" in writer
    assert '_atomic_write_json(target.with_suffix(".source.json"), evidence)' in writer


def test_volume_source_client_never_discovers_or_downgrades_media() -> None:
    source = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    client = source.split("class VolumeSourceClient", 1)[1].split("def run_full_cycle(", 1)[0]
    assert "no discovery occurs here" in client
    assert "return self.videos" in client
    assert 'evidence.get("quality_policy") != "highest_available_no_transcode"' in client
    assert "mounted source master failed SHA-256 verification" in client
    assert "requested render span is outside master" in client


def test_open_model_image_has_qwen3_vl_and_shared_editorial_budget_contract() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert '"torch==2.8.0"' in source
    assert '"torchvision==0.23.0"' in source
    assert "def _editorial_generation_plan(" in source
    assert "def _editorial_context_limit(" in source
    assert "generation_budget_tokens" in source
    assert "logits_to_keep" in source
    assert '"offloaded"' in source
    assert 'return "balanced" if torch.cuda.device_count() > 1 else "auto"' in source
    assert "device_map=_editorial_device_map_policy()" in source
    assert "editorial model did not distribute across all allocated GPUs" in source
    assert '"editorial_placement"' in source
    assert '"model_gpu_indices": list(model_gpu_indices)' in source
    assert '"application_status": application_status' in source
    assert '"smallest_dynamic_oom_input_tokens"' in source
    assert '"largest_dynamic_good_input_tokens"' in source
    assert '"capacity_repartitionable": repartitionable' in source
    deploy_path = Path(".github/workflows/modal-workers-deploy.yml")
    deploy = deploy_path.read_text(encoding="utf-8")
    parsed_deploy = yaml.safe_load(deploy)
    assert isinstance(parsed_deploy, dict)
    assert "jobs" in parsed_deploy
    assert "deploy-modal-workers" in parsed_deploy["jobs"]
    assert "Verify editorial model spans allocated GPUs" in deploy
    assert "worker.ready.remote()" in deploy
    assert 'placement = runtime.get("editorial_placement")' in deploy
    assert "isinstance(observed, (list, tuple))" in deploy
    assert "normalized != required" in deploy
    assert 'model_bytes.get(f"cuda:{index}")' in deploy
    assert "editorial_output_budget" not in source


def test_modal_vision_derives_runtime_capacity_without_fixed_vram_or_batch_limits() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    visual_source = Path("src/clipper/visual_ai.py").read_text(encoding="utf-8")
    assert "VISION_MAX_PIXELS_PER_FRAME" not in source
    assert 'processor.image_processor.size["longest_edge"]' not in source
    assert 'kwargs["max_memory"]' not in source
    assert "class VisionModel:" in source
    assert "@modal.enter()" in source
    assert "VISION_MAX_NEW_TOKENS" not in source
    assert "VISION_FALLBACK_CONTEXT_LIMIT" not in source
    assert "def _vision_context_limit(" in source
    assert "def _vision_generation_capacity(" in source
    assert "history_output_tokens_per_item" in source
    assert "max_new_tokens=generation_budget" in source
    assert "VisionOutputCapacityError" in source
    assert 'gpu="L4:2"' in source
    assert "SOURCE_POLICY_BATCH_SIZE" not in visual_source
    assert "_is_vision_capacity_error" in visual_source
    assert "_load_capacity_state" in visual_source
    assert "_next_batch_after_success" in visual_source
    assert "_next_batch_after_capacity_failure" in visual_source
    assert "checkpoint_commit" in visual_source
    assert '"HF_HUB_CACHE": HF_CACHE' in source
    assert "TRANSFORMERS_CACHE" not in source


def test_modal_vision_enforces_schema_and_recovers_invalid_json_once() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert "def _vision_contract(task: str)" in source
    assert 'if task == "visual_timeline_scout"' in source
    assert '"decision":"PASS"' in source
    assert "VISION_MAX_ATTEMPTS" not in source
    assert '"event": "vision_generation_start"' in source
    assert '"event": "vision_generation_complete"' in source
    assert '"event": "vision_json_validation"' in source
    assert '"event": "vision_generation_capacity_expand"' in source
    assert "vision output capacity exhausted:" in source
    assert "vision model did not return valid JSON after format recovery" in source
    assert "model.generation_config.temperature = None" in source
    assert "temperature=None" in source


def test_modal_json_parser_accepts_bare_or_fenced_objects_only() -> None:
    parse = _script_function("_json_text")
    assert parse('{"events": []}') == {"events": []}
    assert parse('```json\n{"events": []}\n```') == {"events": []}
    with pytest.raises(json.JSONDecodeError):
        parse("")
    with pytest.raises(ValueError, match="JSON object"):
        parse("[]")


def test_modal_usage_accounts_for_each_requested_gpu_and_device_peak() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert '"gpu_seconds": duration * _gpu_count(gpu)' in source
    assert '"peak_vram_mb_by_device": peak_by_device' in source
    assert "torch.cuda.max_memory_allocated(index)" in source


def test_modal_pyannote_normalizes_container_media_to_duration_safe_wav() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert "def _diarization_audio_source" in source
    assert '"-map",\n                "0:a:0"' in source
    assert '"-ar",\n                "16000"' in source
    assert '"-c:a",\n                "pcm_s16le"' in source
    assert "_diarization_pipeline(str(diarization_source))" in source
    assert "cleanup_source.unlink(missing_ok=True)" in source
    assert "traceback.print_exception(exc)" in source


def test_modal_schema_smoke_covers_only_active_editorial_task_families() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    smoke = source.split("def editorial_schema_smoke(", 1)[1].split("def hf_access_smoke(", 1)[0]
    expected = (
        "source_hazards:smoke",
        "semantic_cores:smoke",
        "narrative_envelope:smoke",
        "quality_windows:smoke",
    )
    assert all(f'"{task}"' in smoke for task in expected)
    for retired in (
        "episode_editorial_profile",
        "story_moments:smoke",
        "clip_concepts",
        "global_concept_comparison",
        "hook_variants:smoke",
        "edit_plans:smoke",
        "boundary_audit:smoke",
    ):
        assert f'"{retired}"' not in smoke


def test_modal_full_cycle_is_quality_derived_and_exact_source_verified() -> None:
    source = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    cycle = source.split("def run_full_cycle(", 1)[1]
    assert "not isinstance(raw_sources, list)" in cycle
    assert "run_full_cycle requires a non-empty sources array" in cycle
    assert '"CLIPPER_COMPUTE_PROFILE": "balanced"' in cycle
    assert '"CLIPPER_VISUAL_SCOUT": "true"' in cycle
    assert '"CLIPPER_VISUAL_REVIEW": "true"' in cycle
    assert 'manifest.get("status") not in {"SUCCESS", "DEGRADED"}' in cycle
    assert 'not render and manifest.get("status") == "FAILED"' in cycle
    assert '"status": "FAIL"' in cycle
    assert '"status_reason": manifest.get("status_reason")' in cycle
    assert '"errors": errors' in cycle
    assert '"run_path": run_relative' in cycle
    assert '"git_sha": git_sha' in cycle
    failure_return = cycle.index('"status": "FAIL"')
    assert cycle.rfind("artifact_volume.commit()", 0, failure_return) != -1
    assert "pipeline did not process the Modal-acquired source hash" in cycle
    assert "execution_id=execution_id.lower()" in cycle
    assert '"eligible_quality_moments"' in cycle
    assert '"review_status": "PENDING_ACTUAL_MP4_REVIEW" if render else "NOT_RENDERED"' in cycle
    assert "recover_finalists" not in source


def test_modal_full_cycle_defaults_to_content_addressed_resume() -> None:
    source = Path("scripts/modal_pipeline.py").read_text(encoding="utf-8")
    cycle = source.split("def run_full_cycle(", 1)[1]
    assert 'fresh_inference = bool(payload.get("fresh_inference", False))' in cycle
    assert '"fresh-inference" if fresh_inference else "content-addressed-resume"' in cycle
    assert "if fresh_inference and resume_from_run_id is not None:" in cycle
    assert 'cache_root=Path(ARTIFACT_ROOT) / "_fresh-cache" / uuid.uuid4().hex' in cycle
    assert 'metadata["execution_mode"] = execution_mode' in cycle
    assert '"mode": "content-addressed-stage-resume"' in cycle
    assert '"execution_mode": execution_mode' in cycle


def test_modal_open_model_deploys_required_speech_handles() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    assert {"transcribe", "align", "diarize"}.issubset(functions)

    transcribe_source = source.split("def transcribe(", 1)[1].split("def align(", 1)[0]
    assert "WhisperModel(" in transcribe_source
    assert "ASR_MODEL_ID," in transcribe_source
    assert "revision=ASR_MODEL_REVISION" in transcribe_source
    assert "compute_type=ASR_COMPUTE_TYPE" in transcribe_source
    assert "word_timestamps=True" in transcribe_source
    assert "vad_filter=True" in transcribe_source


def test_modal_worker_deployment_hydrates_all_required_handles() -> None:
    workflow = Path(".github/workflows/modal-workers-deploy.yml").read_text(encoding="utf-8")
    assert "Verify required deployed model handles resolve" in workflow
    assert "modal.Function.from_name(app, name).hydrate()" in workflow
    assert "modal.Cls.from_name(app, name).hydrate()" in workflow
    for name in (
        "transcribe",
        "align",
        "diarize",
        "editorial_schema_smoke",
        "hf_access_smoke",
        "EditorialModel",
        "VisionModel",
        "VisionModelLarge",
    ):
        assert f'"{name}"' in workflow
