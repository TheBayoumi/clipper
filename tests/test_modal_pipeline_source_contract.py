import ast
import json
from pathlib import Path
from typing import Any

import pytest


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
    assert "source acquisition requires an https video_url" in acquire
    assert "source acquisition requires a safe video_id" in acquire
    assert '"--no-playlist"' in acquire
    assert '"bestvideo+bestaudio/best"' in acquire
    assert '"quality_policy": "highest_available_no_transcode"' in source
    assert 'volume_path = f"/inputs/{digest}{suffix}"' in acquire
    assert "content-addressed source master hash mismatch" in acquire


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
    assert "from clipper.providers.editorial_prompt import editorial_output_budget" in source
    assert "base_budget = editorial_output_budget(payload)" in source
    assert "return min(4096, base_budget * _editorial_recovery_attempt(payload))" in source


def test_modal_vision_bounds_visual_tokens_before_using_two_l4_gpus() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert "VISION_MAX_PIXELS_PER_FRAME = 512 * 28 * 28" in source
    assert 'processor.image_processor.size["longest_edge"] = VISION_MAX_PIXELS_PER_FRAME' in source
    assert 'kwargs["device_map"] = "balanced"' in source
    assert 'kwargs["max_memory"] = {0: "20GiB", 1: "20GiB"}' in source
    assert "input_units + VISION_MAX_NEW_TOKENS > context_limit" in source
    assert 'gpu="L4:2"' in source
    assert '_vision_infer(payload, "Qwen/Qwen3-VL-8B-Instruct", "L4:2")' in source
    assert '"HF_HUB_CACHE": HF_CACHE' in source
    assert "TRANSFORMERS_CACHE" not in source


def test_modal_vision_enforces_schema_and_recovers_invalid_json_once() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert "def _vision_contract(task: str)" in source
    assert 'if task == "visual_timeline_scout"' in source
    assert '"decision":"PASS"' in source
    assert "for attempt in range(1, VISION_MAX_ATTEMPTS + 1)" in source
    assert "vision JSON validation failed:" in source
    assert "vision model did not return valid JSON after recovery" in source
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
    assert '"eligible_quality_moments"' in cycle
    assert '"review_status": "PENDING_ACTUAL_MP4_REVIEW" if render else "NOT_RENDERED"' in cycle
    assert "recover_finalists" not in source
