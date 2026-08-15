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


def test_v10_public_youtube_acquisition_activates_bgutil_before_optional_cookies() -> None:
    source = Path("scripts/modal_v10_cycle.py").read_text(encoding="utf-8")
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
    assert "Authenticated yt-dlp cookies are required from this cloud egress." not in source


def test_open_model_image_has_qwen3_vl_torchvision_runtime_contract() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    assert '"torch==2.8.0"' in source
    assert '"torchvision==0.23.0"' in source
    assert 'task.startswith("story_moments:")' in source
    assert "base_budget = 1536" in source
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


def test_modal_cycle_uses_measured_controller_memory_and_single_progress_write() -> None:
    cycle_source = Path("scripts/modal_v10_cycle.py").read_text(encoding="utf-8")
    pipeline_source = Path("src/clipper/pipeline.py").read_text(encoding="utf-8")
    assert "memory=8192" in cycle_source
    callback = pipeline_source.split("def _model_progress(stage: str, event: str) -> None:", 1)[
        1
    ].split("if open_planner is not None:", 1)[0]
    assert 'journal.progress(\n            "model_inference"' in callback
    assert 'journal.start("model_inference"' not in callback


def test_targeted_finalist_recovery_is_fail_closed_and_keeps_source_run_immutable() -> None:
    source = Path("scripts/modal_v10_cycle.py").read_text(encoding="utf-8")
    recovery = source.split("def recover_finalists(", 1)[1].split("def run_full_cycle(", 1)[0]

    assert '_TARGETED_RECOVERY_PLANS = (("c14", "p3"), ("c5", "p1"))' in source
    assert "if requested != _TARGETED_RECOVERY_PLANS:" in recovery
    assert "tracking_transition_sample_times" in recovery
    assert '"source master remained in clipper-media-cache; only the two derived "' in recovery
    assert "partial_run_dir.replace(output_run_dir)" in recovery
    assert "_replace_path_prefix(" in recovery
    assert 'stable_prefix = f"{ARTIFACT_ROOT}/{output_run_id}"' in recovery
    assert "source_run_dir.replace" not in recovery
    assert "shutil.rmtree(source_run_dir" not in recovery


def test_targeted_recovery_launcher_sends_evidence_not_local_frames() -> None:
    source = Path("scripts/run_modal_finalist_recovery.py").read_text(encoding="utf-8")

    assert '"prior_review_recovery": prior_review' in source
    assert '"plan_keys": list(PLAN_KEYS)' in source
    assert '"frames_base64"' not in source
    assert '"c14", "plan_id": "p3"' in source
    assert '"c5", "plan_id": "p1"' in source
