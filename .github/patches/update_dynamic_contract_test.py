from pathlib import Path

path = Path("tests/test_modal_pipeline_source_contract.py")
text = path.read_text(encoding="utf-8")

old = '''def test_modal_vision_bounds_visual_tokens_before_using_two_l4_gpus() -> None:
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
'''

new = '''def test_modal_vision_derives_runtime_capacity_without_fixed_vram_or_batch_limits() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    visual_source = Path("src/clipper/visual_ai.py").read_text(encoding="utf-8")
    assert "VISION_MAX_PIXELS_PER_FRAME" not in source
    assert 'processor.image_processor.size["longest_edge"]' not in source
    assert 'kwargs["max_memory"]' not in source
    assert "class VisionModel:" in source
    assert "@modal.enter()" in source
    assert "input_units + VISION_MAX_NEW_TOKENS > context_limit" in source
    assert 'gpu="L4:2"' in source
    assert "SOURCE_POLICY_BATCH_SIZE" not in visual_source
    assert "_is_vision_capacity_error" in visual_source
    assert "_load_capacity_state" in visual_source
    assert "_next_batch_after_success" in visual_source
    assert "_next_batch_after_capacity_failure" in visual_source
    assert "checkpoint_commit" in visual_source
    assert '"HF_HUB_CACHE": HF_CACHE' in source
    assert "TRANSFORMERS_CACHE" not in source
'''

if old not in text:
    raise RuntimeError("stale fixed-capacity source-contract test block not found")
path.write_text(text.replace(old, new, 1), encoding="utf-8")
