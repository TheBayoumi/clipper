"""Automatic keyframe generation must be a native Clipper step, not an upload."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location(
    "wan_keyframe_contract", ROOT / "scripts/wan_keyframe_contract.py"
)
assert spec and spec.loader
keyframes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = keyframes
spec.loader.exec_module(keyframes)

BOARD = ROOT / "niches/atomic-habits-approved-storyboard.json"


@pytest.fixture
def source_image(tmp_path: Path) -> Path:
    image = tmp_path / "approved-character.webp"
    Image.new("RGB", (512, 768), (50, 60, 80)).save(image)
    return image


def test_keyframe_per_shot_comes_from_storyboard_and_single_character(
    source_image: Path,
) -> None:
    result = [keyframes.prepare(source_image, BOARD, shot) for shot in keyframes.SHOTS]
    assert len({r["fingerprint"] for r in result}) == 3
    assert len({r["source_image_sha256"] for r in result}) == 1
    assert len({r["storyboard_plan_sha256"] for r in result}) == 1
    assert [x["shot_id"] for x in result] == ["hook", "book_intro", "lesson_example"]
    assert all(x["model_id"] == "Qwen/Qwen-Image-Edit-2511" for x in result)
    assert all(x["publish_eligible"] is False for x in result)
    assert "smartphone" in result[0]["prompt"]
    assert "book" in result[1]["prompt"]
    assert "notebook" in result[2]["prompt"]
    assert "black rectangular" in result[0]["prompt"]


def test_same_shot_is_cached_by_seed_prompt_source_and_model(
    source_image: Path,
) -> None:
    one = keyframes.prepare(source_image, BOARD, "hook", seed=724)
    again = keyframes.prepare(source_image, BOARD, "hook", seed=724)
    changed = keyframes.prepare(source_image, BOARD, "hook", seed=725)
    assert one["fingerprint"] == again["fingerprint"]
    assert one["fingerprint"] != changed["fingerprint"]
    assert one["creative_review"] == "PENDING"
    assert one["render_width"] == 832 and one["render_height"] == 1472


def test_unknown_shot_rejected_and_no_automatic_publication(source_image: Path) -> None:
    with pytest.raises(keyframes.KeyframeContractError, match="three-shot"):
        keyframes.prepare(source_image, BOARD, "random_clip")


def test_model_revision_matches_modal_backend_static_spec() -> None:
    import ast

    source = (ROOT / "scripts/modal_qwen_keyframes.py").read_text()
    tree = ast.parse(source)
    constants = {
        node.targets[0].id: node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert constants["MODEL_ID"] == keyframes.MODEL_ID
    assert constants["MODEL_REVISION"] == keyframes.MODEL_REVISION
    assert constants["SCHEMA"] == keyframes.SCHEMA


def test_storyboard_readonly_inputs(source_image: Path, tmp_path: Path) -> None:
    board = json.loads(BOARD.read_text())
    board["pilot"]["shots"][0]["id"] = "different_hook"
    bogus = tmp_path / "bogus.json"
    bogus.write_text(json.dumps(board))
    with pytest.raises(ValueError, match="shot identity"):
        keyframes.prepare(source_image, bogus, "hook")


def test_modal_fallback_local_cache_is_hash_and_identity_locked(
    source_image: Path, tmp_path: Path
) -> None:
    request = keyframes.prepare(source_image, BOARD, "hook")
    frame = tmp_path / "hook.png"
    manifest = tmp_path / "hook.json"
    Image.effect_noise((832, 1472), 60).convert("RGB").save(frame)
    import hashlib

    evidence = {
        **request,
        "status": "FRAME_GENERATED_TECHNICAL_CANDIDATE",
        "creative_review": "PENDING",
        "publish_eligible": False,
        "image_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
    }
    manifest.write_text(json.dumps(evidence), encoding="utf-8")
    assert keyframes.verified_local_keyframe(frame, manifest, request)

    wrong = dict(request, fingerprint="0" * 64)
    assert not keyframes.verified_local_keyframe(frame, manifest, wrong)
    manifest.write_text(json.dumps({**evidence, "publish_eligible": True}))
    assert not keyframes.verified_local_keyframe(frame, manifest, request)
    manifest.write_text(json.dumps(evidence))
    frame.write_bytes(frame.read_bytes() + b"tampered")
    assert not keyframes.verified_local_keyframe(frame, manifest, request)


def test_versioned_deployment_and_cache_steps_are_isolated_from_credential_checks() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/wan-cinematic-production.yml").read_text())
    steps = workflow["jobs"]["cinematic"]["steps"]
    auth = next(
        item
        for item in steps
        if item.get("name") == "Authenticate each Modal workspace without allocating a GPU"
    )
    assert auth["if"].endswith("inputs.mode == 'credentials'")
    titles = (
        "Generate cinematic keyframes using versioned deployed Modal app",
        "Generate speaking video using versioned deployed Modal app",
    )
    for title in titles:
        step = next(item for item in steps if item.get("name") == title)
        assert "modal_versioned_dispatch.sh" in step["run"]
        assert "wan_shared_shot_cache.py check" in step["run"]
        assert "modal run " not in step["run"]
        assert "inputs.mode == 'credentials'" not in step["if"]


def test_cache_only_wan_replay_cannot_allocate_gpu_on_cache_miss() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/wan-cinematic-production.yml").read_text())
    steps = workflow["jobs"]["cinematic"]["steps"]
    video = next(
        step for step in steps
        if step.get("name") == "Generate speaking video using versioned deployed Modal app"
    )
    run = video["run"]
    assert 'REQUIRE_CACHED_VIDEO: ${{ inputs.require_cached_video }}' in (
        ROOT / ".github/workflows/wan-cinematic-production.yml"
    ).read_text()
    assert 'if [[ "$REQUIRE_CACHED_VIDEO" == "true" ]]; then' in run
    assert run.index('if [[ "$REQUIRE_CACHED_VIDEO"') < run.index(
        "bash scripts/modal_versioned_dispatch.sh"
    )
    for title in (
        "Restore verified cross-run Wan video shot cache",
        "Generate speaking video using versioned deployed Modal app",
        "Audit actual MP4, keyframe provenance, and closed creative gate",
    ):
        step = next(x for x in steps if x.get("name") == title)
        assert "--seed" in step["run"]
