"""Automatic keyframe generation must be a native Clipper step, not an upload."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
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


@pytest.mark.parametrize(
    "stage_name",
    (
        "Generate cinematic keyframes with quota-aware Modal routing",
        "Generate speaking video with quota-aware Modal routing",
    ),
)
@pytest.mark.parametrize(
    "failure_type",
    (
        "quota",
        "quota_twice",
        "cuda",
        "budget",
        "credits",
        "auth",
        "deploy",
        "quota_auth",
        "quota_deploy",
        "timeout",
        "model",
        "quota_warning",
    ),
)
def test_modal_account_failover_is_restricted_to_quota_failures(
    tmp_path: Path, stage_name: str, failure_type: str
) -> None:
    """Exercise the real routing shell with fake Modal, without GPUs or credentials."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/wan-cinematic-production.yml").read_text())
    stage = next(
        step["run"]
        for step in workflow["jobs"]["cinematic"]["steps"]
        if step.get("name") == stage_name
    )
    executable_dir = tmp_path / "bin"
    executable_dir.mkdir()
    modal = executable_dir / "modal"
    modal.write_text(
        "#!/usr/bin/env bash\n"
        'echo "$MODAL_TOKEN_ID $1" >> "$CLIPPER_MODAL_CALLS"\n'
        'if [[ "$MODAL_TOKEN_ID" == "primary" ]]; then\n'
        '  if [[ "$CLIPPER_FAILURE_TYPE" == "quota_auth" && "$1" == "app" ]]; then\n'
        '    echo "Workspace compute quota exceeded" >&2; exit 33\n'
        "  fi\n"
        '  if [[ "$CLIPPER_FAILURE_TYPE" == "auth" && "$1" == "app" ]]; then\n'
        '    echo "AuthenticationError: Invalid Modal token" >&2; exit 33\n'
        "  fi\n"
        '  if [[ "$CLIPPER_FAILURE_TYPE" == "quota_deploy" && "$1" == "deploy" ]]; then\n'
        '    echo "Workspace compute quota exceeded" >&2; exit 33\n'
        "  fi\n"
        '  if [[ "$CLIPPER_FAILURE_TYPE" == "deploy" && "$1" == "deploy" ]]; then\n'
        '    echo "ImageBuilderError: Model dependencies failed" >&2; exit 33\n'
        "  fi\n"
        "fi\n"
        'if [[ "$1" == "run" && ( "$MODAL_TOKEN_ID" == "primary" || '
        '( "$MODAL_TOKEN_ID" == "secondary" && "$CLIPPER_FAILURE_TYPE" == "quota_twice" ) ) ]]; then\n'
        '  case "$CLIPPER_FAILURE_TYPE" in\n'
        '    quota|quota_twice) echo "QuotaExceededError: Workspace compute quota exceeded" >&2 ;;\n'
        '    cuda) echo "CUDA out of memory" >&2 ;;\n'
        '    budget) echo "Account spending budget limit reached" >&2 ;;\n'
        '    credits) echo "Insufficient credits for this request" >&2 ;;\n'
        '    timeout) echo "FunctionTimeoutError: timed out" >&2 ;;\n'
        '    model) echo "RuntimeError: cannot load model weights" >&2 ;;\n'
        '    quota_warning) echo "Warning: quota exceeded yesterday" >&2; '
        'echo "RuntimeError: corrupted model state" >&2 ;;\n'
        "    *) exit 0 ;;\n"
        "  esac\n"
        "  exit 33\n"
        "fi\n"
        "exit 0\n"
    )
    modal.chmod(0o755)
    (tmp_path / "artifacts/cinematic/evidence").mkdir(parents=True)
    env_file = tmp_path / "github_env"
    calls = tmp_path / "modal_calls.log"
    env = {
        **os.environ,
        "PATH": f"{executable_dir}:{os.environ['PATH']}",
        "MODAL_TOKEN_ID": "primary",
        "MODAL_TOKEN_SECRET": "primary-secret",
        "MODAL_TOKEN_ID_2": "secondary",
        "MODAL_TOKEN_SECRET_2": "secondary-secret",
        "MODAL_TOKEN_ID_3": "tertiary",
        "MODAL_TOKEN_SECRET_3": "tertiary-secret",
        "HF_TOKEN": "example-hf-token",
        "SHOT_IDS": "hook",
        "WAN_REFERENCE_FILE": "fake-source.webp",
        "WAN_PROMPT": "test-prompt",
        "GITHUB_ENV": str(env_file),
        "CLIPPER_MODAL_CALLS": str(calls),
        "CLIPPER_FAILURE_TYPE": failure_type,
    }
    process = subprocess.run(
        ["bash", "-c", stage],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )
    invocations = calls.read_text(encoding="utf-8")
    if failure_type in ("quota", "quota_twice"):
        assert process.returncode == 0, process.stderr + process.stdout
        assert "secondary run" in invocations
        winning_slot = 3 if failure_type == "quota_twice" else 2
        assert f"MODAL_ACCOUNT_SLOT={winning_slot}" in env_file.read_text()
        route = next((tmp_path / "artifacts/cinematic/evidence").glob("*-modal-route.json"))
        assert json.loads(route.read_text())["credential_slot"] == winning_slot
        assert ("tertiary run" in invocations) == (winning_slot == 3)
    elif failure_type in ("auth", "deploy", "quota_auth", "quota_deploy"):
        assert process.returncode != 0
        assert "secondary " not in invocations
        assert not env_file.exists()
    else:
        assert process.returncode != 0
        assert "secondary " not in invocations
        assert "tertiary " not in invocations
        assert not env_file.exists()


def test_modal_connection_mode_cannot_trigger_gpu_inference() -> None:
    """The credential check must never fall through to keyframe or S2V GPU steps."""
    workflow = yaml.safe_load((ROOT / ".github/workflows/wan-cinematic-production.yml").read_text())
    steps = workflow["jobs"]["cinematic"]["steps"]
    assert next(
        step["if"]
        for step in steps
        if step.get("name") == "Authenticate each Modal workspace without allocating a GPU"
    ).endswith("inputs.mode == 'credentials'")
    for step in steps:
        condition = step.get("if", "")
        name = step.get("name", "")
        if name.startswith(
            ("Decode one", "Install Modal client", "Generate cinematic", "Verify actual")
        ):
            assert "inputs.mode == 'keyframes'" in condition
            assert "inputs.mode == 'speaking'" in condition
            assert "inputs.mode == 'credentials'" not in condition
