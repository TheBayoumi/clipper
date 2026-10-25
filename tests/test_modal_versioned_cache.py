"""Validate the actual versioned Modal runner without launching a GPU."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from wan_keyframe_contract import prepare  # noqa: E402
from wan_shared_shot_cache import (  # noqa: E402
    copy_verified_keyframes,
    copy_verified_wan,
    local_keyframes_complete,
    local_wan_complete,
    verified_wan_output,
)


@pytest.mark.parametrize(
    "failure",
    (
        "success",
        "fresh_deploy",
        "race",
        "quota",
        "quota_twice",
        "cuda",
        "budget",
        "credits",
        "auth",
        "deploy",
        "timeout",
        "model",
        "quota_warning",
        "quota_probe",
    ),
)
@pytest.mark.parametrize("mode", ("keyframes", "speaking"))
def test_versioned_modal_reuses_deploy_and_fails_over_only_on_quota(
    tmp_path: Path, failure: str, mode: str
) -> None:
    (tmp_path / "scripts").mkdir()
    source_script = (
        tmp_path
        / "scripts"
        / ("modal_qwen_keyframes.py" if mode == "keyframes" else "modal_wan_s2v.py")
    )
    source_script.write_text("app = 'versioned'\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    modal = bin_dir / "modal"
    modal.write_text(
        "#!/usr/bin/env bash\n"
        'echo "modal:$MODAL_TOKEN_ID:$1:$2" >> "$CALLS"\n'
        'if [[ "$FAILURE" == "auth" && "$MODAL_TOKEN_ID" == "primary" '
        '&& "$1" == "app" ]]; then echo "Bad token" >&2; exit 17; fi\n'
        'if [[ "$1" == "function" ]]; then '
        'echo "Unsupported Modal function CLI invoked" >&2; exit 99; fi\n'
        'if [[ "$1" == "deploy" && "$FAILURE" == "deploy" ]]; then '
        'echo "build failed" >&2; exit 17; fi\n'
        'if [[ "$1" == "deploy" && "$FAILURE" == "race" ]]; then '
        'touch "$RACE_DEPLOYED_FILE"; '
        'echo "App deploy name x is already taken by a deployed app." >&2; exit 17; fi\n'
        "exit 0\n"
    )
    modal.chmod(0o755)
    python = bin_dir / "python"
    python.write_text(
        "#!/usr/bin/env bash\n"
        'echo "infer:$MODAL_TOKEN_ID:$1:$2" >> "$CALLS"\n'
        'if [[ "$1" == "scripts/modal_deployment_probe.py" ]]; then\n'
        '  if [[ "$FAILURE" == "quota_probe" && "$MODAL_TOKEN_ID" == "primary" ]]; then '
        'echo "quota exceeded in deployment probe" >&2; exit 17; fi\n'
        '  if [[ "$FAILURE" == "race" ]]; then '
        'test -f "$RACE_DEPLOYED_FILE" || exit 44; exit 0; fi\n'
        '  if [[ "$DEPLOY_STATE" == "missing" ]]; then exit 44; fi\n'
        "  exit 0\n"
        "fi\n"
        'if [[ "$MODAL_TOKEN_ID" == "primary" || '
        '( "$MODAL_TOKEN_ID" == "secondary" && "$FAILURE" == "quota_twice" ) ]]; then\n'
        '  case "$FAILURE" in\n'
        '    quota|quota_twice) echo "QuotaExceededError: quota exceeded" >&2; exit 17 ;;\n'
        '    cuda) echo "CUDA out of memory" >&2; exit 17 ;;\n'
        '    budget) echo "Spending budget limit reached" >&2; exit 17 ;;\n'
        '    credits) echo "Insufficient credits" >&2; exit 17 ;;\n'
        '    timeout) echo "Function timed out" >&2; exit 17 ;;\n'
        '    model) echo "Bad model weights" >&2; exit 17 ;;\n'
        '    quota_warning) echo "Warning: quota exceeded earlier" >&2; '
        'echo "Model exception" >&2; exit 17 ;;\n'
        "  esac\n"
        "fi\n"
        "exit 0\n"
    )
    python.chmod(0o755)
    calls = tmp_path / "calls.log"
    env_file = tmp_path / "github_env"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "MODAL_TOKEN_ID": "primary",
        "MODAL_TOKEN_SECRET": "s1",
        "MODAL_TOKEN_ID_2": "secondary",
        "MODAL_TOKEN_SECRET_2": "s2",
        "MODAL_TOKEN_ID_3": "tertiary",
        "MODAL_TOKEN_SECRET_3": "s3",
        "HF_TOKEN": "synthetic",
        "GITHUB_ENV": str(env_file),
        "CALLS": str(calls),
        "FAILURE": failure,
        "DEPLOY_STATE": "missing" if failure in ("deploy", "fresh_deploy", "race") else "found",
        "RACE_DEPLOYED_FILE": str(tmp_path / "deployed"),
    }
    process = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts/modal_versioned_dispatch.sh"),
            str(source_script.relative_to(tmp_path)),
            "generate_keyframe" if mode == "keyframes" else "generate_s2v",
            mode,
            "--reference",
            "synthetic",
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=12,
        check=False,
    )
    used = calls.read_text()
    if failure in ("success", "fresh_deploy", "race", "quota", "quota_twice"):
        assert process.returncode == 0, process.stderr + process.stdout
        winner = {"success": 1, "fresh_deploy": 1, "race": 1, "quota": 2, "quota_twice": 3}[failure]
        assert f"MODAL_ACCOUNT_SLOT={winner}" in env_file.read_text()
        assert ("infer:secondary:" in used) == (winner >= 2)
        assert ("infer:tertiary:" in used) == (winner == 3)
    else:
        assert process.returncode != 0
        assert "infer:secondary:" not in used
        assert "infer:tertiary:" not in used
        assert not env_file.exists()
    if failure == "success":
        assert "modal:primary:deploy" not in used
    if failure in ("fresh_deploy", "deploy", "race"):
        assert "modal:primary:deploy" in used
    if failure == "race":
        assert "MODAL_DEPLOY_REUSED_AFTER_RACE" in process.stdout
        assert "infer:secondary:" not in used


def test_wan_shared_cache_requires_every_input_hash_and_candidate_status(
    tmp_path: Path,
) -> None:
    movie = tmp_path / "wan-s2v-candidate.mp4"
    evidence = tmp_path / "model-evidence.json"
    movie.write_bytes(b"mp4" * 120000)
    image_sha = "a" * 64
    audio_sha = "b" * 64
    prompt = "Narration scene"
    metadata = {
        "schema": "clipper-wan-s2v-modal-v1",
        "model_id": "Wan-AI/Wan2.2-S2V-14B",
        "model_revision": "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b",
        "image_sha256": image_sha,
        "audio_sha256": audio_sha,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "seed": 724,
        "status": "CANDIDATE_GENERATED_NOT_VISUALLY_APPROVED",
        "creative_review": "PENDING",
        "publish_eligible": False,
        "video_sha256": hashlib.sha256(movie.read_bytes()).hexdigest(),
    }
    evidence.write_text(json.dumps(metadata))
    assert verified_wan_output(movie, evidence, image_sha, audio_sha, prompt, 724)
    assert not verified_wan_output(movie, evidence, image_sha, audio_sha, prompt, 725)
    assert not verified_wan_output(movie, evidence, image_sha, "c" * 64, prompt, 724)
    evidence.write_text(json.dumps({**metadata, "publish_eligible": True}))
    assert not verified_wan_output(movie, evidence, image_sha, audio_sha, prompt, 724)
    evidence.write_text(json.dumps(metadata))
    movie.write_bytes(movie.read_bytes() + b"x")
    assert not verified_wan_output(movie, evidence, image_sha, audio_sha, prompt, 724)


def test_cross_workspace_artifact_reuse_restores_verified_keyframe_and_mp4(
    tmp_path: Path,
) -> None:
    """A backup account reads existing GitHub artifact bytes, not a Modal Volume."""
    source = tmp_path / "hero.webp"
    Image.new("RGB", (512, 768), (20, 45, 60)).save(source)
    board = ROOT / "niches/atomic-habits-approved-storyboard.json"
    req = prepare(source, board, "hook")
    archive = tmp_path / "github-artifact"
    shot_dir = archive / "keyframes"
    shot_dir.mkdir(parents=True)
    frame = shot_dir / "hook.png"
    Image.effect_noise((832, 1472), 60).convert("RGB").save(frame)
    shot_evidence = {
        **req,
        "status": "FRAME_GENERATED_TECHNICAL_CANDIDATE",
        "creative_review": "PENDING",
        "publish_eligible": False,
        "image_sha256": hashlib.sha256(frame.read_bytes()).hexdigest(),
    }
    (shot_dir / "hook.json").write_text(json.dumps(shot_evidence))
    output = tmp_path / "new-github-run"
    args = SimpleNamespace(
        reference=str(source),
        storyboard=str(board),
        shot_ids="hook",
        output=str(output),
    )
    assert copy_verified_keyframes(archive, args) == 1
    assert local_keyframes_complete(args)
    assert copy_verified_keyframes(archive, args) == 0

    narration = tmp_path / "narration.wav"
    narration.write_bytes(b"a" * 50000)
    prompt = "The same host reading a book"
    sha_image = hashlib.sha256(frame.read_bytes()).hexdigest()
    sha_audio = hashlib.sha256(narration.read_bytes()).hexdigest()
    video = archive / "wan-s2v-candidate.mp4"
    video.write_bytes(b"mp4" * 120000)
    evidence = {
        "schema": "clipper-wan-s2v-modal-v1",
        "model_id": "Wan-AI/Wan2.2-S2V-14B",
        "model_revision": "dab4e9c55bbe4c8c4d03db1c2c98c7f0ac9c454b",
        "image_sha256": sha_image,
        "audio_sha256": sha_audio,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "seed": 724,
        "status": "CANDIDATE_GENERATED_NOT_VISUALLY_APPROVED",
        "creative_review": "PENDING",
        "publish_eligible": False,
        "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
    }
    (archive / "model-evidence.json").write_text(json.dumps(evidence))
    wan_args = SimpleNamespace(
        output=str(output),
        reference=str(output / "keyframes/hook.png"),
        audio=str(narration),
        prompt=prompt,
        seed=724,
    )
    assert copy_verified_wan(archive, wan_args) == 1
    assert local_wan_complete(wan_args)
    assert copy_verified_wan(archive, wan_args) == 0
    wan_args.prompt = "different narration direction"
    assert not local_wan_complete(wan_args)
