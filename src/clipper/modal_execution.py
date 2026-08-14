from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

from .brief import load_brief
from .models import CampaignBrief, VideoCandidate
from .rights import assert_campaign_authorized, assert_video_allowed
from .youtube import YouTubeClient

LOGGER = logging.getLogger("clipper")
DEFAULT_MODEL_APP = "clipper-open-editor"
DEFAULT_PIPELINE_APP = "clipper-v10-cycle"
DEFAULT_ARTIFACT_VOLUME = "clipper-v10-artifacts"
_REQUIRED_MODEL_FUNCTIONS = (
    "transcribe",
    "align",
    "diarize",
    "embedding",
    "editorial",
    "vision",
)
_REQUIRED_PIPELINE_FUNCTIONS = ("acquire_source", "run_full_cycle")


def _function(app_name: str, function_name: str) -> Any:
    modal = importlib.import_module("modal")
    handle = modal.Function.from_name(app_name, function_name)
    handle.hydrate()
    return handle


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_script(name: str) -> Path:
    return _repo_root() / "scripts" / name


def _deploy(script: Path) -> None:
    executable = shutil.which("modal")
    if executable is None:
        raise RuntimeError("Modal CLI is required to deploy the default production runtime")
    if not script.is_file():
        raise RuntimeError(
            f"Modal runtime is not deployed and deployment source is unavailable: {script}"
        )
    LOGGER.info("deploying exact-checkout Modal runtime from %s", script)
    subprocess.run(
        [executable, "deploy", str(script)],
        check=True,
        timeout=1800,
        cwd=_repo_root(),
    )


def _hydrate_required(app_name: str, functions: tuple[str, ...]) -> None:
    for function_name in functions:
        try:
            _function(app_name, function_name)
        except Exception as exc:
            message = (
                f"required Modal function {app_name}/{function_name} "
                "is unavailable after deploy"
            )
            raise RuntimeError(message) from exc


def ensure_modal_runtime() -> None:
    """Ensure model workers exist and deploy the exact checkout pipeline worker.

    The source acquisition and full-cycle worker are always deployed from the active checkout so
    `clipper run` cannot silently execute stale pipeline code from an older Modal deployment.
    Model workers are redeployed only when their required functions are unavailable.
    """

    model_app = os.getenv("CLIPPER_MODAL_APP", DEFAULT_MODEL_APP)
    pipeline_app = os.getenv("CLIPPER_V10_MODAL_APP", DEFAULT_PIPELINE_APP)

    model_missing = False
    for function_name in _REQUIRED_MODEL_FUNCTIONS:
        try:
            _function(model_app, function_name)
        except Exception:
            model_missing = True
            break
    if model_missing:
        _deploy(_repo_script("modal_open_models.py"))
        _hydrate_required(model_app, _REQUIRED_MODEL_FUNCTIONS)

    _deploy(_repo_script("modal_v10_cycle.py"))
    _hydrate_required(pipeline_app, _REQUIRED_PIPELINE_FUNCTIONS)


def _authorized_candidates(brief: CampaignBrief) -> list[VideoCandidate]:
    if brief.allowed_video_ids:
        channel_id = brief.source_channel_ids[0] if len(brief.source_channel_ids) == 1 else ""
        candidates = [
            VideoCandidate(
                video_id=video_id,
                title=f"{brief.title} authorized source",
                channel_id=channel_id,
                channel_title="Authorized campaign source",
                url=brief.source_media_urls.get(video_id)
                or f"https://www.youtube.com/watch?v={video_id}",
            )
            for video_id in brief.allowed_video_ids[: brief.source_limit]
        ]
    else:
        candidates = YouTubeClient().discover(brief)

    allowed: list[VideoCandidate] = []
    for candidate in candidates:
        assert_video_allowed(brief, candidate)
        allowed.append(candidate)
    return allowed[: brief.source_limit]


def _acquire_remote_source(function: Any, candidate: VideoCandidate) -> dict[str, Any]:
    payload = {
        "video_id": candidate.video_id,
        "video_url": candidate.url,
    }
    variants = (
        ("cloud:gcp", "cloud", "gcp"),
        ("cloud:aws", "cloud", "aws"),
        ("cloud:oci", "cloud", "oci"),
        ("region:eu", "region", "eu"),
        ("region:ap", "region", "ap"),
        ("region:sa", "region", "sa"),
        ("region:af", "region", "af"),
        ("default", "default", "auto"),
    )
    failures: list[dict[str, str]] = []
    for label, kind, value in variants:
        try:
            if kind == "cloud":
                result = function.with_options(cloud=value, timeout=1800).remote(payload)
            elif kind == "region":
                result = function.with_options(region=value, timeout=1800).remote(payload)
            else:
                result = function.remote(payload)
        except Exception as exc:
            failures.append(
                {
                    "egress": label,
                    "error_type": type(exc).__name__,
                    "error": str(exc)[-2000:],
                }
            )
            continue
        if not isinstance(result, dict):
            failures.append(
                {"egress": label, "error_type": "InvalidResponse", "error": repr(result)[:2000]}
            )
            continue
        if str(result.get("quality_policy")) != "highest_available_no_transcode":
            failures.append(
                {
                    "egress": label,
                    "error_type": "QualityPolicyError",
                    "error": str(result.get("quality_policy")),
                }
            )
            continue
        LOGGER.info(
            "Modal acquired source %s through %s: %s bytes sha256=%s",
            candidate.video_id,
            label,
            result.get("bytes"),
            result.get("sha256"),
        )
        return result
    raise RuntimeError(
        f"Modal source acquisition failed for {candidate.video_id}: "
        + json.dumps(failures, ensure_ascii=False)
    )


def _materialize_remote_run(
    *, artifact_root: Path, volume_name: str, remote_run_path: str
) -> Path:
    executable = shutil.which("modal")
    if executable is None:
        raise RuntimeError("Modal CLI is required to download final production artifacts")
    run_name = Path(remote_run_path.rstrip("/")).name
    if not run_name:
        raise RuntimeError(f"invalid remote run path: {remote_run_path}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    target = artifact_root / run_name
    if target.exists():
        raise RuntimeError(f"refusing to overwrite existing local run: {target}")

    staging = artifact_root / f"._modal-transfer-{uuid.uuid4().hex}"
    staging.mkdir(parents=True, exist_ok=False)
    try:
        subprocess.run(
            [
                executable,
                "volume",
                "get",
                "--force",
                volume_name,
                remote_run_path,
                str(staging),
            ],
            check=True,
            timeout=7200,
        )
        manifests = list(staging.rglob("manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError(
                f"expected one manifest after Modal artifact download, found {len(manifests)}"
            )
        downloaded_run = manifests[0].parent
        if downloaded_run == staging:
            target.mkdir(parents=True)
            for child in list(staging.iterdir()):
                shutil.move(str(child), target / child.name)
        else:
            shutil.move(str(downloaded_run), target)
        return target
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def run_modal_pipeline(
    brief_path: Path,
    *,
    artifact_root: Path,
    resume_from_run_id: str | None,
    render: bool,
    fresh_inference: bool,
) -> Path:
    """Execute the canonical production pipeline entirely inside Modal.

    Large source media is acquired directly by Modal and remains in the Modal media Volume.
    Only the completed run artifacts are materialized to the caller's artifact root.
    """

    brief = load_brief(brief_path)
    assert_campaign_authorized(brief)
    ensure_modal_runtime()

    pipeline_app = os.getenv("CLIPPER_V10_MODAL_APP", DEFAULT_PIPELINE_APP)
    acquire = _function(pipeline_app, "acquire_source")
    runner = _function(pipeline_app, "run_full_cycle")
    candidates = _authorized_candidates(brief)
    if not candidates:
        raise RuntimeError("campaign contains no authorized source candidates")
    if len(candidates) != 1:
        raise RuntimeError(
            "default Modal production execution currently requires source_limit=1; "
            "split multi-source campaigns before execution"
        )

    source = _acquire_remote_source(acquire, candidates[0])
    channel_id = candidates[0].channel_id or (
        brief.source_channel_ids[0] if brief.source_channel_ids else "authorized-source"
    )
    response = runner.remote(
        {
            "source": source,
            "brief_yaml": brief_path.read_text(encoding="utf-8"),
            "channel_id": channel_id,
            "render": render,
            "fresh_inference": fresh_inference,
            "resume_from_run_id": resume_from_run_id,
        }
    )
    if not isinstance(response, dict):
        raise RuntimeError("Modal production runner returned an invalid response")
    remote_run_path = str(response.get("run_path") or "")
    volume_name = str(response.get("run_volume") or DEFAULT_ARTIFACT_VOLUME)
    if not remote_run_path:
        raise RuntimeError(f"Modal production runner returned no run path: {response}")
    return _materialize_remote_run(
        artifact_root=artifact_root,
        volume_name=volume_name,
        remote_run_path=remote_run_path,
    )
