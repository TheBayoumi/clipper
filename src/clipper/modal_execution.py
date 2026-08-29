from __future__ import annotations

import importlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from .brief import load_brief, load_explicit_targets
from .models import VideoCandidate
from .rights import assert_campaign_authorized, assert_video_allowed

LOGGER = logging.getLogger("clipper")
DEFAULT_MODEL_APP = "clipper-open-editor"
DEFAULT_PIPELINE_APP = "clipper-production-pipeline"
DEFAULT_ARTIFACT_VOLUME = "clipper-production-artifacts"
_REQUIRED_MODEL_FUNCTIONS = (
    "transcribe",
    "align",
    "diarize",
    "editorial",
    "vision",
    "hf_access_smoke",
    "deployment_identity",
)
_REQUIRED_PIPELINE_FUNCTIONS = ("acquire_source", "run_full_cycle", "deployment_identity")
_CONTROL_PLANE_ATTEMPTS = 3
_DEPLOY_ATTEMPTS = 3
_RETRY_DELAYS_SECONDS = (2.0, 5.0)


def _exception_class_names(exc: BaseException) -> set[str]:
    return {cls.__name__ for cls in type(exc).__mro__}


def _is_modal_not_found(exc: BaseException) -> bool:
    return "NotFoundError" in _exception_class_names(exc)


def _is_retryable_modal_control_plane_error(exc: BaseException) -> bool:
    names = _exception_class_names(exc)
    return bool(names & {"ServiceError", "InternalFailure", "TimeoutError"})


def _retry_delay(attempt: int) -> float:
    index = max(0, min(attempt - 1, len(_RETRY_DELAYS_SECONDS) - 1))
    return _RETRY_DELAYS_SECONDS[index]


def _function(app_name: str, function_name: str) -> Any:
    """Hydrate a deployed Modal Function with bounded retries for control-plane flakes."""
    modal = importlib.import_module("modal")
    for attempt in range(1, _CONTROL_PLANE_ATTEMPTS + 1):
        handle = modal.Function.from_name(app_name, function_name)
        try:
            handle.hydrate()
        except Exception as exc:
            if _is_retryable_modal_control_plane_error(exc) and attempt < _CONTROL_PLANE_ATTEMPTS:
                delay = _retry_delay(attempt)
                LOGGER.warning(
                    "Modal control-plane request failed while hydrating %s/%s "
                    "(%s: %s); retrying in %.1fs (%d/%d)",
                    app_name,
                    function_name,
                    type(exc).__name__,
                    str(exc),
                    delay,
                    attempt + 1,
                    _CONTROL_PLANE_ATTEMPTS,
                )
                time.sleep(delay)
                continue
            raise
        return handle
    raise RuntimeError(f"failed to hydrate Modal function {app_name}/{function_name}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _repo_script(name: str) -> Path:
    return _repo_root() / "scripts" / name


def _local_git_sha() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_repo_root(),
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    sha = completed.stdout.strip().lower()
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise RuntimeError(f"local checkout did not resolve to a full git SHA: {sha!r}")
    return sha


def _verify_deployed_runtime_sha(*, model_app: str, pipeline_app: str) -> str:
    expected = _local_git_sha()
    for label, app_name in (("model", model_app), ("pipeline", pipeline_app)):
        identity = _function(app_name, "deployment_identity").remote()
        if not isinstance(identity, dict):
            raise RuntimeError(f"{label} deployment identity is not an object: {identity!r}")
        deployed = str(identity.get("deployed_git_sha") or "").strip().lower()
        if deployed != expected:
            raise RuntimeError(
                f"{label} deployed SHA mismatch: expected={expected} "
                f"deployed={deployed or '<missing>'}"
            )
    return expected


def _positive_budget(value: float, *, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return resolved


def _invoke_remote_with_budget(
    function: Any,
    request: dict[str, Any],
    *,
    max_gpu_seconds: float,
    max_estimated_usd: float,
) -> object:
    max_gpu_seconds = _positive_budget(max_gpu_seconds, name="max_gpu_seconds")
    max_estimated_usd = _positive_budget(max_estimated_usd, name="max_estimated_usd")
    poll_seconds = float(os.getenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "5"))
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("CLIPPER_MODAL_SPY_POLL_SECONDS must be finite and positive")

    call = function.spawn(request)
    call.hydrate()
    started = time.monotonic()

    def budget_usage() -> tuple[float, float]:
        elapsed = max(0.0, time.monotonic() - started)
        return elapsed * 2.0, elapsed * 0.000444

    while True:
        gpu_seconds, estimated_usd = budget_usage()
        if gpu_seconds >= max_gpu_seconds or estimated_usd >= max_estimated_usd:
            call.cancel(terminate_containers=False)
            raise RuntimeError(
                "CLI production call exceeded its in-flight compute budget: "
                f"gpu_seconds={gpu_seconds:.3f}/{max_gpu_seconds:.3f} "
                f"estimated_usd={estimated_usd:.6f}/{max_estimated_usd:.6f}"
            )
        try:
            result = call.get(timeout=poll_seconds)
        except TimeoutError:
            continue

        gpu_seconds, estimated_usd = budget_usage()
        if gpu_seconds >= max_gpu_seconds or estimated_usd >= max_estimated_usd:
            raise RuntimeError(
                "CLI production call exceeded its compute budget on the final poll: "
                f"gpu_seconds={gpu_seconds:.3f}/{max_gpu_seconds:.3f} "
                f"estimated_usd={estimated_usd:.6f}/{max_estimated_usd:.6f}"
            )
        return result


def _deploy(script: Path) -> None:
    executable = shutil.which("modal")
    if executable is None:
        raise RuntimeError("Modal CLI is required to deploy a missing production runtime")
    if not script.is_file():
        raise RuntimeError(
            f"Modal runtime is not deployed and deployment source is unavailable: {script}"
        )
    command = [executable, "deploy", str(script)]
    deploy_env = os.environ.copy()
    deploy_env["CLIPPER_DEPLOYED_GIT_SHA"] = _local_git_sha()
    for attempt in range(1, _DEPLOY_ATTEMPTS + 1):
        LOGGER.info(
            "deploying missing Modal runtime from %s (attempt %d/%d)",
            script,
            attempt,
            _DEPLOY_ATTEMPTS,
        )
        try:
            subprocess.run(
                command,
                check=True,
                timeout=1800,
                cwd=_repo_root(),
                env=deploy_env,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            if attempt >= _DEPLOY_ATTEMPTS:
                raise
            delay = _retry_delay(attempt)
            LOGGER.warning(
                "Modal deployment command failed (%s); retrying in %.1fs (%d/%d)",
                type(exc).__name__,
                delay,
                attempt + 1,
                _DEPLOY_ATTEMPTS,
            )
            time.sleep(delay)
            continue
        return


def _hydrate_required(app_name: str, functions: tuple[str, ...]) -> None:
    for function_name in functions:
        try:
            _function(app_name, function_name)
        except Exception as exc:
            raise RuntimeError(
                f"required Modal function {app_name}/{function_name} is unavailable after runtime "
                f"repair ({type(exc).__name__}: {exc})"
            ) from exc


def _ensure_deployed_runtime(
    *,
    app_name: str,
    functions: tuple[str, ...],
    deployment_script: str,
) -> None:
    missing_function: str | None = None
    for function_name in functions:
        try:
            _function(app_name, function_name)
        except Exception as exc:
            if _is_modal_not_found(exc):
                missing_function = function_name
                break
            raise RuntimeError(
                "Modal control-plane validation failed while checking "
                f"{app_name}/{function_name} ({type(exc).__name__}: {exc}); "
                "refusing to misclassify this as a missing deployment"
            ) from exc
    if missing_function is None:
        LOGGER.info("attached to deployed Modal runtime %s", app_name)
        return
    LOGGER.info(
        "Modal runtime %s/%s is not deployed; repairing from local checkout",
        app_name,
        missing_function,
    )
    _deploy(_repo_script(deployment_script))
    _hydrate_required(app_name, functions)


def _validate_model_access(app_name: str) -> None:
    smoke = _function(app_name, "hf_access_smoke")
    try:
        result = smoke.remote()
    except Exception as exc:
        raise RuntimeError(
            "Modal Hugging Face access preflight failed for "
            f"{app_name}/hf_access_smoke ({type(exc).__name__}: {exc})"
        ) from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError(
            f"Modal Hugging Face access preflight returned an invalid result: {result!r}"
        )
    LOGGER.info(
        "Modal Hugging Face access verified for %s at revision %s",
        result.get("model_id", "pyannote/speaker-diarization-community-1"),
        result.get("revision", "unknown"),
    )


def ensure_modal_runtime() -> None:
    """Attach to deployed runtimes; redeploy only genuinely missing functions."""
    model_app = os.getenv("CLIPPER_MODAL_APP", DEFAULT_MODEL_APP)
    pipeline_app = os.getenv("CLIPPER_MODAL_PIPELINE_APP", DEFAULT_PIPELINE_APP)
    _ensure_deployed_runtime(
        app_name=model_app,
        functions=_REQUIRED_MODEL_FUNCTIONS,
        deployment_script="modal_open_models.py",
    )
    _ensure_deployed_runtime(
        app_name=pipeline_app,
        functions=_REQUIRED_PIPELINE_FUNCTIONS,
        deployment_script="modal_pipeline.py",
    )
    _validate_model_access(model_app)


def _explicit_candidates(brief_path: Path) -> list[VideoCandidate]:
    """Resolve exactly the campaign's explicit targets; production never performs discovery."""
    brief = load_brief(brief_path)
    specs = load_explicit_targets(brief_path)
    candidates = [
        VideoCandidate(
            video_id=spec.video_id,
            title=f"{brief.title} explicit target",
            channel_id=spec.channel_id,
            channel_title="Authorized explicit target",
            url=spec.media_url or spec.url,
        )
        for spec in specs
    ]
    for candidate in candidates:
        assert_video_allowed(brief, candidate)
    return candidates


def _acquire_remote_source(
    function: Any,
    candidate: VideoCandidate,
    *,
    expected_git_sha: str,
) -> dict[str, Any]:
    if len(expected_git_sha) != 40 or any(
        character not in "0123456789abcdef" for character in expected_git_sha.lower()
    ):
        raise ValueError("source acquisition requires a full expected_git_sha")
    payload = {
        "video_id": candidate.video_id,
        "channel_id": candidate.channel_id,
        "video_url": candidate.url,
        "expected_git_sha": expected_git_sha.lower(),
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
        if (
            str(result.get("video_id") or "") != candidate.video_id
            or str(result.get("channel_id") or "") != candidate.channel_id
        ):
            failures.append(
                {
                    "egress": label,
                    "error_type": "SourceIdentityError",
                    "error": (
                        f"expected={candidate.video_id}/{candidate.channel_id} "
                        f"actual={result.get('video_id')}/{result.get('channel_id')}"
                    ),
                }
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


def _materialize_remote_run(*, artifact_root: Path, volume_name: str, remote_run_path: str) -> Path:
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
        modal_env = os.environ.copy()
        modal_env["PYTHONUTF8"] = "1"
        modal_env["PYTHONIOENCODING"] = "utf-8"
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
            env=modal_env,
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
    max_gpu_seconds: float = 21600.0,
    max_estimated_usd: float = 10.0,
) -> Path:
    """Execute every explicit campaign target inside one content-addressed Modal run."""
    brief = load_brief(brief_path)
    assert_campaign_authorized(brief)
    ensure_modal_runtime()

    candidates = _explicit_candidates(brief_path)
    if not candidates:
        raise RuntimeError("campaign contains no explicit authorized targets")

    model_app = os.getenv("CLIPPER_MODAL_APP", DEFAULT_MODEL_APP)
    pipeline_app = os.getenv("CLIPPER_MODAL_PIPELINE_APP", DEFAULT_PIPELINE_APP)
    verified_git_sha = _verify_deployed_runtime_sha(
        model_app=model_app,
        pipeline_app=pipeline_app,
    )
    execution_id = uuid.uuid4().hex
    max_gpu_seconds = _positive_budget(max_gpu_seconds, name="max_gpu_seconds")
    max_estimated_usd = _positive_budget(max_estimated_usd, name="max_estimated_usd")
    acquire = _function(pipeline_app, "acquire_source")
    runner = _function(pipeline_app, "run_full_cycle")

    sources = [
        _acquire_remote_source(
            acquire,
            candidate,
            expected_git_sha=verified_git_sha,
        )
        for candidate in candidates
    ]
    source_payloads = [
        {
            "evidence": evidence,
            "video_id": candidate.video_id,
            "channel_id": candidate.channel_id,
            "canonical_url": candidate.url,
        }
        for candidate, evidence in zip(candidates, sources, strict=True)
    ]
    request = {
        "sources": source_payloads,
        "brief_yaml": brief_path.read_text(encoding="utf-8"),
        "render": render,
        "fresh_inference": fresh_inference,
        "resume_from_run_id": resume_from_run_id,
        "git_sha": verified_git_sha,
        "execution_id": execution_id,
        "max_gpu_seconds": max_gpu_seconds,
        "max_estimated_usd": max_estimated_usd,
    }
    response = _invoke_remote_with_budget(
        runner,
        request,
        max_gpu_seconds=max_gpu_seconds,
        max_estimated_usd=max_estimated_usd,
    )
    if not isinstance(response, dict):
        raise RuntimeError("Modal production runner returned an invalid response")
    if response.get("execution_id") != execution_id:
        raise RuntimeError(
            "Modal production runner returned a mismatched execution ID: "
            f"{response.get('execution_id')!r}"
        )
    if str(response.get("deployed_git_sha") or "").lower() != verified_git_sha:
        raise RuntimeError(
            "Modal production runner returned a mismatched deployed SHA: "
            f"{response.get('deployed_git_sha')!r}"
        )
    remote_run_path = str(response.get("run_path") or "")
    volume_name = str(response.get("run_volume") or DEFAULT_ARTIFACT_VOLUME)
    if not remote_run_path:
        raise RuntimeError(f"Modal production runner returned no run path: {response}")
    return _materialize_remote_run(
        artifact_root=artifact_root,
        volume_name=volume_name,
        remote_run_path=remote_run_path,
    )
