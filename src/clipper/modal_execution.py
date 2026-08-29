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
from dataclasses import dataclass
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
    "editorial_schema_smoke",
    "hf_access_smoke",
    "deployment_identity",
)
_REQUIRED_MODEL_CLASSES = ("EditorialModel", "VisionModel", "VisionModelLarge")
_REQUIRED_PIPELINE_FUNCTIONS = ("acquire_source", "run_full_cycle", "deployment_identity")
_MODAL_ROOT_GPU_COUNT = 2.0
_MODAL_ROOT_ESTIMATED_USD_PER_SECOND = 0.000444
# Conservative 4-GiB CPU acquisition rate, including the highest configured
# regional premium. GPU cost is zero because acquire_source has no GPU request.
_MODAL_ACQUISITION_ESTIMATED_USD_PER_SECOND = 0.000019
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


def _hydrate_named_handle(app_name: str, handle_name: str, *, kind: str) -> Any:
    """Hydrate a deployed Modal Function or Cls with bounded control-plane retries."""
    modal = importlib.import_module("modal")
    namespace = modal.Function if kind == "function" else modal.Cls
    for attempt in range(1, _CONTROL_PLANE_ATTEMPTS + 1):
        handle = namespace.from_name(app_name, handle_name)
        try:
            handle.hydrate()
        except Exception as exc:
            if _is_retryable_modal_control_plane_error(exc) and attempt < _CONTROL_PLANE_ATTEMPTS:
                delay = _retry_delay(attempt)
                LOGGER.warning(
                    "Modal control-plane request failed while hydrating %s %s/%s "
                    "(%s: %s); retrying in %.1fs (%d/%d)",
                    kind,
                    app_name,
                    handle_name,
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
    raise RuntimeError(f"failed to hydrate Modal {kind} {app_name}/{handle_name}")


def _function(app_name: str, function_name: str) -> Any:
    return _hydrate_named_handle(app_name, function_name, kind="function")


def _class(app_name: str, class_name: str) -> Any:
    return _hydrate_named_handle(app_name, class_name, kind="class")


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


def _validated_source_sha(value: object, *, origin: str) -> str:
    sha = str(value or "").strip().lower()
    if len(sha) != 40 or any(character not in "0123456789abcdef" for character in sha):
        raise RuntimeError(f"{origin} did not provide a full immutable source SHA: {sha!r}")
    return sha


def _embedded_source_sha() -> str | None:
    candidates: list[tuple[str, str]] = []
    environment_sha = os.getenv("CLIPPER_SOURCE_SHA", "").strip()
    if environment_sha:
        candidates.append(
            (
                "CLIPPER_SOURCE_SHA",
                _validated_source_sha(environment_sha, origin="CLIPPER_SOURCE_SHA"),
            )
        )
    embedded_path = _repo_root() / ".clipper-source-sha"
    if embedded_path.is_file():
        candidates.append(
            (
                str(embedded_path),
                _validated_source_sha(
                    embedded_path.read_text(encoding="utf-8"), origin=str(embedded_path)
                ),
            )
        )
    if not candidates:
        return None
    values = {value for _origin, value in candidates}
    if len(values) != 1:
        raise RuntimeError(f"embedded source SHA evidence disagrees: {candidates!r}")
    return candidates[0][1]


def _runtime_source_sha() -> str:
    embedded = _embedded_source_sha()
    try:
        checkout = _local_git_sha()
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        if embedded is not None:
            return embedded
        raise RuntimeError(
            "runtime source SHA is unavailable: no embedded image SHA and no Git checkout"
        ) from exc
    if embedded is not None and embedded != checkout:
        raise RuntimeError(f"runtime source SHA mismatch: embedded={embedded} checkout={checkout}")
    return checkout


def _verify_deployed_runtime_sha(*, model_app: str, pipeline_app: str) -> str:
    expected = _runtime_source_sha()
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


class ProductionBudgetExceeded(RuntimeError):
    pass


@dataclass(slots=True)
class _BudgetLedger:
    max_gpu_seconds: float
    max_estimated_usd: float
    gpu_seconds: float = 0.0
    estimated_usd: float = 0.0

    def __post_init__(self) -> None:
        self.max_gpu_seconds = _positive_budget(self.max_gpu_seconds, name="max_gpu_seconds")
        self.max_estimated_usd = _positive_budget(self.max_estimated_usd, name="max_estimated_usd")

    @staticmethod
    def _rate(value: float, *, name: str) -> float:
        resolved = float(value)
        if not math.isfinite(resolved) or resolved < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        return resolved

    def projected_usage(
        self, elapsed_seconds: float, *, gpu_count: float, estimated_usd_per_second: float
    ) -> tuple[float, float]:
        elapsed = max(0.0, float(elapsed_seconds))
        gpu_rate = self._rate(gpu_count, name="gpu_count")
        cost_rate = self._rate(estimated_usd_per_second, name="estimated_usd_per_second")
        return (self.gpu_seconds + elapsed * gpu_rate, self.estimated_usd + elapsed * cost_rate)

    def remaining_wall_seconds(
        self, elapsed_seconds: float, *, gpu_count: float, estimated_usd_per_second: float
    ) -> float:
        projected_gpu, projected_cost = self.projected_usage(
            elapsed_seconds,
            gpu_count=gpu_count,
            estimated_usd_per_second=estimated_usd_per_second,
        )
        gpu_rate = self._rate(gpu_count, name="gpu_count")
        cost_rate = self._rate(estimated_usd_per_second, name="estimated_usd_per_second")
        limits: list[float] = []
        if gpu_rate > 0:
            limits.append((self.max_gpu_seconds - projected_gpu) / gpu_rate)
        elif projected_gpu > self.max_gpu_seconds:
            return 0.0
        if cost_rate > 0:
            limits.append((self.max_estimated_usd - projected_cost) / cost_rate)
        elif projected_cost > self.max_estimated_usd:
            return 0.0
        return min(limits) if limits else float("inf")

    def charge(
        self, elapsed_seconds: float, *, gpu_count: float, estimated_usd_per_second: float
    ) -> None:
        elapsed = max(0.0, float(elapsed_seconds))
        self.gpu_seconds += elapsed * self._rate(gpu_count, name="gpu_count")
        self.estimated_usd += elapsed * self._rate(
            estimated_usd_per_second, name="estimated_usd_per_second"
        )

    def remaining_budgets(self) -> tuple[float, float]:
        return (
            max(0.0, self.max_gpu_seconds - self.gpu_seconds),
            max(0.0, self.max_estimated_usd - self.estimated_usd),
        )

    def to_dict(self) -> dict[str, float]:
        remaining_gpu, remaining_cost = self.remaining_budgets()
        return {
            "max_gpu_seconds": self.max_gpu_seconds,
            "max_estimated_usd": self.max_estimated_usd,
            "gpu_seconds": self.gpu_seconds,
            "estimated_usd": self.estimated_usd,
            "remaining_gpu_seconds": remaining_gpu,
            "remaining_estimated_usd": remaining_cost,
        }


def _invoke_remote_with_budget(
    function: Any,
    request: dict[str, Any],
    *,
    max_gpu_seconds: float | None = None,
    max_estimated_usd: float | None = None,
    budget: _BudgetLedger | None = None,
    gpu_count: float = _MODAL_ROOT_GPU_COUNT,
    estimated_usd_per_second: float = _MODAL_ROOT_ESTIMATED_USD_PER_SECOND,
) -> object:
    if budget is None:
        if max_gpu_seconds is None or max_estimated_usd is None:
            raise ValueError("remote invocation requires a complete compute budget")
        budget = _BudgetLedger(max_gpu_seconds, max_estimated_usd)
    poll_seconds = float(os.getenv("CLIPPER_MODAL_SPY_POLL_SECONDS", "5"))
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("CLIPPER_MODAL_SPY_POLL_SECONDS must be finite and positive")
    budget._rate(gpu_count, name="gpu_count")
    budget._rate(estimated_usd_per_second, name="estimated_usd_per_second")
    call = function.spawn(request)
    started = time.monotonic()
    terminal_result = False
    budget_charged = False

    def budget_usage() -> tuple[float, float]:
        return budget.projected_usage(
            max(0.0, time.monotonic() - started),
            gpu_count=gpu_count,
            estimated_usd_per_second=estimated_usd_per_second,
        )

    def remaining_budget_wall_seconds() -> float:
        return budget.remaining_wall_seconds(
            max(0.0, time.monotonic() - started),
            gpu_count=gpu_count,
            estimated_usd_per_second=estimated_usd_per_second,
        )

    def charge_elapsed() -> None:
        nonlocal budget_charged
        if budget_charged:
            return
        budget.charge(
            max(0.0, time.monotonic() - started),
            gpu_count=gpu_count,
            estimated_usd_per_second=estimated_usd_per_second,
        )
        budget_charged = True

    try:
        call.hydrate()
        while True:
            gpu_seconds, estimated_usd = budget_usage()
            remaining_seconds = remaining_budget_wall_seconds()
            if remaining_seconds <= 0:
                raise ProductionBudgetExceeded(
                    "CLI production call exceeded its in-flight compute budget: "
                    f"gpu_seconds={gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                    f"estimated_usd={estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
                )
            try:
                result = call.get(timeout=min(poll_seconds, remaining_seconds))
            except TimeoutError:
                continue
            terminal_result = True
            charge_elapsed()
            gpu_seconds = budget.gpu_seconds
            estimated_usd = budget.estimated_usd
            if gpu_seconds > budget.max_gpu_seconds or estimated_usd > budget.max_estimated_usd:
                raise ProductionBudgetExceeded(
                    "CLI production call exceeded its compute budget on the final poll: "
                    f"gpu_seconds={gpu_seconds:.3f}/{budget.max_gpu_seconds:.3f} "
                    f"estimated_usd={estimated_usd:.6f}/{budget.max_estimated_usd:.6f}"
                )
            return result
    finally:
        charge_elapsed()
        if not terminal_result:
            try:
                call.cancel(terminate_containers=False)
            except Exception:
                LOGGER.exception("failed to cancel nonterminal Modal production call")


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


def _hydrate_required(
    app_name: str,
    functions: tuple[str, ...],
    classes: tuple[str, ...] = (),
) -> None:
    for kind, names, resolver in (
        ("function", functions, _function),
        ("class", classes, _class),
    ):
        for handle_name in names:
            try:
                resolver(app_name, handle_name)
            except Exception as exc:
                raise RuntimeError(
                    f"required Modal {kind} {app_name}/{handle_name} is unavailable after "
                    f"runtime repair ({type(exc).__name__}: {exc})"
                ) from exc


def _ensure_deployed_runtime(
    *,
    app_name: str,
    functions: tuple[str, ...],
    classes: tuple[str, ...] = (),
    deployment_script: str,
) -> None:
    missing_handle: str | None = None
    for kind, names, resolver in (
        ("function", functions, _function),
        ("class", classes, _class),
    ):
        for handle_name in names:
            try:
                resolver(app_name, handle_name)
            except Exception as exc:
                if _is_modal_not_found(exc):
                    missing_handle = f"{kind}:{handle_name}"
                    break
                raise RuntimeError(
                    "Modal control-plane validation failed while checking "
                    f"{kind} {app_name}/{handle_name} ({type(exc).__name__}: {exc}); "
                    "refusing to misclassify this as a missing deployment"
                ) from exc
        if missing_handle is not None:
            break
    if missing_handle is None:
        LOGGER.info("attached to deployed Modal runtime %s", app_name)
        return
    LOGGER.info(
        "Modal runtime %s/%s is not deployed; repairing from local checkout",
        app_name,
        missing_handle,
    )
    _deploy(_repo_script(deployment_script))
    _hydrate_required(app_name, functions, classes)


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
        classes=_REQUIRED_MODEL_CLASSES,
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
            url=spec.url,
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
    budget: _BudgetLedger | None = None,
    attempt_evidence: list[dict[str, object]] | None = None,
    execution_id: str | None = None,
) -> dict[str, Any]:
    if len(expected_git_sha) != 40 or any(
        character not in "0123456789abcdef" for character in expected_git_sha.lower()
    ):
        raise ValueError("source acquisition requires a full expected_git_sha")
    if budget is None:
        budget = _BudgetLedger(21600.0, 10.0)
    payload: dict[str, Any] = {
        "video_id": candidate.video_id,
        "channel_id": candidate.channel_id,
        "video_url": candidate.url,
        "expected_git_sha": expected_git_sha.lower(),
    }
    if execution_id:
        payload["execution_id"] = execution_id
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

    def record_failure(
        *,
        label: str,
        error_type: str,
        error: object,
        phase: str,
        before_gpu: float,
        before_cost: float,
    ) -> None:
        rendered_error = str(error)[-2000:]
        failures.append(
            {
                "egress": label,
                "error_type": error_type,
                "error": rendered_error,
            }
        )
        if attempt_evidence is not None:
            attempt_evidence.append(
                {
                    "egress": label,
                    "status": "FAIL",
                    "phase": phase,
                    "error_type": error_type,
                    "error": rendered_error,
                    "estimated_usd": budget.estimated_usd - before_cost,
                    "gpu_seconds": budget.gpu_seconds - before_gpu,
                }
            )

    for label, kind, value in variants:
        before_gpu, before_cost = budget.gpu_seconds, budget.estimated_usd
        try:
            if kind == "cloud":
                variant = function.with_options(cloud=value, timeout=1800)
            elif kind == "region":
                variant = function.with_options(region=value, timeout=1800)
            else:
                variant = function.with_options(timeout=1800)
        except Exception as exc:
            record_failure(
                label=label,
                error_type=type(exc).__name__,
                error=exc,
                phase="configuration",
                before_gpu=before_gpu,
                before_cost=before_cost,
            )
            continue

        try:
            result = _invoke_remote_with_budget(
                variant,
                payload,
                budget=budget,
                gpu_count=0.0,
                estimated_usd_per_second=_MODAL_ACQUISITION_ESTIMATED_USD_PER_SECOND,
            )
        except ProductionBudgetExceeded:
            if attempt_evidence is not None:
                attempt_evidence.append(
                    {
                        "egress": label,
                        "status": "BUDGET_EXCEEDED",
                        "phase": "invoke",
                        "estimated_usd": budget.estimated_usd - before_cost,
                        "gpu_seconds": budget.gpu_seconds - before_gpu,
                    }
                )
            raise
        except Exception as exc:
            record_failure(
                label=label,
                error_type=type(exc).__name__,
                error=exc,
                phase="invoke",
                before_gpu=before_gpu,
                before_cost=before_cost,
            )
            continue

        if not isinstance(result, dict):
            record_failure(
                label=label,
                error_type="InvalidResponse",
                error=repr(result),
                phase="validation",
                before_gpu=before_gpu,
                before_cost=before_cost,
            )
            continue
        if (
            str(result.get("video_id") or "") != candidate.video_id
            or str(result.get("channel_id") or "") != candidate.channel_id
        ):
            record_failure(
                label=label,
                error_type="SourceIdentityError",
                error=(
                    f"expected={candidate.video_id}/{candidate.channel_id} "
                    f"actual={result.get('video_id')}/{result.get('channel_id')}"
                ),
                phase="validation",
                before_gpu=before_gpu,
                before_cost=before_cost,
            )
            continue
        if str(result.get("quality_policy")) != "highest_available_no_transcode":
            record_failure(
                label=label,
                error_type="QualityPolicyError",
                error=result.get("quality_policy"),
                phase="validation",
                before_gpu=before_gpu,
                before_cost=before_cost,
            )
            continue
        if attempt_evidence is not None:
            attempt_evidence.append(
                {
                    "egress": label,
                    "status": "PASS",
                    "phase": "complete",
                    "estimated_usd": budget.estimated_usd - before_cost,
                    "gpu_seconds": budget.gpu_seconds - before_gpu,
                }
            )
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
    budget = _BudgetLedger(max_gpu_seconds, max_estimated_usd)
    acquire = _function(pipeline_app, "acquire_source")
    runner = _function(pipeline_app, "run_full_cycle")

    sources = [
        _acquire_remote_source(
            acquire,
            candidate,
            expected_git_sha=verified_git_sha,
            budget=budget,
            execution_id=execution_id,
        )
        for candidate in candidates
    ]
    remaining_gpu_seconds, remaining_estimated_usd = budget.remaining_budgets()
    if remaining_gpu_seconds <= 0 or remaining_estimated_usd <= 0:
        raise ProductionBudgetExceeded(
            "source acquisition exhausted the production compute budget before pipeline execution"
        )
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
        "max_gpu_seconds": remaining_gpu_seconds,
        "max_estimated_usd": remaining_estimated_usd,
    }
    response = _invoke_remote_with_budget(
        runner,
        request,
        budget=budget,
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
