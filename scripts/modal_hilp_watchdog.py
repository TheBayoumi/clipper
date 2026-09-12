from __future__ import annotations

import json
import math
import os
import signal
import threading
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from modal_execution_spy import ModalExecutionSpy

from clipper.modal_execution import (
    ProductionCallNotTerminated,
    _acquire_remote_source,
    _BudgetLedger,
    _spawn_recoverable_modal_call,
)
from clipper.models import VideoCandidate


class ProductionCallCancelled(RuntimeError):
    pass


_CANCEL_CONFIRM_TERMINAL_MODAL_ERRORS = frozenset(
    {
        "DeserializationError",
        "ExecutionError",
        "FunctionTimeoutError",
        "InputCancellation",
        "InternalFailure",
        "OutputExpiredError",
        "RemoteError",
    }
)
_CANCEL_CONFIRM_UNCONFIRMED_MODAL_ERRORS = frozenset(
    {
        "AuthError",
        "ClientClosed",
        "ConflictError",
        "ConnectionError",
        "DataLossError",
        "InternalError",
        "InvalidError",
        "NotFoundError",
        "PermissionDeniedError",
        "RequestSizeError",
        "ResourceExhaustedError",
        "ServiceError",
        "UnimplementedError",
        "VersionError",
    }
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + chr(10),
        encoding="utf-8",
    )


def _append_github_env(name: str, value: object) -> None:
    target = os.environ.get("GITHUB_ENV")
    if not target:
        return
    with Path(target).open("a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}" + chr(10))


def _cached_source_evidence() -> dict[str, Any] | None:
    path = Path("open-evidence/source-master.json")
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict):
        return None
    if (
        str(raw.get("video_id") or "") != os.environ["CLIPPER_TARGET_VIDEO_ID"]
        or str(raw.get("channel_id") or "") != os.environ["CLIPPER_TARGET_CHANNEL_ID"]
        or str(raw.get("quality_policy") or "") != "highest_available_no_transcode"
        or len(str(raw.get("sha256") or "")) != 64
        or not str(raw.get("volume_path") or "").startswith("/inputs/")
    ):
        return None
    return {str(key): value for key, value in raw.items()}


def _source_payload(
    modal: Any,
    budget: _BudgetLedger,
    *,
    execution_id: str,
) -> dict[str, Any]:
    evidence = _cached_source_evidence()
    attempts: list[dict[str, object]] = []
    reused_evidence = evidence is not None
    if evidence is None:
        candidate = VideoCandidate(
            os.environ["CLIPPER_TARGET_VIDEO_ID"],
            "Authorized production target",
            os.environ["CLIPPER_TARGET_CHANNEL_ID"],
            "Authorized production channel",
            os.environ["CLIPPER_TARGET_VIDEO_URL"],
        )
        acquire = modal.Function.from_name(
            os.environ["CLIPPER_MODAL_PIPELINE_APP"],
            "acquire_source",
        )
        try:
            evidence = _acquire_remote_source(
                acquire,
                candidate,
                expected_git_sha=os.environ["CLIPPER_ACCEPTANCE_SHA"],
                budget=budget,
                attempt_evidence=attempts,
                execution_id=execution_id,
            )
            _write_json(Path("open-evidence/source-master.json"), evidence)
        finally:
            _write_json(
                Path("open-evidence/source-egress-attempts.json"),
                {"attempts": attempts, "reused_evidence": False},
            )
            _write_json(Path("open-evidence/source-budget.json"), budget.to_dict())
    else:
        _write_json(
            Path("open-evidence/source-egress-attempts.json"),
            {"attempts": attempts, "reused_evidence": reused_evidence},
        )
        _write_json(Path("open-evidence/source-budget.json"), budget.to_dict())

    return {
        "video_id": os.environ["CLIPPER_TARGET_VIDEO_ID"],
        "channel_id": os.environ["CLIPPER_TARGET_CHANNEL_ID"],
        "canonical_url": os.environ["CLIPPER_TARGET_VIDEO_URL"],
        "evidence": evidence,
    }


def _scoped_brief_yaml() -> str:
    brief_path = Path(os.environ["CLIPPER_CAMPAIGN_BRIEF"])
    text = brief_path.read_text(encoding="utf-8")
    data = json.loads(text) if brief_path.suffix.lower() == ".json" else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise RuntimeError("campaign brief root must be an object")
    targets = data.get("targets")
    if not isinstance(targets, dict) or str(targets.get("mode") or "").lower() != "explicit":
        raise RuntimeError("production execution requires explicit targets")
    videos = targets.get("videos")
    if not isinstance(videos, list) or not all(isinstance(item, dict) for item in videos):
        raise RuntimeError("production execution requires explicit target videos")

    video_id = os.environ["CLIPPER_TARGET_VIDEO_ID"]
    channel_id = os.environ["CLIPPER_TARGET_CHANNEL_ID"]
    video_url = os.environ["CLIPPER_TARGET_VIDEO_URL"]
    matches = [
        item
        for item in videos
        if str(item.get("video_id") or "").strip() == video_id
        and str(item.get("channel_id") or "").strip() == channel_id
        and str(item.get("url") or "").strip() == video_url
    ]
    if len(matches) != 1:
        raise RuntimeError("selected production target does not exactly match the campaign brief")

    scoped = dict(data)
    scoped_targets = dict(targets)
    scoped_targets["videos"] = [dict(matches[0])]
    scoped["targets"] = scoped_targets
    rendered = yaml.safe_dump(scoped, sort_keys=False, allow_unicode=True)
    if not rendered.strip():
        raise RuntimeError("selected production target produced an empty scoped brief")
    return rendered


def _validate_result(result: object, *, render: bool) -> dict[str, Any]:
    if not isinstance(result, dict):
        raise RuntimeError(f"production cycle returned {type(result).__name__}, expected object")
    normalized = {str(key): value for key, value in result.items()}
    _write_json(Path("open-evidence/cycle-result.json"), normalized)

    if normalized.get("status") != "PASS":
        raise RuntimeError(f"production cycle failed: {normalized}")
    if normalized.get("execution_mode") != os.environ["CLIPPER_EXECUTION_MODE"]:
        raise RuntimeError(
            f"production execution mode drifted from request: {normalized.get('execution_mode')}"
        )
    if normalized.get("execution_id") != os.environ["CLIPPER_EXECUTION_ID"]:
        raise RuntimeError(
            f"production execution ID drifted from request: {normalized.get('execution_id')}"
        )
    if normalized.get("deployed_git_sha") != os.environ["CLIPPER_ACCEPTANCE_SHA"]:
        raise RuntimeError(
            "production worker did not prove the requested deployed SHA: "
            f"{normalized.get('deployed_git_sha')}"
        )
    if normalized.get("pipeline_status") not in {"SUCCESS", "DEGRADED"}:
        raise RuntimeError(f"production pipeline did not complete: {normalized}")

    if render:
        rendered = int(normalized.get("rendered") or 0)
        reviewable = int(normalized.get("reviewable") or 0)
        if normalized.get("review_status") != "PENDING_ACTUAL_MP4_REVIEW":
            raise RuntimeError(f"production result bypassed actual-MP4 review: {normalized}")
        if reviewable != rendered:
            raise RuntimeError("every accepted rendered quality moment must remain reviewable")
    elif normalized.get("review_status") != "NOT_RENDERED":
        raise RuntimeError(f"editorial-only acceptance unexpectedly rendered media: {normalized}")
    return normalized


def _validated_resume_provenance(source_payload: dict[str, Any]) -> dict[str, Any] | None:
    requested = os.environ.get("CLIPPER_RESUME_FROM_RUN_ID", "").strip()
    if not requested:
        return None
    path_value = os.environ.get("CLIPPER_RESUME_PROVENANCE_PATH", "").strip()
    if not path_value:
        raise RuntimeError("resume execution is missing reviewed artifact provenance")
    path = Path(path_value)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("resume provenance request must be an object")
    provenance = {str(key): value for key, value in raw.items()}
    if str(provenance.get("workflow_run_id") or "") != requested:
        raise RuntimeError("resume provenance workflow run ID does not match request")

    evidence = source_payload.get("evidence")
    if not isinstance(evidence, dict):
        raise RuntimeError("resume source payload is missing source evidence")
    video_id = str(source_payload.get("video_id") or "")
    source_hash = str(evidence.get("sha256") or "")
    source_hashes = provenance.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise RuntimeError("resume provenance source hashes must be an object")
    if str(source_hashes.get(video_id) or "") != source_hash:
        raise RuntimeError("resume provenance source hash does not match acquired source master")
    return provenance


def _finite_positive_env(name: str, *, default: str | None = None) -> float:
    raw = os.environ.get(name, default) if default is not None else os.environ.get(name)
    try:
        value = float(raw) if raw is not None else float("nan")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _cancellation_confirmation_is_terminal(modal_module: Any, exc: BaseException) -> bool:
    error_name = type(exc).__name__
    exception_namespace = getattr(modal_module, "exception", None)
    if exception_namespace is None:
        return error_name not in _CANCEL_CONFIRM_UNCONFIRMED_MODAL_ERRORS

    terminal_types = tuple(
        error_type
        for name in _CANCEL_CONFIRM_TERMINAL_MODAL_ERRORS
        if isinstance((error_type := getattr(exception_namespace, name, None)), type)
    )
    if terminal_types and isinstance(exc, terminal_types):
        return True

    modal_error_type = getattr(exception_namespace, "Error", None)
    if isinstance(modal_error_type, type) and isinstance(exc, modal_error_type):
        return False
    return error_name not in _CANCEL_CONFIRM_UNCONFIRMED_MODAL_ERRORS


def run(*, render: bool) -> dict[str, Any]:
    import modal

    evidence_dir = Path("open-evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    execution_id = uuid.uuid4().hex
    os.environ["CLIPPER_EXECUTION_ID"] = execution_id

    max_gpu_seconds = _finite_positive_env("CLIPPER_MAX_GPU_SECONDS")
    max_estimated_usd = _finite_positive_env("CLIPPER_MAX_ESTIMATED_USD")
    budget = _BudgetLedger(max_gpu_seconds, max_estimated_usd)
    poll_seconds = _finite_positive_env("CLIPPER_MODAL_SPY_POLL_SECONDS", default="5")
    barrier_timeout_seconds = _finite_positive_env(
        "CLIPPER_MODAL_SPY_BARRIER_TIMEOUT_SECONDS",
        default="30",
    )
    cancel_confirmation_seconds = _finite_positive_env(
        "CLIPPER_MODAL_ROOT_CANCEL_CONFIRM_SECONDS",
        default="30",
    )

    scoped_brief_yaml = _scoped_brief_yaml()
    (evidence_dir / "scoped-brief.yaml").write_text(scoped_brief_yaml, encoding="utf-8")

    spy = ModalExecutionSpy(
        (
            os.environ["CLIPPER_MODAL_APP"],
            os.environ["CLIPPER_MODAL_PIPELINE_APP"],
        ),
        evidence_dir / "modal-spy.ndjson",
        execution_id=execution_id,
    )
    spy_thread_failure: list[str] = []

    def run_spy() -> None:
        try:
            spy.run()
        except BaseException as exc:
            rendered = f"{type(exc).__name__}: {exc}"
            spy_thread_failure.append(rendered)

    spy_thread = threading.Thread(target=run_spy, daemon=True)
    spy_thread_started = False
    call: Any | None = None
    call_id = ""
    call_started = 0.0
    remote_completed = False
    root_budget_charged = False
    cancelled = threading.Event()
    cancellation_failure_reason: str | None = None
    root_cancel_terminal_confirmed = False
    root_hard_termination_succeeded = False
    run_succeeded = False
    run_failure_reason: str | None = None

    def cancel_call(reason: str) -> None:
        nonlocal cancellation_failure_reason
        nonlocal root_cancel_terminal_confirmed
        nonlocal root_hard_termination_succeeded
        if call is None or cancelled.is_set():
            return

        errors: list[str] = []

        def request_cancel(*, terminate_containers: bool, phase: str) -> bool:
            for attempt in range(1, 4):
                try:
                    call.cancel(terminate_containers=terminate_containers)
                except BaseException as exc:
                    if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                        raise
                    errors.append(f"{phase} {type(exc).__name__}: {exc}")
                    print(
                        json.dumps(
                            {
                                "event": "production_call_cancel_retry",
                                "function_call_id": call_id,
                                "reason": reason,
                                "phase": phase,
                                "attempt": attempt,
                                "terminate_containers": terminate_containers,
                                "error": f"{type(exc).__name__}: {exc}",
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    if attempt < 3:
                        time.sleep(0.25 * attempt)
                    continue
                print(
                    json.dumps(
                        {
                            "event": "production_call_cancel_requested",
                            "function_call_id": call_id,
                            "reason": reason,
                            "phase": phase,
                            "attempt": attempt,
                            "terminate_containers": terminate_containers,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                return True
            return False

        def confirm_terminal(*, phase: str) -> str | None:
            nonlocal root_cancel_terminal_confirmed
            try:
                call.get(timeout=cancel_confirmation_seconds)
            except TimeoutError:
                errors.append(
                    f"{phase} terminal confirmation timed out after "
                    f"{cancel_confirmation_seconds:.3f}s"
                )
                return None
            except BaseException as exc:
                if isinstance(exc, (KeyboardInterrupt, SystemExit, GeneratorExit)):
                    raise
                if not _cancellation_confirmation_is_terminal(modal, exc):
                    errors.append(f"{phase} confirmation {type(exc).__name__}: {exc}")
                    return None
                root_cancel_terminal_confirmed = True
                return type(exc).__name__
            root_cancel_terminal_confirmed = True
            return "result"

        soft_requested = request_cancel(
            terminate_containers=False,
            phase="soft_cancel",
        )
        confirmed_by = confirm_terminal(phase="soft_cancel") if soft_requested else None
        if confirmed_by is not None:
            cancelled.set()
            cancellation_failure_reason = None
            print(
                json.dumps(
                    {
                        "event": "production_call_cancelled",
                        "function_call_id": call_id,
                        "reason": reason,
                        "terminal_confirmation": confirmed_by,
                        "hard_termination": False,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return

        print(
            json.dumps(
                {
                    "event": "production_call_cancel_escalated",
                    "function_call_id": call_id,
                    "reason": reason,
                    "errors": list(errors),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        hard_requested = request_cancel(
            terminate_containers=True,
            phase="hard_cancel",
        )
        if not hard_requested:
            failure = ProductionCallNotTerminated(call_id, errors)
            cancellation_failure_reason = str(failure)
            print(
                json.dumps(
                    {
                        "event": "production_call_cancel_unconfirmed",
                        "function_call_id": call_id,
                        "reason": reason,
                        "error": str(failure),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            raise failure

        # Modal documents terminate_containers=True as terminating the containers
        # running this FunctionCall's cancelled inputs; concurrently affected inputs
        # are rescheduled. Treat the successful exact-call hard-cancel request as the
        # termination backstop, while still trying to obtain terminal evidence.
        root_hard_termination_succeeded = True
        confirmed_by = confirm_terminal(phase="hard_cancel")
        cancelled.set()
        cancellation_failure_reason = None
        print(
            json.dumps(
                {
                    "event": "production_call_cancelled",
                    "function_call_id": call_id,
                    "reason": reason,
                    "terminal_confirmation": confirmed_by or "hard_cancel_api_ack",
                    "hard_termination": True,
                    "terminal_confirmed": root_cancel_terminal_confirmed,
                    "confirmation_errors": list(errors),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: object) -> None:
        cancel_call(f"runner_signal_{signum}")
        raise ProductionCallCancelled(f"runner received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    try:
        spy_thread.start()
        spy_thread_started = True

        source_payload = _source_payload(modal, budget, execution_id=execution_id)
        resume_provenance = _validated_resume_provenance(source_payload)
        if resume_provenance is not None:
            _write_json(evidence_dir / "resume-provenance-validated.json", resume_provenance)
        # Compute thresholds are retained for accounting/evidence only. They must never
        # prevent root execution or terminate HILP/production once rights and static gates pass.
        request = {
            "sources": [source_payload],
            "brief_yaml": scoped_brief_yaml,
            "render": render,
            "editorial_acceptance_probe": not render,
            "fresh_inference": os.environ["CLIPPER_FRESH_INFERENCE"] == "true",
            "resume_from_run_id": os.environ.get("CLIPPER_RESUME_FROM_RUN_ID") or None,
            "resume_provenance": resume_provenance,
            "git_sha": os.environ["CLIPPER_ACCEPTANCE_SHA"],
            "execution_id": execution_id,
            "max_gpu_seconds": max_gpu_seconds,
            "max_estimated_usd": max_estimated_usd,
        }

        function = modal.Function.from_name(
            os.environ["CLIPPER_MODAL_PIPELINE_APP"],
            "run_full_cycle",
        )
        call, call_started, submission_error = _spawn_recoverable_modal_call(
            function,
            request,
            budget=budget,
            gpu_count=2.0,
            estimated_usd_per_second=0.000444,
        )
        # FunctionCall.from_id(...) is intentionally recoverable but unhydrated.
        # Hydrate it before reading object_id; Modal 1.5.x raises AttributeError
        # when object_id is accessed on the unhydrated recovery handle.
        call.hydrate()
        call_id = str(call.object_id)
        spy.root_function_call_id = call_id
        if submission_error is not None:
            raise submission_error

        metadata = {
            "event": "production_call_spawned",
            "function_call_id": call_id,
            "execution_id": execution_id,
            "pipeline_app": os.environ["CLIPPER_MODAL_PIPELINE_APP"],
            "model_app": os.environ["CLIPPER_MODAL_APP"],
            "acceptance_sha": os.environ["CLIPPER_ACCEPTANCE_SHA"],
            "resume_from_run_id": request["resume_from_run_id"],
            "resume_artifact_run_path": (
                str((resume_provenance or {}).get("artifact_run_path") or "") or None
            ),
            "fresh_inference": request["fresh_inference"],
            "render": render,
            "editorial_acceptance_probe": request["editorial_acceptance_probe"],
            "max_gpu_seconds": max_gpu_seconds,
            "max_estimated_usd": max_estimated_usd,
            "budget_enforcement": False,
            "spawned_at": datetime.now(UTC).isoformat(),
        }
        _write_json(evidence_dir / "modal-function-call.json", metadata)
        print(json.dumps(metadata, sort_keys=True), flush=True)

        while True:
            if spy.abort_reason is not None:
                cancel_call(spy.abort_reason)
                raise RuntimeError(f"Modal spy aborted production early: {spy.abort_reason}")
            if not spy_thread.is_alive():
                detail = spy_thread_failure[-1] if spy_thread_failure else "no exception detail"
                reason_prefix = "Modal spy thread exited unexpectedly before production completion"
                reason = f"{reason_prefix}: {detail}"
                cancel_call(reason)
                raise RuntimeError(reason)

            try:
                # Poll cadence is independent of accounting thresholds. A threshold crossing is
                # observable in the final ledger but is never a cancellation or acceptance signal.
                result = call.get(timeout=poll_seconds)
                remote_completed = True
                elapsed = max(0.0, time.monotonic() - call_started)
                budget.charge(
                    elapsed,
                    gpu_count=2.0,
                    estimated_usd_per_second=0.000444,
                )
                root_budget_charged = True
                break
            except TimeoutError:
                continue

        if not spy.wait_for_producer_barrier(timeout_seconds=barrier_timeout_seconds):
            if spy.abort_reason is not None:
                raise RuntimeError(f"Modal spy aborted before producer barrier: {spy.abort_reason}")
            raise RuntimeError(
                "Modal spy did not observe a closed pipeline producer set "
                "before the production terminal barrier"
            )
        if spy.abort_reason is not None:
            raise RuntimeError(f"Modal spy aborted before producer barrier: {spy.abort_reason}")
        terminal_event = spy.summary().get("terminal_event")
        if not isinstance(terminal_event, dict):
            raise RuntimeError("Modal spy terminal evidence is missing after drain")
        if not isinstance(result, dict):
            raise RuntimeError(
                "Modal spy observed terminal evidence but production result is not an object: "
                f"terminal={terminal_event} result={result!r}"
            )
        for field in ("status", "execution_id", "pipeline_status", "review_status"):
            if terminal_event.get(field) != result.get(field):
                raise RuntimeError(
                    "Modal spy terminal evidence does not match production result: "
                    f"field={field} terminal={terminal_event.get(field)!r} "
                    f"result={result.get(field)!r}"
                )

        validated = _validate_result(result, render=render)
        _append_github_env("CLIPPER_RUN_VOLUME", validated["run_volume"])
        _append_github_env("CLIPPER_RUN_PATH", validated["run_path"])
        run_succeeded = True
        return validated
    except BaseException as exc:
        run_failure_reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        cleanup_failure_reason: str | None = None
        try:
            if call is not None and not remote_completed and not cancelled.is_set():
                for cleanup_attempt in range(1, 4):
                    try:
                        cancel_call("watchdog exited before production call completed")
                    except BaseException as cleanup_exc:
                        cleanup_failure_reason = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                        if cleanup_attempt < 3:
                            time.sleep(0.25 * cleanup_attempt)
                        continue
                    cleanup_failure_reason = None
                    break
        finally:
            if call_started > 0 and not root_budget_charged:
                budget.charge(
                    max(0.0, time.monotonic() - call_started),
                    gpu_count=2.0,
                    estimated_usd_per_second=0.000444,
                )
                root_budget_charged = True
        if cleanup_failure_reason:
            if run_failure_reason:
                run_failure_reason = f"{run_failure_reason}; cleanup={cleanup_failure_reason}"
            else:
                run_failure_reason = cleanup_failure_reason
        spy.request_stop()
        if spy_thread_started:
            spy_thread.join(timeout=max(1.0, poll_seconds))
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        summary = spy.summary()
        summary.update(
            {
                "function_call_id": call_id,
                "call_cancelled": cancelled.is_set(),
                "root_call_terminal_confirmed": (
                    remote_completed or root_cancel_terminal_confirmed
                ),
                "root_call_hard_termination_succeeded": root_hard_termination_succeeded,
                "root_call_stopped": (
                    remote_completed
                    or root_cancel_terminal_confirmed
                    or root_hard_termination_succeeded
                ),
                "root_call_cancellation_failure": (
                    cancellation_failure_reason or cleanup_failure_reason
                ),
                "render": render,
                "budget_enforcement": False,
                "budget": budget.to_dict(),
            }
        )
        if not run_succeeded:
            summary["status"] = "ABORT"
            if not summary.get("abort_reason"):
                summary["abort_reason"] = (
                    run_failure_reason or "watchdog exited before validated production success"
                )
        _write_json(evidence_dir / "modal-spy-summary.json", summary)
        print(
            f"[modal-spy:summary] {json.dumps(summary, sort_keys=True)}",
            flush=True,
        )


def main() -> None:
    render = os.environ.get("CLIPPER_RENDER", "true").lower() == "true"
    run(render=render)


if __name__ == "__main__":
    main()
