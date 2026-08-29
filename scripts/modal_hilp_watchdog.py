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

from modal_execution_spy import ModalExecutionSpy


class ProductionCallCancelled(RuntimeError):
    pass


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


def _source_payload() -> dict[str, Any]:
    evidence = json.loads(Path("open-evidence/source-master.json").read_text(encoding="utf-8"))
    return {
        "video_id": os.environ["CLIPPER_TARGET_VIDEO_ID"],
        "channel_id": os.environ["CLIPPER_TARGET_CHANNEL_ID"],
        "canonical_url": os.environ["CLIPPER_TARGET_VIDEO_URL"],
        "evidence": evidence,
    }


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


def run(*, render: bool) -> dict[str, Any]:
    import modal

    evidence_dir = Path("open-evidence")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    execution_id = uuid.uuid4().hex
    os.environ["CLIPPER_EXECUTION_ID"] = execution_id
    max_gpu_seconds = float(os.environ["CLIPPER_MAX_GPU_SECONDS"])
    max_estimated_usd = float(os.environ["CLIPPER_MAX_ESTIMATED_USD"])
    if (
        not math.isfinite(max_gpu_seconds)
        or not math.isfinite(max_estimated_usd)
        or max_gpu_seconds <= 0
        or max_estimated_usd <= 0
    ):
        raise ValueError("production compute budget limits must be finite and positive")

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
    spy_thread.start()

    request = {
        "sources": [_source_payload()],
        "brief_yaml": Path(os.environ["CLIPPER_CAMPAIGN_BRIEF"]).read_text(encoding="utf-8"),
        "render": render,
        "editorial_acceptance_probe": not render,
        "fresh_inference": os.environ["CLIPPER_FRESH_INFERENCE"] == "true",
        "resume_from_run_id": os.environ.get("CLIPPER_RESUME_FROM_RUN_ID") or None,
        "git_sha": os.environ["CLIPPER_ACCEPTANCE_SHA"],
        "execution_id": execution_id,
        "max_gpu_seconds": max_gpu_seconds,
        "max_estimated_usd": max_estimated_usd,
    }
    function = modal.Function.from_name(
        os.environ["CLIPPER_MODAL_PIPELINE_APP"],
        "run_full_cycle",
    )
    call = function.spawn(request)
    call.hydrate()
    call_id = str(call.object_id)
    spy.root_function_call_id = call_id
    call_started = time.monotonic()

    metadata = {
        "event": "production_call_spawned",
        "function_call_id": call_id,
        "execution_id": execution_id,
        "pipeline_app": os.environ["CLIPPER_MODAL_PIPELINE_APP"],
        "model_app": os.environ["CLIPPER_MODAL_APP"],
        "acceptance_sha": os.environ["CLIPPER_ACCEPTANCE_SHA"],
        "resume_from_run_id": request["resume_from_run_id"],
        "fresh_inference": request["fresh_inference"],
        "render": render,
        "editorial_acceptance_probe": request["editorial_acceptance_probe"],
        "max_gpu_seconds": max_gpu_seconds,
        "max_estimated_usd": max_estimated_usd,
        "spawned_at": datetime.now(UTC).isoformat(),
    }
    _write_json(evidence_dir / "modal-function-call.json", metadata)
    print(json.dumps(metadata, sort_keys=True), flush=True)

    cancelled = threading.Event()

    def cancel_call(reason: str) -> None:
        if cancelled.is_set():
            return
        cancelled.set()
        print(
            json.dumps(
                {
                    "event": "production_call_cancel",
                    "function_call_id": call_id,
                    "reason": reason,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        call.cancel(terminate_containers=False)

    previous_handlers: dict[int, Any] = {}

    def handle_signal(signum: int, _frame: object) -> None:
        cancel_call(f"runner_signal_{signum}")
        raise ProductionCallCancelled(f"runner received signal {signum}")

    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.getsignal(signum)
        signal.signal(signum, handle_signal)

    poll_seconds = float(os.environ.get("CLIPPER_MODAL_SPY_POLL_SECONDS", "5"))
    if not math.isfinite(poll_seconds) or poll_seconds <= 0:
        raise ValueError("CLIPPER_MODAL_SPY_POLL_SECONDS must be finite and positive")
    barrier_timeout_seconds = float(
        os.environ.get("CLIPPER_MODAL_SPY_BARRIER_TIMEOUT_SECONDS", "30")
    )
    if not math.isfinite(barrier_timeout_seconds) or barrier_timeout_seconds <= 0:
        raise ValueError("Modal spy producer barrier timeout must be finite and positive")

    try:
        while True:
            if spy.abort_reason is not None:
                cancel_call(spy.abort_reason)
                raise RuntimeError(f"Modal spy aborted production early: {spy.abort_reason}")
            if not spy_thread.is_alive():
                detail = spy_thread_failure[-1] if spy_thread_failure else "no exception detail"
                reason = f"Modal spy thread exited unexpectedly before production completion: {detail}"
                cancel_call(reason)
                raise RuntimeError(reason)

            elapsed = max(0.0, time.monotonic() - call_started)
            conservative_gpu_seconds = elapsed * 2.0
            conservative_cost_usd = elapsed * 0.000444
            if conservative_gpu_seconds >= max_gpu_seconds:
                reason = (
                    "conservative in-flight GPU budget reached before completion: "
                    f"{conservative_gpu_seconds:.1f} >= {max_gpu_seconds:.1f}"
                )
                cancel_call(reason)
                raise RuntimeError(reason)
            if conservative_cost_usd >= max_estimated_usd:
                reason = (
                    "conservative in-flight cost budget reached before completion: "
                    f"{conservative_cost_usd:.4f} >= {max_estimated_usd:.4f}"
                )
                cancel_call(reason)
                raise RuntimeError(reason)
            try:
                result = call.get(timeout=poll_seconds)
                elapsed = max(0.0, time.monotonic() - call_started)
                conservative_gpu_seconds = elapsed * 2.0
                conservative_cost_usd = elapsed * 0.000444
                if conservative_gpu_seconds >= max_gpu_seconds:
                    raise RuntimeError(
                        "conservative GPU budget exceeded by completed production call: "
                        f"{conservative_gpu_seconds:.1f} >= {max_gpu_seconds:.1f}"
                    )
                if conservative_cost_usd >= max_estimated_usd:
                    raise RuntimeError(
                        "conservative cost budget exceeded by completed production call: "
                        f"{conservative_cost_usd:.4f} >= {max_estimated_usd:.4f}"
                    )
                break
            except TimeoutError:
                continue

        if not spy.wait_for_producer_barrier(timeout_seconds=barrier_timeout_seconds):
            if spy.abort_reason is not None:
                raise RuntimeError(f"Modal spy aborted before producer barrier: {spy.abort_reason}")
            raise RuntimeError(
                "Modal spy did not observe a closed pipeline editorial-call set "
                "before the production terminal barrier"
            )
        if spy.abort_reason is not None:
            raise RuntimeError(f"Modal spy aborted before producer barrier: {spy.abort_reason}")
        terminal_event = spy.summary().get("terminal_event")
        if not isinstance(terminal_event, dict):
            raise RuntimeError("Modal spy terminal evidence is missing after drain")
        if not isinstance(result, dict) or terminal_event.get("status") != result.get("status"):
            raise RuntimeError(
                "Modal spy terminal status does not match production result: "
                f"terminal={terminal_event} result={result}"
            )

        validated = _validate_result(result, render=render)
        _append_github_env("CLIPPER_RUN_VOLUME", validated["run_volume"])
        _append_github_env("CLIPPER_RUN_PATH", validated["run_path"])
        return validated
    finally:
        spy.request_stop()
        spy_thread.join(timeout=max(1.0, poll_seconds))
        for signum, previous in previous_handlers.items():
            signal.signal(signum, previous)
        summary = spy.summary()
        summary.update(
            {
                "function_call_id": call_id,
                "call_cancelled": cancelled.is_set(),
                "render": render,
            }
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
