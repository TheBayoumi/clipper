from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import subprocess
import threading
import time
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MARKER = "<!-- clipper-modal-spy -->"
RECOGNIZED_EVENTS = {
    "editorial_model_ready",
    "editorial_evidence_projection",
    "editorial_repartition",
    "editorial_context_repartition",
    "editorial_capacity_probe",
    "editorial_acceptance_probe_result",
    "editorial_request_plan",
    "editorial_generation_start",
    "editorial_generation_complete",
    "editorial_execution_timeout",
    "editorial_oom",
    "editorial_capacity_fallback",
    "application_result",
    "vision_generation_start",
    "vision_generation_complete",
    "vision_json_validation",
    "vision_generation_capacity_expand",
}


class ModalExecutionSpy:
    """Stream Modal telemetry, publish compact evidence, and flag unsafe execution."""

    _CALL_ID_RE = re.compile(r"\b(fc-[A-Z0-9]+)\b")

    def __init__(
        self,
        apps: tuple[str, ...],
        output: Path,
        *,
        root_function_call_id: str | None = None,
        execution_id: str | None = None,
        generation_stall_seconds: float | None = None,
    ) -> None:
        self.apps = apps
        self.pipeline_app = apps[-1] if apps else ""
        self.root_function_call_id = root_function_call_id
        self.execution_id = execution_id
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.processes: list[subprocess.Popen[str]] = []
        self.events_seen = 0
        self.event_counts: dict[str, int] = {}
        self.latest: dict[str, dict[str, Any]] = {}
        self.pr_number: int | None = None
        self.comment_id: int | None = None
        self.abort_reason: str | None = None
        self.abort_event: dict[str, Any] | None = None
        self.stream_errors: list[str] = []
        self.started_at = datetime.now(UTC).isoformat()
        self._last_request_plan_signature: tuple[object, ...] | None = None
        self._active_generations: dict[str, float] = {}
        self.generation_stall_seconds = (
            float(generation_stall_seconds)
            if generation_stall_seconds is not None
            else float(os.getenv("CLIPPER_MODAL_GENERATION_STALL_SECONDS", "720"))
        )
        if self.generation_stall_seconds <= 0:
            raise ValueError("generation stall timeout must be positive")

    @staticmethod
    def _positive_int(value: object) -> int | None:
        if isinstance(value, bool) or not isinstance(value, int):
            return None
        return value if value > 0 else None

    @staticmethod
    def _github_request(
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        token = os.environ.get("GITHUB_TOKEN", "").strip()
        repository = os.environ.get("GITHUB_REPOSITORY", "").strip()
        if not token or not repository:
            return None
        url = f"https://api.github.com/repos/{repository}{path}"
        if not url.startswith("https://api.github.com/"):
            raise ValueError(f"refusing non-GitHub API URL: {url}")
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            method=method,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "Content-Type": "application/json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            raw = response.read()
        return json.loads(raw) if raw else None

    def _resolve_pr(self) -> None:
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        branch = os.environ.get("GITHUB_HEAD_REF", "") or os.environ.get("GITHUB_REF_NAME", "")
        if not repository or not branch:
            return
        owner = repository.split("/", 1)[0]
        query = urllib.parse.urlencode(
            {"state": "open", "head": f"{owner}:{branch}", "per_page": 20}
        )
        pulls = self._github_request("GET", f"/pulls?{query}")
        if isinstance(pulls, list) and pulls:
            number = pulls[0].get("number")
            if isinstance(number, int) and not isinstance(number, bool):
                self.pr_number = number

    def _resolve_comment(self) -> None:
        if self.pr_number is None:
            return
        comments = self._github_request(
            "GET",
            f"/issues/{self.pr_number}/comments?per_page=100",
        )
        if not isinstance(comments, list):
            return
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            body = comment.get("body")
            comment_id = comment.get("id")
            if (
                isinstance(body, str)
                and MARKER in body
                and isinstance(comment_id, int)
                and not isinstance(comment_id, bool)
            ):
                self.comment_id = comment_id
                return

    @classmethod
    def _function_call_id(cls, line: str) -> str | None:
        match = cls._CALL_ID_RE.search(line)
        return match.group(1) if match else None

    def _belongs_to_execution(
        self,
        app: str,
        line: str,
        payload: dict[str, Any],
    ) -> bool:
        if self.root_function_call_id is None and self.execution_id is None:
            return True
        if app == self.pipeline_app:
            return self._function_call_id(line) == self.root_function_call_id
        return (
            self.execution_id is not None
            and str(payload.get("execution_id") or "") == self.execution_id
        )

    @staticmethod
    def _parse_json(line: str) -> dict[str, Any] | None:
        marker = line.find("{")
        if marker < 0:
            return None
        try:
            value = json.loads(line[marker:])
        except json.JSONDecodeError:
            return None
        if not isinstance(value, dict):
            return None
        return {str(key): item for key, item in value.items()}

    @staticmethod
    def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "event",
            "execution_id",
            "stage",
            "task",
            "application_status",
            "recovery_action",
            "reason",
            "worker_lifecycle_id",
            "cache_implementation",
            "from_cache_implementation",
            "to_cache_implementation",
            "input_tokens",
            "context_limit_tokens",
            "available_output_tokens",
            "target_input_tokens",
            "observed_input_tokens",
            "generation_budget_tokens",
            "runtime_safe_input_tokens",
            "capacity_repartitionable",
            "timeout_seconds",
            "serialized_request_bytes",
            "output_tokens",
            "duration_seconds",
            "partition_count",
            "ranges",
            "status",
            "video_id",
            "previous_range",
            "next_range",
            "raw_event_count",
            "projected_event_count",
            "raw_serialized_bytes",
            "projected_serialized_bytes",
            "model_gpu_indices",
            "model_bytes_by_device",
            "placement_policy",
        }
        return {key: value for key, value in event.items() if key in allowed}

    def _set_abort(self, reason: str, event: dict[str, Any] | None = None) -> None:
        if self.abort_reason is not None:
            return
        self.abort_reason = reason
        self.abort_event = dict(event or {})
        self.stop.set()
        print(
            "[modal-spy:abort] "
            + json.dumps(
                {"reason": reason, "event": self.abort_event},
                ensure_ascii=False,
                sort_keys=True,
            ),
            flush=True,
        )

    def _validate_event(self, event: dict[str, Any]) -> str | None:
        name = str(event.get("event") or "")

        if name == "editorial_execution_timeout":
            return f"editorial execution hit its runtime timeout: {event}"

        if name == "application_result":
            if str(event.get("application_status") or "") == "FAILED":
                return f"non-recoverable Modal application failure: {event}"
            return None

        if name == "editorial_evidence_projection":
            raw_count = self._positive_int(event.get("raw_event_count"))
            projected_count = event.get("projected_event_count")
            raw_bytes = self._positive_int(event.get("raw_serialized_bytes"))
            projected_bytes = event.get("projected_serialized_bytes")
            if not isinstance(projected_count, int) or isinstance(projected_count, bool):
                return f"projection emitted invalid projected_event_count: {event}"
            if projected_count < 0:
                return f"projection emitted negative event count: {event}"
            if raw_count is not None and projected_count > raw_count:
                return f"projection expanded event cardinality: {event}"
            if not isinstance(projected_bytes, int) or isinstance(projected_bytes, bool):
                return f"projection emitted invalid projected byte count: {event}"
            if projected_bytes < 0:
                return f"projection emitted negative byte count: {event}"
            if (
                raw_count is not None
                and raw_count > 0
                and raw_bytes is not None
                and projected_bytes > raw_bytes
            ):
                return f"projection expanded serialized evidence: {event}"
            return None

        if name == "editorial_repartition":
            partition_count = self._positive_int(event.get("partition_count"))
            ranges = event.get("ranges")
            if partition_count is None or not isinstance(ranges, list):
                return f"repartition telemetry is malformed: {event}"
            if partition_count < 2 or len(ranges) != partition_count:
                return f"repartition cardinality is inconsistent: {event}"
            normalized: list[tuple[int, int]] = []
            for item in ranges:
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or not all(
                        isinstance(value, int) and not isinstance(value, bool) for value in item
                    )
                ):
                    return f"repartition range is malformed: {event}"
                left, right = item
                if right <= left:
                    return f"repartition range is empty or reversed: {event}"
                normalized.append((left, right))
            if any(
                normalized[index][1] != normalized[index + 1][0]
                for index in range(len(normalized) - 1)
            ):
                return f"repartition ranges are not contiguous: {event}"

            observed = self._positive_int(event.get("observed_input_tokens"))
            target = self._positive_int(event.get("target_input_tokens"))
            if observed is not None and target is not None and target < observed:
                required = math.ceil(observed / target)
                if partition_count < required:
                    return (
                        "token-aware repartition under-partitioned measured input: "
                        f"required={required} event={event}"
                    )
            return None

        if name == "editorial_context_repartition":
            previous = event.get("previous_range")
            next_range = event.get("next_range")
            if (
                not isinstance(previous, list)
                or not isinstance(next_range, list)
                or len(previous) != 2
                or len(next_range) != 2
                or not all(
                    isinstance(value, int) and not isinstance(value, bool)
                    for value in (*previous, *next_range)
                )
            ):
                return f"context repartition telemetry is malformed: {event}"
            previous_size = previous[1] - previous[0]
            next_size = next_range[1] - next_range[0]
            if previous_size <= 0 or next_size <= 0 or next_size >= previous_size:
                return f"context repartition did not reduce evidence: {event}"
            return None

        if name == "editorial_capacity_probe":
            status = str(event.get("status") or "")
            input_tokens = self._positive_int(event.get("input_tokens"))
            context_limit = self._positive_int(event.get("context_limit_tokens"))
            if status not in {"FIT", "CAPACITY_REJECTED"}:
                return f"capacity probe emitted invalid status: {event}"
            if input_tokens is None or context_limit is None:
                return f"capacity probe omitted measured token/context evidence: {event}"
            return None

        if name == "editorial_acceptance_probe_result":
            return None

        if name == "editorial_request_plan":
            input_tokens = self._positive_int(event.get("input_tokens"))
            context_limit = self._positive_int(event.get("context_limit_tokens"))
            available_output = self._positive_int(event.get("available_output_tokens"))
            budget = self._positive_int(event.get("generation_budget_tokens"))
            signature = (
                event.get("task"),
                input_tokens,
                budget,
                self._positive_int(event.get("serialized_request_bytes")),
            )
            if signature == self._last_request_plan_signature:
                return f"editorial request plan repeated without forward progress: {event}"
            self._last_request_plan_signature = signature
            if (
                input_tokens is not None
                and context_limit is not None
                and input_tokens >= context_limit
            ):
                return f"generation plan exceeded model context: {event}"
            if available_output is not None and budget is not None and budget > available_output:
                return f"generation plan exceeded available output capacity: {event}"
            return None

        return None

    def _comment_body(self) -> str:
        run_id = os.environ.get("GITHUB_RUN_ID", "unknown")
        sha = os.environ.get("CLIPPER_ACCEPTANCE_SHA") or os.environ.get("GITHUB_SHA", "unknown")
        lines = [
            MARKER,
            "### Modal execution spy",
            f"<code>run={run_id}</code> · <code>sha={sha}</code> · "
            f"structured events <code>{self.events_seen}</code>",
            "",
            "| Signal | Latest evidence |",
            "|---|---|",
        ]

        def row(label: str, value: object) -> None:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
            lines.append(f"| {label} | <code>{rendered}</code> |")

        model = self.latest.get("editorial_model_ready")
        if model:
            row(
                "Editorial placement",
                {
                    "policy": model.get("placement_policy"),
                    "gpu_indices": model.get("model_gpu_indices"),
                    "model_bytes": model.get("model_bytes_by_device"),
                },
            )
        projection = self.latest.get("editorial_evidence_projection")
        if projection:
            row(
                "Evidence projection",
                {
                    "stage": projection.get("stage"),
                    "events": [
                        projection.get("raw_event_count"),
                        projection.get("projected_event_count"),
                    ],
                    "bytes": [
                        projection.get("raw_serialized_bytes"),
                        projection.get("projected_serialized_bytes"),
                    ],
                },
            )
        repartition = self.latest.get("editorial_repartition")
        if repartition:
            row(
                "Token-aware repartition",
                {
                    "stage": repartition.get("stage"),
                    "reason": repartition.get("reason"),
                    "observed_tokens": repartition.get("observed_input_tokens"),
                    "target_tokens": repartition.get("target_input_tokens"),
                    "partitions": repartition.get("partition_count"),
                },
            )
        request = self.latest.get("editorial_request_plan")
        if request:
            row(
                "Current request",
                {
                    "task": request.get("task"),
                    "input_tokens": request.get("input_tokens"),
                    "context_limit": request.get("context_limit_tokens"),
                    "generation_budget": request.get("generation_budget_tokens"),
                },
            )
        generation = self.latest.get("editorial_generation_complete")
        if generation:
            row(
                "Last generation",
                {
                    "task": generation.get("task"),
                    "cache": generation.get("cache_implementation"),
                    "input_tokens": generation.get("input_tokens"),
                    "output_tokens": generation.get("output_tokens"),
                    "duration_seconds": generation.get("duration_seconds"),
                },
            )
        oom = self.latest.get("editorial_oom")
        if oom:
            row(
                "Last OOM",
                {
                    "task": oom.get("task"),
                    "cache": oom.get("cache_implementation"),
                    "input_tokens": oom.get("input_tokens"),
                },
            )
        application = self.latest.get("application_result")
        if application:
            row(
                "Application result",
                {
                    "status": application.get("application_status"),
                    "recovery": application.get("recovery_action"),
                },
            )
        row(
            "Watchdog",
            {
                "status": "ABORT" if self.abort_reason else "PASS",
                "reason": self.abort_reason,
            },
        )
        row("Event counts", self.event_counts)
        return "\n".join(lines)

    def _publish_comment(self) -> None:
        if self.pr_number is None:
            return
        body = self._comment_body()
        try:
            if self.comment_id is None:
                response = self._github_request(
                    "POST",
                    f"/issues/{self.pr_number}/comments",
                    {"body": body},
                )
                if isinstance(response, dict):
                    comment_id = response.get("id")
                    if isinstance(comment_id, int) and not isinstance(comment_id, bool):
                        self.comment_id = comment_id
            else:
                self._github_request(
                    "PATCH",
                    f"/issues/comments/{self.comment_id}",
                    {"body": body},
                )
        except Exception as exc:
            print(
                f"[modal-spy] GitHub comment update failed: {type(exc).__name__}: {exc}",
                flush=True,
            )

    def _record(self, app: str, line: str) -> None:
        payload = self._parse_json(line)
        if payload is None or not self._belongs_to_execution(app, line, payload):
            return
        name = payload.get("event")
        if not isinstance(name, str) or name not in RECOGNIZED_EVENTS:
            return
        compact = self._compact_event(payload)
        with self.lock:
            task = str(compact.get("task") or "")
            if name == "editorial_generation_start" and task:
                self._active_generations[task] = time.monotonic()
            elif name in {
                "editorial_generation_complete",
                "editorial_oom",
                "editorial_execution_timeout",
            } and task:
                self._active_generations.pop(task, None)
            if name in {
                "editorial_generation_complete",
                "editorial_repartition",
                "editorial_context_repartition",
            }:
                self._last_request_plan_signature = None
            violation = self._validate_event(compact)
            self.events_seen += 1
            self.event_counts[name] = self.event_counts.get(name, 0) + 1
            self.latest[name] = compact
            record = {"app": app, **compact}
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"[modal-spy:{app}] {json.dumps(compact, ensure_ascii=False, sort_keys=True)}",
                flush=True,
            )
            self._publish_comment()
            if violation is not None:
                self._set_abort(violation, compact)

    def _follow(self, app: str) -> None:
        command = [
            "modal",
            "app",
            "logs",
            app,
            "--follow",
            "--timestamps",
            "--show-function-id",
            "--show-function-call-id",
            "--show-container-id",
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        with self.lock:
            self.processes.append(process)
        if process.stdout is None:
            self._set_abort(f"Modal log follower for {app} has no stdout pipe")
            return
        try:
            for line in process.stdout:
                if self.stop.is_set():
                    break
                self._record(app, line.rstrip("\n"))
            process.wait()
            if not self.stop.is_set() and process.returncode not in {0, None}:
                self._set_abort(
                    f"Modal log follower exited unexpectedly: app={app} "
                    f"returncode={process.returncode}"
                )
        except BaseException as exc:
            if not self.stop.is_set():
                rendered = f"{app}: {type(exc).__name__}: {exc}"
                with self.lock:
                    self.stream_errors.append(rendered)
                self._set_abort(f"Modal log follower failed: {rendered}")
        finally:
            if process.poll() is None:
                process.terminate()

    def _check_stalled_generations(self) -> None:
        if self.abort_reason is not None or not self._active_generations:
            return
        now = time.monotonic()
        stalled = [
            (task, now - started)
            for task, started in self._active_generations.items()
            if now - started >= self.generation_stall_seconds
        ]
        if stalled:
            task, elapsed = max(stalled, key=lambda item: item[1])
            self._set_abort(
                "editorial generation made no completion progress before watchdog deadline",
                {
                    "event": "editorial_generation_stall",
                    "task": task,
                    "elapsed_seconds": round(elapsed, 3),
                    "stall_limit_seconds": self.generation_stall_seconds,
                    "execution_id": self.execution_id,
                },
            )

    def summary(self) -> dict[str, object]:
        return {
            "status": "ABORT" if self.abort_reason else "PASS",
            "abort_reason": self.abort_reason,
            "abort_event": self.abort_event,
            "started_at": self.started_at,
            "root_function_call_id": self.root_function_call_id,
            "execution_id": self.execution_id,
            "generation_stall_seconds": self.generation_stall_seconds,
            "active_generations": sorted(self._active_generations),
            "events_seen": self.events_seen,
            "event_counts": self.event_counts,
            "latest": self.latest,
            "stream_errors": self.stream_errors,
            "pr_number": self.pr_number,
            "comment_id": self.comment_id,
        }

    def run(self) -> int:
        self._resolve_pr()
        self._resolve_comment()
        self._publish_comment()
        threads = [
            threading.Thread(target=self._follow, args=(app,), daemon=True) for app in self.apps
        ]
        for thread in threads:
            thread.start()
        while not self.stop.wait(1):
            self._check_stalled_generations()
            if self.abort_reason is not None:
                break
            if all(not thread.is_alive() for thread in threads):
                if self.abort_reason is None:
                    self._set_abort("all Modal log streams ended unexpectedly")
                break
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for thread in threads:
            thread.join(timeout=10)
        self._publish_comment()
        self.output.with_suffix(".summary.json").write_text(
            json.dumps(self.summary(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 2 if self.abort_reason else 0

    def request_stop(self, _signum: int | None = None, _frame: object = None) -> None:
        self.stop.set()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--app", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    spy = ModalExecutionSpy(tuple(dict.fromkeys(args.app)), args.output)
    signal.signal(signal.SIGTERM, spy.request_stop)
    signal.signal(signal.SIGINT, spy.request_stop)
    return spy.run()


if __name__ == "__main__":
    raise SystemExit(main())
