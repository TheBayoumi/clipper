from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import threading
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

MARKER = "<!-- clipper-modal-spy -->"
RECOGNIZED_EVENTS = {
    "editorial_model_ready",
    "editorial_evidence_projection",
    "editorial_repartition",
    "editorial_request_plan",
    "editorial_generation_start",
    "editorial_generation_complete",
    "editorial_oom",
    "editorial_capacity_fallback",
    "application_result",
    "vision_generation_start",
    "vision_generation_complete",
    "vision_json_validation",
    "vision_generation_capacity_expand",
}


class ModalExecutionSpy:
    def __init__(self, apps: tuple[str, ...], output: Path) -> None:
        self.apps = apps
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
        if not url.startswith("https://api.github.com/"):
            raise ValueError(f"refusing non-GitHub API URL: {url}")
        with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
            raw = response.read()
        return json.loads(raw) if raw else None

    def _resolve_pr(self) -> None:
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        branch = os.environ.get("GITHUB_REF_NAME", "")
        if not repository or not branch:
            return
        owner = repository.split("/", 1)[0]
        query = urllib.parse.urlencode(
            {
                "state": "open",
                "head": f"{owner}:{branch}",
                "per_page": 20,
            }
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

    @staticmethod
    def _compact_event(event: dict[str, Any]) -> dict[str, Any]:
        allowed = {
            "event",
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
            "target_input_tokens",
            "observed_input_tokens",
            "generation_budget_tokens",
            "output_tokens",
            "duration_seconds",
            "partition_count",
            "ranges",
            "raw_event_count",
            "projected_event_count",
            "raw_serialized_bytes",
            "projected_serialized_bytes",
            "model_gpu_indices",
            "model_bytes_by_device",
            "placement_policy",
        }
        return {key: value for key, value in event.items() if key in allowed}

    def _comment_body(self) -> str:
        projection = self.latest.get("editorial_evidence_projection", {})
        repartition = self.latest.get("editorial_repartition", {})
        request = self.latest.get("editorial_request_plan", {})
        generation = self.latest.get("editorial_generation_complete", {})
        oom = self.latest.get("editorial_oom", {})
        application = self.latest.get("application_result", {})
        model = self.latest.get("editorial_model_ready", {})
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

        if model:
            row(
                "Editorial placement",
                {
                    "policy": model.get("placement_policy"),
                    "gpu_indices": model.get("model_gpu_indices"),
                    "model_bytes": model.get("model_bytes_by_device"),
                },
            )
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
        if oom:
            row(
                "Last OOM",
                {
                    "task": oom.get("task"),
                    "cache": oom.get("cache_implementation"),
                    "input_tokens": oom.get("input_tokens"),
                },
            )
        if application:
            row(
                "Application result",
                {
                    "status": application.get("application_status"),
                    "recovery": application.get("recovery_action"),
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

    @staticmethod
    def _parse_json(line: str) -> dict[str, Any] | None:
        marker = line.find("{")
        if marker < 0:
            return None
        try:
            value = json.loads(line[marker:])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None

    def _record(self, app: str, line: str) -> None:
        payload = self._parse_json(line)
        if payload is None:
            return
        event = payload.get("event")
        if not isinstance(event, str) or event not in RECOGNIZED_EVENTS:
            return
        compact = self._compact_event(payload)
        record = {"app": app, **compact}
        with self.lock:
            self.events_seen += 1
            self.event_counts[event] = self.event_counts.get(event, 0) + 1
            self.latest[event] = compact
            with self.output.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            print(
                f"[modal-spy:{app}] {json.dumps(compact, sort_keys=True)}",
                flush=True,
            )
            self._publish_comment()

    def _follow(self, app: str) -> None:
        command = [
            "modal",
            "app",
            "logs",
            app,
            "--follow",
            "--timestamps",
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
            raise RuntimeError(f"Modal log follower for {app} has no stdout pipe")
        try:
            for line in process.stdout:
                if self.stop.is_set():
                    break
                self._record(app, line.rstrip("\n"))
        finally:
            if process.poll() is None:
                process.terminate()

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
            if all(not thread.is_alive() for thread in threads):
                break
        for process in self.processes:
            if process.poll() is None:
                process.terminate()
        for thread in threads:
            thread.join(timeout=10)
        self._publish_comment()
        summary = {
            "events_seen": self.events_seen,
            "event_counts": self.event_counts,
            "latest": self.latest,
            "pr_number": self.pr_number,
            "comment_id": self.comment_id,
        }
        self.output.with_suffix(".summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 0

    def request_stop(self, _signum: int, _frame: object) -> None:
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
