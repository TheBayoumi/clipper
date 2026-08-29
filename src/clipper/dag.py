from __future__ import annotations

import json
import math
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .stage_contracts import StageIdentity, content_fingerprint

StageStatus = Literal["RUNNING", "PASS", "FAILED"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class StageResult:
    output: object
    usage: dict[str, object] = field(default_factory=dict)
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_usd < 0:
            raise ValueError("stage cost cannot be negative")


@dataclass(frozen=True, slots=True)
class StageRecord:
    identity: StageIdentity
    status: StageStatus
    attempt_count: int
    started_at: str
    completed_at: str | None
    output_fingerprint: str | None
    usage: dict[str, object]
    cost_usd: float
    error_type: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.attempt_count <= 0:
            raise ValueError("stage attempt_count must be positive")
        if self.cost_usd < 0:
            raise ValueError("stage cost cannot be negative")
        if self.status == "PASS" and (not self.completed_at or not self.output_fingerprint):
            raise ValueError("passing stage requires completion and output evidence")
        if self.status == "FAILED" and (not self.completed_at or not self.error_type):
            raise ValueError("failed stage requires completion and error evidence")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["identity"] = self.identity.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class DagLeaseCoordinator:
    claim: Callable[[StageIdentity, str, float], bool]
    renew: Callable[[StageIdentity, str, float], bool]
    release: Callable[[StageIdentity, str], bool]
    commit: Callable[[], None]
    reload: Callable[[], None]


class DagStore:
    """Persistent content-addressed stage store with exact dependency reuse."""

    def __init__(
        self,
        root: str | Path,
        *,
        execution_lease_seconds: float = 86_400.0,
        follower_poll_seconds: float = 0.05,
        coordinator: DagLeaseCoordinator | None = None,
    ) -> None:
        self.root = Path(root)
        self.execution_lease_seconds = float(execution_lease_seconds)
        self.follower_poll_seconds = float(follower_poll_seconds)
        self.coordinator = coordinator
        if not math.isfinite(self.execution_lease_seconds) or self.execution_lease_seconds <= 0:
            raise ValueError("DAG execution lease must be finite and positive")
        if not math.isfinite(self.follower_poll_seconds) or self.follower_poll_seconds <= 0:
            raise ValueError("DAG follower poll interval must be finite and positive")

    def _directory(self, identity: StageIdentity) -> Path:
        safe_stage = "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in identity.stage_name
        )
        return self.root / safe_stage / identity.cache_key

    def _record_path(self, identity: StageIdentity) -> Path:
        return self._directory(identity) / "stage.json"

    def _output_path(self, identity: StageIdentity) -> Path:
        return self._directory(identity) / "output.json"

    def _execution_claim_path(self, identity: StageIdentity) -> Path:
        return self._directory(identity) / "execution-claim.json"

    def _read_execution_claim(self, identity: StageIdentity) -> dict[str, Any] | None:
        path = self._execution_claim_path(identity)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    @staticmethod
    def _claim_owner(claim: dict[str, Any] | None) -> str:
        return str(claim.get("owner_id") or "") if claim is not None else ""

    @staticmethod
    def _claim_expired(claim: dict[str, Any] | None, *, now: float) -> bool:
        if claim is None:
            return True
        expires_at = claim.get("expires_at")
        if not isinstance(expires_at, (int, float)) or isinstance(expires_at, bool):
            return True
        return not math.isfinite(float(expires_at)) or float(expires_at) <= now

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _write_lock(
        self,
        identity: StageIdentity,
        *,
        timeout_seconds: float = 30.0,
    ) -> Iterator[None]:
        lock = self._directory(identity) / ".write-lock"
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                lock.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out acquiring DAG write lock for {identity.stage_name}"
                    ) from None
                time.sleep(0.01)
        try:
            yield
        finally:
            lock.rmdir()

    def _read_record_payload(self, identity: StageIdentity) -> dict[str, Any] | None:
        path = self._record_path(identity)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def attempt_count(self, identity: StageIdentity) -> int:
        payload = self._read_record_payload(identity)
        if payload is None:
            return 0
        raw = payload.get("attempt_count")
        return int(raw) if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else 0

    def cached_output(self, identity: StageIdentity) -> object | None:
        record = self._read_record_payload(identity)
        output_path = self._output_path(identity)
        if record is None or record.get("status") != "PASS" or not output_path.is_file():
            return None
        if record.get("identity", {}).get("cache_key") != identity.cache_key:
            return None
        try:
            output: object = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expected = record.get("output_fingerprint")
        if not isinstance(expected, str) or content_fingerprint(output) != expected:
            return None
        return output

    def execute(
        self,
        identity: StageIdentity,
        operation: Callable[[], StageResult | object],
    ) -> tuple[object, bool]:
        if self.coordinator is not None:
            return self._execute_coordinated(identity, operation)
        return self._execute_local(identity, operation)

    def _coordinated_cached_output(self, identity: StageIdentity) -> object | None:
        if self.coordinator is None:
            raise RuntimeError("DAG coordinator is unavailable")
        self.coordinator.reload()
        return self.cached_output(identity)

    def _execute_coordinated(
        self,
        identity: StageIdentity,
        operation: Callable[[], StageResult | object],
    ) -> tuple[object, bool]:
        coordinator = self.coordinator
        if coordinator is None:
            raise RuntimeError("DAG coordinator is unavailable")
        owner_id = uuid.uuid4().hex
        attempt = 0
        started = ""

        while True:
            cached = self._coordinated_cached_output(identity)
            if cached is not None:
                return cached, True
            if coordinator.claim(identity, owner_id, self.execution_lease_seconds):
                coordinator.reload()
                cached = self.cached_output(identity)
                if cached is not None:
                    coordinator.release(identity, owner_id)
                    return cached, True
                attempt = self.attempt_count(identity) + 1
                started = _now()
                running = StageRecord(
                    identity=identity,
                    status="RUNNING",
                    attempt_count=attempt,
                    started_at=started,
                    completed_at=None,
                    output_fingerprint=None,
                    usage={},
                    cost_usd=0.0,
                )
                self._write_json(self._record_path(identity), running.to_dict())
                coordinator.commit()
                break
            time.sleep(self.follower_poll_seconds)

        renewal_stop = threading.Event()
        lease_lost = threading.Event()

        def renew_lease() -> None:
            interval = min(30.0, max(0.5, self.execution_lease_seconds / 3.0))
            while not renewal_stop.wait(interval):
                try:
                    if not coordinator.renew(identity, owner_id, self.execution_lease_seconds):
                        lease_lost.set()
                        return
                except Exception:
                    lease_lost.set()
                    return

        renewal_thread = threading.Thread(target=renew_lease, daemon=True)
        renewal_thread.start()
        try:
            raw_result = operation()
            result = raw_result if isinstance(raw_result, StageResult) else StageResult(raw_result)
            if lease_lost.is_set() or not coordinator.renew(
                identity, owner_id, self.execution_lease_seconds
            ):
                raise RuntimeError(
                    f"DAG distributed execution lease lost before PASS for {identity.stage_name}"
                )
            fingerprint = content_fingerprint(result.output)
            self._write_json(self._output_path(identity), result.output)
            passed = StageRecord(
                identity=identity,
                status="PASS",
                attempt_count=attempt,
                started_at=started,
                completed_at=_now(),
                output_fingerprint=fingerprint,
                usage=dict(result.usage),
                cost_usd=result.cost_usd,
            )
            self._write_json(self._record_path(identity), passed.to_dict())
            coordinator.commit()
            if not coordinator.release(identity, owner_id):
                raise RuntimeError(
                    f"DAG distributed execution lease release failed for {identity.stage_name}"
                )
            return result.output, False
        except Exception as exc:
            if not lease_lost.is_set():
                try:
                    owns_lease = coordinator.renew(
                        identity, owner_id, self.execution_lease_seconds
                    )
                except Exception:
                    owns_lease = False
                if owns_lease:
                    failed = StageRecord(
                        identity=identity,
                        status="FAILED",
                        attempt_count=attempt,
                        started_at=started,
                        completed_at=_now(),
                        output_fingerprint=None,
                        usage={},
                        cost_usd=0.0,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    self._write_json(self._record_path(identity), failed.to_dict())
                    coordinator.commit()
                    coordinator.release(identity, owner_id)
            raise
        finally:
            renewal_stop.set()
            renewal_thread.join(timeout=1.0)

    def _execute_local(
        self,
        identity: StageIdentity,
        operation: Callable[[], StageResult | object],
    ) -> tuple[object, bool]:
        owner_id = uuid.uuid4().hex
        attempt = 0
        started = ""

        while True:
            cached = self.cached_output(identity)
            if cached is not None:
                return cached, True

            claimed = False
            with self._write_lock(identity):
                cached = self.cached_output(identity)
                if cached is not None:
                    return cached, True

                now = time.time()
                claim = self._read_execution_claim(identity)
                if self._claim_expired(claim, now=now):
                    attempt = self.attempt_count(identity) + 1
                    started = _now()
                    self._write_json(
                        self._execution_claim_path(identity),
                        {
                            "owner_id": owner_id,
                            "claimed_at": now,
                            "expires_at": now + self.execution_lease_seconds,
                        },
                    )
                    running = StageRecord(
                        identity=identity,
                        status="RUNNING",
                        attempt_count=attempt,
                        started_at=started,
                        completed_at=None,
                        output_fingerprint=None,
                        usage={},
                        cost_usd=0.0,
                    )
                    self._write_json(self._record_path(identity), running.to_dict())
                    claimed = True

            if claimed:
                break
            time.sleep(self.follower_poll_seconds)

        try:
            raw_result = operation()
            result = raw_result if isinstance(raw_result, StageResult) else StageResult(raw_result)
            fingerprint = content_fingerprint(result.output)
            with self._write_lock(identity):
                claim = self._read_execution_claim(identity)
                if self._claim_owner(claim) != owner_id:
                    raise RuntimeError(
                        f"DAG execution lease lost before PASS for {identity.stage_name}"
                    )
                self._write_json(self._output_path(identity), result.output)
                passed = StageRecord(
                    identity=identity,
                    status="PASS",
                    attempt_count=attempt,
                    started_at=started,
                    completed_at=_now(),
                    output_fingerprint=fingerprint,
                    usage=dict(result.usage),
                    cost_usd=result.cost_usd,
                )
                self._write_json(self._record_path(identity), passed.to_dict())
                self._execution_claim_path(identity).unlink(missing_ok=True)
            return result.output, False
        except Exception as exc:
            with self._write_lock(identity):
                claim = self._read_execution_claim(identity)
                if self._claim_owner(claim) == owner_id:
                    failed = StageRecord(
                        identity=identity,
                        status="FAILED",
                        attempt_count=attempt,
                        started_at=started,
                        completed_at=_now(),
                        output_fingerprint=None,
                        usage={},
                        cost_usd=0.0,
                        error_type=type(exc).__name__,
                        error=str(exc),
                    )
                    self._write_json(self._record_path(identity), failed.to_dict())
                    self._execution_claim_path(identity).unlink(missing_ok=True)
            raise
