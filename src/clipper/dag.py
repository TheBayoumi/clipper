from __future__ import annotations

import json
import logging
import math
import os
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

LOGGER = logging.getLogger("clipper.dag")

StageStatus = Literal["RUNNING", "PASS", "FAILED"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@contextmanager
def _preserve_primary(action: str) -> Iterator[None]:
    """Guard secondary cleanup only while an exception is already active."""
    try:
        yield
    except BaseException as exc:
        LOGGER.warning("DAG %s failed during error cleanup: %s", action, type(exc).__name__)


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
    """Owner-bound lease callbacks; release must refuse a different owner's lease."""

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
        private_stages: frozenset[str] = frozenset(),
    ) -> None:
        self.root = Path(root)
        self.execution_lease_seconds = float(execution_lease_seconds)
        self.follower_poll_seconds = float(follower_poll_seconds)
        self.coordinator = coordinator
        self.private_stages = private_stages
        self._private_stage_dirs = frozenset(
            self._stage_directory_name(stage) for stage in private_stages
        )
        if not math.isfinite(self.execution_lease_seconds) or self.execution_lease_seconds <= 0:
            raise ValueError("DAG execution lease must be finite and positive")
        if not math.isfinite(self.follower_poll_seconds) or self.follower_poll_seconds <= 0:
            raise ValueError("DAG follower poll interval must be finite and positive")

    @staticmethod
    def _stage_directory_name(stage_name: str) -> str:
        return "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in stage_name
        )

    def _directory(self, identity: StageIdentity) -> Path:
        return self.root / self._stage_directory_name(identity.stage_name) / identity.cache_key

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
    def _private_directory(path: Path) -> None:
        if path.is_symlink():
            raise ValueError("private DAG directory cannot be a symbolic link")
        path.mkdir(mode=0o700, parents=False, exist_ok=True)
        os.chmod(path, 0o700)

    def protect_stage(self, stage_name: str) -> None:
        """Restrict pre-existing content without rewriting its bytes or identities."""
        if stage_name not in self.private_stages:
            raise ValueError("DAG stage is not configured for private persistence")
        self.root.parent.mkdir(parents=True, exist_ok=True)
        self._private_directory(self.root)
        stage = self.root / self._stage_directory_name(stage_name)
        self._private_directory(stage)
        for directory in stage.iterdir():
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("unexpected entry in private DAG stage")
            self._private_directory(directory)
            self._protect_artifacts(directory)

    @staticmethod
    def _protect_artifacts(directory: Path) -> None:
        for path in directory.iterdir():
            if path.is_symlink():
                raise ValueError("private DAG artifact cannot be a symbolic link")
            try:
                if path.is_dir():
                    os.chmod(path, 0o700)
                elif path.is_file():
                    os.chmod(path, 0o600)
            except FileNotFoundError:
                # A concurrent atomic writer may have replaced a temp or claim.
                continue

    def _prepare_private_stage(self, identity: StageIdentity) -> None:
        if identity.stage_name in self.private_stages:
            self.root.parent.mkdir(parents=True, exist_ok=True)
            self._private_directory(self.root)
            self._private_directory(self._directory(identity).parent)
            directory = self._directory(identity)
            self._private_directory(directory)
            self._protect_artifacts(directory)

    def _write_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
            if path.parent.parent.name in self._private_stage_dirs:
                descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                    handle.write(serialized)
            else:
                temporary.write_text(serialized, encoding="utf-8")
            temporary.replace(path)
        except BaseException:
            with _preserve_primary("temporary file removal"):
                temporary.unlink(missing_ok=True)
            raise
        else:
            temporary.unlink(missing_ok=True)

    @contextmanager
    def _write_lock(
        self,
        identity: StageIdentity,
        *,
        timeout_seconds: float = 30.0,
    ) -> Iterator[None]:
        self._prepare_private_stage(identity)
        lock = self._directory(identity) / ".write-lock"
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                lock.mkdir(
                    parents=True,
                    exist_ok=False,
                    mode=0o700 if identity.stage_name in self.private_stages else 0o777,
                )
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"timed out acquiring DAG write lock for {identity.stage_name}"
                    ) from None
                time.sleep(0.01)
        try:
            yield
        except BaseException:
            with _preserve_primary("write lock removal"):
                lock.rmdir()
            raise
        else:
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
        self._prepare_private_stage(identity)
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

    def _release_lease_best_effort(
        self,
        coordinator: DagLeaseCoordinator,
        identity: StageIdentity,
        owner_id: str,
    ) -> None:
        for release_attempt in range(3):
            try:
                if coordinator.release(identity, owner_id):
                    return
            except Exception as exc:
                LOGGER.warning(
                    "DAG lease cleanup failed for %s: %s",
                    identity.stage_name,
                    exc,
                )
            if release_attempt < 2:
                time.sleep(0.05 * (release_attempt + 1))
        LOGGER.warning("DAG lease cleanup not acknowledged for %s", identity.stage_name)

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

        renewal_stop = threading.Event()
        lease_lost = threading.Event()

        def renew_lease() -> None:
            interval = min(30.0, max(0.5, self.execution_lease_seconds / 3.0))
            while not renewal_stop.wait(interval):
                try:
                    if not coordinator.renew(identity, owner_id, self.execution_lease_seconds):
                        lease_lost.set()
                        return
                except BaseException as exc:
                    lease_lost.set()
                    LOGGER.warning("DAG renewal stopped: %s", type(exc).__name__)
                    return

        renewal_thread = threading.Thread(target=renew_lease, daemon=True)

        def stop_renewal() -> None:
            renewal_stop.set()
            if renewal_thread.ident is not None:
                renewal_thread.join(timeout=1.0)

        while True:
            cached = self._coordinated_cached_output(identity)
            if cached is not None:
                return cached, True
            try:
                claimed = coordinator.claim(identity, owner_id, self.execution_lease_seconds)
            except BaseException:
                # A failed acknowledgement does not prove ownership was never persisted.
                with _preserve_primary("lease release"):
                    self._release_lease_best_effort(coordinator, identity, owner_id)
                raise
            if claimed:
                try:
                    coordinator.reload()
                    cached = self.cached_output(identity)
                    if cached is None:
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
                except BaseException:
                    with _preserve_primary("lease release"):
                        self._release_lease_best_effort(coordinator, identity, owner_id)
                    raise
                if cached is not None:
                    self._release_lease_best_effort(coordinator, identity, owner_id)
                    return cached, True
                break
            time.sleep(self.follower_poll_seconds)

        try:
            renewal_thread.start()
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
        except BaseException as exc:
            with _preserve_primary("renewal shutdown"):
                stop_renewal()
            with _preserve_primary("FAILED record publication"):
                if not lease_lost.is_set() and coordinator.renew(
                    identity, owner_id, self.execution_lease_seconds
                ):
                    self._record_failure(identity, attempt, started, exc)
                    coordinator.commit()
            with _preserve_primary("lease release"):
                self._release_lease_best_effort(coordinator, identity, owner_id)
            raise
        else:
            # A committed PASS remains valid, but new cancellation must propagate.
            try:
                stop_renewal()
            except BaseException:
                with _preserve_primary("lease release"):
                    self._release_lease_best_effort(coordinator, identity, owner_id)
                raise
            self._release_lease_best_effort(coordinator, identity, owner_id)
            return result.output, False

    def _record_failure(
        self, identity: StageIdentity, attempt: int, started: str, exc: BaseException
    ) -> None:
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

    def _release_local_claim(self, identity: StageIdentity, owner_id: str) -> None:
        """Called with the write lock held; never remove a different owner's claim."""
        if self._claim_owner(self._read_execution_claim(identity)) == owner_id:
            self._execution_claim_path(identity).unlink(missing_ok=True)

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
            try:
                with self._write_lock(identity):
                    cached = self.cached_output(identity)
                    if cached is not None:
                        return cached, True

                    now = time.time()
                    claim = self._read_execution_claim(identity)
                    if self._claim_expired(claim, now=now):
                        attempt = self.attempt_count(identity) + 1
                        started = _now()
                        try:
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
                        except BaseException:
                            with _preserve_primary("initial claim release"):
                                self._release_local_claim(identity, owner_id)
                            raise
                        claimed = True
            except BaseException:
                if claimed:
                    # Initial publication succeeded but exiting its lock failed.
                    with (
                        _preserve_primary("initial claim release after lock failure"),
                        self._write_lock(identity, timeout_seconds=0.0),
                    ):
                        self._release_local_claim(identity, owner_id)
                raise

            if claimed:
                break
            time.sleep(self.follower_poll_seconds)

        committed = False
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
                committed = True
                self._release_local_claim(identity, owner_id)
            return result.output, False
        except BaseException as exc:
            with _preserve_primary("local failure cleanup"), self._write_lock(identity):
                if self._claim_owner(self._read_execution_claim(identity)) == owner_id:
                    if not committed:
                        with _preserve_primary("FAILED record publication"):
                            self._record_failure(identity, attempt, started, exc)
                    with _preserve_primary("local claim release"):
                        self._release_local_claim(identity, owner_id)
            raise
