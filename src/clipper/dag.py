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

if os.name == "nt":
    import msvcrt
else:
    import fcntl

_FROM_DISK = object()


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
class StageClaim:
    """An authority-fenced claim with a persistent per-identity attempt counter."""

    acquired: bool
    attempt_count: int


@dataclass(frozen=True, slots=True)
class DagLeaseCoordinator:
    """Owner-bound leases and optional authoritative terminal publication.

    The authoritative adapter's read_committed must not flush a dirty writer view.
    publish_terminal must fence against the active owner and lease expiry before
    committing both the record and output. A rejected publication returns False.
    """

    claim: Callable[[StageIdentity, str, float], bool | StageClaim]
    renew: Callable[[StageIdentity, str, float], bool]
    release: Callable[[StageIdentity, str], bool]
    commit: Callable[[], None]
    reload: Callable[[], None]
    read_committed: Callable[[StageIdentity], dict[str, Any] | None] | None = None
    publish_terminal: Callable[[StageIdentity, str, StageRecord, object], bool] | None = None


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
        if coordinator is not None and (
            (coordinator.read_committed is None) != (coordinator.publish_terminal is None)
        ):
            raise ValueError("coordinated terminal read and publication must be paired")
        self._publication_lock = threading.Lock()
        self._pending_passes: dict[str, tuple[str, Literal["publishing", "uncertain"]]] = {}
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

    @staticmethod
    def _lock_file(descriptor: int, *, acquire: bool) -> None:
        if os.name == "nt":
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK if acquire else msvcrt.LK_UNLCK, 1)
        else:
            fcntl.flock(descriptor, (fcntl.LOCK_EX | fcntl.LOCK_NB) if acquire else fcntl.LOCK_UN)

    @contextmanager
    def _write_lock(
        self,
        identity: StageIdentity,
        *,
        timeout_seconds: float = 30.0,
    ) -> Iterator[None]:
        """OS-managed lock: death/cancellation closes the descriptor, not a mkdir lease."""
        self._prepare_private_stage(identity)
        directory = self._directory(identity)
        legacy = directory / ".write-lock"
        path = directory / ".write-lock-file"
        deadline = time.monotonic() + timeout_seconds
        # The old mkdir lock may still belong to another running code version.
        # Never remove it or claim that it is stale.
        if legacy.exists():
            raise TimeoutError(
                f"legacy DAG lock requires owner intervention: {identity.stage_name}"
            )
        fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if identity.stage_name in self.private_stages:
                os.chmod(path, 0o600)
            if os.name == "nt" and os.fstat(fd).st_size == 0:
                os.write(fd, b"0")
            while True:
                if legacy.exists():
                    raise TimeoutError(
                        f"legacy DAG lock requires owner intervention: {identity.stage_name}"
                    )
                try:
                    self._lock_file(fd, acquire=True)
                    break
                except (BlockingIOError, PermissionError, OSError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"timed out acquiring DAG write lock for {identity.stage_name}"
                        ) from None
                    time.sleep(0.01)
            try:
                yield
            except BaseException:
                with _preserve_primary("write lock release"):
                    self._lock_file(fd, acquire=False)
                raise
            else:
                self._lock_file(fd, acquire=False)
        finally:
            os.close(fd)

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

    @staticmethod
    def _matches_identity(record: dict[str, Any], identity: StageIdentity) -> bool:
        return record.get("identity") == identity.to_dict()

    @staticmethod
    def _complete_pass_record(record: dict[str, Any]) -> bool:
        """Validate all persisted PASS fields; older complete records remain readable."""
        fields = {
            "identity",
            "status",
            "attempt_count",
            "started_at",
            "completed_at",
            "output_fingerprint",
            "usage",
            "cost_usd",
            "error_type",
            "error",
        }
        cost = record.get("cost_usd")
        attempts = record.get("attempt_count")
        fingerprint = record.get("output_fingerprint")
        return (
            set(record) == fields
            and record.get("status") == "PASS"
            and isinstance(attempts, int)
            and not isinstance(attempts, bool)
            and attempts > 0
            and isinstance(record.get("started_at"), str)
            and bool(record["started_at"])
            and isinstance(record.get("completed_at"), str)
            and bool(record["completed_at"])
            and isinstance(fingerprint, str)
            and len(fingerprint) == 64
            and all(character in "0123456789abcdef" for character in fingerprint)
            and isinstance(record.get("usage"), dict)
            and all(isinstance(key, str) for key in record["usage"])
            and isinstance(cost, (int, float))
            and not isinstance(cost, bool)
            and math.isfinite(cost)
            and cost >= 0
            and record.get("error_type") is None
            and record.get("error") is None
        )

    def _validated_pass(
        self,
        identity: StageIdentity,
        record: dict[str, Any] | None,
        *,
        output: object = _FROM_DISK,
    ) -> tuple[object] | None:
        if record is None or not self._complete_pass_record(record):
            return None
        if not self._matches_identity(record, identity):
            return None
        try:
            value: object = (
                json.loads(self._output_path(identity).read_text(encoding="utf-8"))
                if output is _FROM_DISK
                else output
            )
            if content_fingerprint(value) != record["output_fingerprint"]:
                return None
        except (OSError, ValueError, RecursionError, TypeError):
            return None
        return (value,)

    def _cache_hit(self, identity: StageIdentity) -> tuple[bool, object]:
        """Explicit hit bit, so a valid JSON null output is reusable."""
        coordinator = self.coordinator
        if coordinator is not None and coordinator.read_committed is not None:
            snapshot = coordinator.read_committed(identity)
            if snapshot is None:
                return False, None
            if not isinstance(snapshot, dict) or set(snapshot) != {"record", "output"}:
                raise RuntimeError("DAG committed terminal evidence is malformed")
            record = snapshot["record"]
            if not isinstance(record, dict):
                raise RuntimeError("DAG committed terminal record is malformed")
            if record.get("status") == "FAILED":
                return False, None
            validated = self._validated_pass(identity, record, output=snapshot["output"])
            if validated is None:
                raise RuntimeError("DAG committed PASS identity or output is invalid")
            return True, validated[0]
        self._prepare_private_stage(identity)
        with self._publication_lock:
            if identity.cache_key in self._pending_passes:
                return False, None
            try:
                validated = self._validated_pass(identity, self._read_record_payload(identity))
            except OSError:
                return False, None
            return (True, validated[0]) if validated is not None else (False, None)

    def cached_output(self, identity: StageIdentity) -> object | None:
        """Legacy public API: None is ambiguous; execute uses the explicit hit bit."""
        hit, output = self._cache_hit(identity)
        return output if hit else None

    def _may_report_after_pass(
        self, identity: StageIdentity, attempted: StageRecord, *, coordinated: bool
    ) -> bool:
        """Decide from published evidence, not a write/commit acknowledgement alone."""
        record = self._read_record_payload(identity)
        if record is not None and self._validated_pass(identity, record) is not None:
            if content_fingerprint(record) == content_fingerprint(attempted.to_dict()):
                LOGGER.info("DAG PASS publication confirmed for %s", identity.stage_name)
            else:
                LOGGER.warning("DAG preserved a different valid PASS for %s", identity.stage_name)
            return False
        if (
            record is not None
            and self._matches_identity(record, identity)
            and record.get("attempt_count") == attempted.attempt_count
        ):
            if record.get("status") == "RUNNING":
                return True
            if record.get("status") == "FAILED":
                return False
            if not coordinated and record.get("status") == "PASS":
                return True
        LOGGER.warning("DAG PASS publication remains uncertain for %s", identity.stage_name)
        return False

    def execute(
        self,
        identity: StageIdentity,
        operation: Callable[[], StageResult | object],
    ) -> tuple[object, bool]:
        if self.coordinator is not None:
            return self._execute_coordinated(identity, operation)
        return self._execute_local(identity, operation)

    def _reload_snapshot(self, coordinator: DagLeaseCoordinator, identity: StageIdentity) -> None:
        key = identity.cache_key
        with self._publication_lock:
            pending = self._pending_passes.get(key)
        coordinator.reload()
        if pending is not None and pending[1] == "uncertain":
            with self._publication_lock:
                if self._pending_passes.get(key) == pending:
                    self._pending_passes.pop(key, None)

    def _coordinated_cached_output(self, identity: StageIdentity) -> object | None:
        if self.coordinator is None:
            raise RuntimeError("DAG coordinator is unavailable")
        if self.coordinator.read_committed is not None:
            hit, cached = self._cache_hit(identity)
        else:
            self._reload_snapshot(self.coordinator, identity)
            hit, cached = self._cache_hit(identity)
        return cached if hit else None

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

    def _execute_authoritative(
        self,
        identity: StageIdentity,
        operation: Callable[[], StageResult | object],
        coordinator: DagLeaseCoordinator,
    ) -> tuple[object, bool]:
        """Distributed terminal state is published only by the fenced authority.

        Worker mounts never write the canonical terminal record. A lost publish
        acknowledgement is an error, even if a later committed read sees PASS.
        """
        publish = coordinator.publish_terminal
        if publish is None:
            raise RuntimeError("DAG terminal authority is unavailable")
        owner_id = uuid.uuid4().hex
        attempt = 0
        while True:
            hit, cached = self._cache_hit(identity)
            if hit:
                return cached, True
            try:
                receipt = coordinator.claim(identity, owner_id, self.execution_lease_seconds)
            except BaseException:
                with _preserve_primary("authoritative claim release"):
                    self._release_lease_best_effort(coordinator, identity, owner_id)
                raise
            if not isinstance(receipt, StageClaim):
                with _preserve_primary("invalid claim release"):
                    self._release_lease_best_effort(coordinator, identity, owner_id)
                raise RuntimeError("DAG authority must return a fenced claim receipt")
            if receipt.acquired:
                try:
                    attempt = receipt.attempt_count
                    if attempt <= 0:
                        raise RuntimeError("DAG authority returned an invalid attempt number")
                    hit, cached = self._cache_hit(identity)
                except BaseException:
                    with _preserve_primary("authoritative claim release"):
                        self._release_lease_best_effort(coordinator, identity, owner_id)
                    raise
                if hit:
                    self._release_lease_best_effort(coordinator, identity, owner_id)
                    return cached, True
                break
            time.sleep(self.follower_poll_seconds)

        started = _now()
        renewal_stop = threading.Event()
        lease_lost = threading.Event()
        termination_lock = threading.Lock()
        termination: BaseException | None = None

        def renew_worker() -> None:
            nonlocal termination
            interval = min(30.0, max(0.5, self.execution_lease_seconds / 3.0))
            try:
                while not renewal_stop.wait(interval):
                    if not coordinator.renew(identity, owner_id, self.execution_lease_seconds):
                        lease_lost.set()
                        return
            except Exception as exc:
                lease_lost.set()
                LOGGER.warning("DAG authority renewal failed: %s", type(exc).__name__)
            except BaseException as exc:
                with termination_lock:
                    termination = exc

        worker = threading.Thread(target=renew_worker, daemon=True)

        def check_worker() -> None:
            with termination_lock:
                result = termination
            if result is not None:
                raise result
            if lease_lost.is_set():
                raise RuntimeError(
                    f"DAG distributed execution lease lost for {identity.stage_name}"
                )

        def shutdown_worker() -> None:
            renewal_stop.set()
            if worker.ident is not None:
                worker.join(timeout=1.0)
            if worker.is_alive():
                raise RuntimeError("DAG renewal worker did not stop within the shutdown budget")
            check_worker()

        pass_submission_started = False
        try:
            worker.start()
            raw = operation()
            check_worker()
            result = raw if isinstance(raw, StageResult) else StageResult(raw)
            owns_lease = coordinator.renew(identity, owner_id, self.execution_lease_seconds)
            check_worker()
            if not owns_lease:
                raise RuntimeError(
                    f"DAG distributed execution lease lost before PASS for {identity.stage_name}"
                )
            passed = StageRecord(
                identity=identity,
                status="PASS",
                attempt_count=attempt,
                started_at=started,
                completed_at=_now(),
                output_fingerprint=content_fingerprint(result.output),
                usage=dict(result.usage),
                cost_usd=result.cost_usd,
            )
            check_worker()
            pass_submission_started = True
            if not publish(identity, owner_id, passed, result.output):
                raise RuntimeError("DAG owner-fenced PASS publication rejected")
            # The authority may have committed while the worker was terminating.
            # A successful acknowledgement cannot mask that termination or lease loss.
            shutdown_worker()
        except BaseException as exc:
            with _preserve_primary("authoritative renewal shutdown"):
                shutdown_worker()
            if not pass_submission_started:
                with _preserve_primary("authoritative FAILED publication"):
                    if not lease_lost.is_set() and coordinator.renew(
                        identity, owner_id, self.execution_lease_seconds
                    ):
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
                        publish(identity, owner_id, failed, None)
            # After a submitted PASS, acknowledgement may be lost. Never replace
            # its unknown durable result with FAILED as part of cleanup.
            with _preserve_primary("authoritative claim release"):
                self._release_lease_best_effort(coordinator, identity, owner_id)
            raise
        else:
            self._release_lease_best_effort(coordinator, identity, owner_id)
            return result.output, False

    def _execute_coordinated(
        self,
        identity: StageIdentity,
        operation: Callable[[], StageResult | object],
    ) -> tuple[object, bool]:
        coordinator = self.coordinator
        if coordinator is None:
            raise RuntimeError("DAG coordinator is unavailable")
        if coordinator.read_committed is not None:
            return self._execute_authoritative(identity, operation, coordinator)
        owner_id = uuid.uuid4().hex
        attempt = 0
        started = ""

        renewal_stop = threading.Event()
        lease_lost = threading.Event()
        termination_lock = threading.Lock()
        worker_termination: BaseException | None = None

        def check_worker_termination() -> None:
            with termination_lock:
                termination = worker_termination
            if termination is not None:
                raise termination

        def renew_lease() -> None:
            nonlocal worker_termination
            interval = min(30.0, max(0.5, self.execution_lease_seconds / 3.0))
            try:
                while not renewal_stop.wait(interval):
                    if not coordinator.renew(identity, owner_id, self.execution_lease_seconds):
                        lease_lost.set()
                        return
            except Exception as exc:
                lease_lost.set()
                LOGGER.warning("DAG renewal stopped: %s", type(exc).__name__)
            except BaseException as exc:
                with termination_lock:
                    worker_termination = exc
                LOGGER.warning("DAG renewal terminated: %s", type(exc).__name__)

        renewal_thread = threading.Thread(target=renew_lease, daemon=True)

        def stop_renewal() -> None:
            renewal_stop.set()
            if renewal_thread.ident is not None:
                renewal_thread.join(timeout=1.0)
            if renewal_thread.is_alive():
                lease_lost.set()
                raise RuntimeError("DAG renewal worker did not stop within the shutdown budget")

        while True:
            if coordinator.read_committed is None:
                self._reload_snapshot(coordinator, identity)
            hit, cached = self._cache_hit(identity)
            if hit:
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
                    self._reload_snapshot(coordinator, identity)
                    hit, cached = self._cache_hit(identity)
                    if not hit:
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
                if hit:
                    self._release_lease_best_effort(coordinator, identity, owner_id)
                    return cached, True
                break
            time.sleep(self.follower_poll_seconds)

        attempted_pass: StageRecord | None = None
        try:
            renewal_thread.start()
            raw_result = operation()
            check_worker_termination()
            result = raw_result if isinstance(raw_result, StageResult) else StageResult(raw_result)
            renewed = not lease_lost.is_set() and coordinator.renew(
                identity, owner_id, self.execution_lease_seconds
            )
            check_worker_termination()
            if not renewed or lease_lost.is_set():
                raise RuntimeError(
                    f"DAG distributed execution lease lost before PASS for {identity.stage_name}"
                )
            fingerprint = content_fingerprint(result.output)
            self._write_json(self._output_path(identity), result.output)
            check_worker_termination()
            attempted_pass = StageRecord(
                identity=identity,
                status="PASS",
                attempt_count=attempt,
                started_at=started,
                completed_at=_now(),
                output_fingerprint=fingerprint,
                usage=dict(result.usage),
                cost_usd=result.cost_usd,
            )
            with self._publication_lock:
                self._pending_passes[identity.cache_key] = (owner_id, "publishing")
            self._write_json(self._record_path(identity), attempted_pass.to_dict())
            check_worker_termination()
            coordinator.commit()
            with self._publication_lock:
                if self._pending_passes.get(identity.cache_key) == (owner_id, "publishing"):
                    self._pending_passes.pop(identity.cache_key, None)
        except BaseException as exc:
            with _preserve_primary("PASS acknowledgement tracking"), self._publication_lock:
                if self._pending_passes.get(identity.cache_key) == (owner_id, "publishing"):
                    self._pending_passes[identity.cache_key] = (owner_id, "uncertain")
            with _preserve_primary("renewal shutdown"):
                stop_renewal()
            with _preserve_primary("FAILED record publication"):
                # A local PASS does not establish durable publication after a failed commit.
                may_report = True
                if attempted_pass is not None:
                    self._reload_snapshot(coordinator, identity)
                    may_report = self._may_report_after_pass(
                        identity, attempted_pass, coordinated=True
                    )
                if (
                    may_report
                    and not lease_lost.is_set()
                    and coordinator.renew(identity, owner_id, self.execution_lease_seconds)
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
                check_worker_termination()
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
            hit, cached = self._cache_hit(identity)
            if hit:
                return cached, True

            claimed = False
            try:
                with self._write_lock(identity):
                    hit, cached = self._cache_hit(identity)
                    if hit:
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

        attempted_pass: StageRecord | None = None
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
                attempted_pass = StageRecord(
                    identity=identity,
                    status="PASS",
                    attempt_count=attempt,
                    started_at=started,
                    completed_at=_now(),
                    output_fingerprint=fingerprint,
                    usage=dict(result.usage),
                    cost_usd=result.cost_usd,
                )
                self._write_json(self._record_path(identity), attempted_pass.to_dict())
                self._release_local_claim(identity, owner_id)
            return result.output, False
        except BaseException as exc:
            with _preserve_primary("local failure cleanup"), self._write_lock(identity):
                if self._claim_owner(self._read_execution_claim(identity)) == owner_id:
                    with _preserve_primary("FAILED record publication"):
                        if attempted_pass is None or self._may_report_after_pass(
                            identity, attempted_pass, coordinated=False
                        ):
                            self._record_failure(identity, attempt, started, exc)
                    with _preserve_primary("local claim release"):
                        self._release_local_claim(identity, owner_id)
            raise
