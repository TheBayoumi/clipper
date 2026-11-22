from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from clipper.dag import DagStore, StageResult
from clipper.stage_contracts import StageIdentity


def _identity() -> StageIdentity:
    return StageIdentity(
        stage_name="concurrent-stage",
        source_hash="source-hash",
        contract_hash="contract-hash",
        model_revision="model-revision",
    )


def test_concurrent_followers_never_execute_duplicate_paid_operation(tmp_path) -> None:
    store = DagStore(tmp_path)
    identity = _identity()
    calls = {"owner": 0, "follower": 0}

    def succeed():
        calls["owner"] += 1
        time.sleep(0.03)
        return StageResult({"status": "PASS", "value": 1})

    def must_not_run():
        calls["follower"] += 1
        raise AssertionError("follower must wait for the owner's terminal record")

    with ThreadPoolExecutor(max_workers=2) as pool:
        success_future = pool.submit(store.execute, identity, succeed)
        time.sleep(0.005)
        follower_future = pool.submit(store.execute, identity, must_not_run)
        left = success_future.result(timeout=5)
        right = follower_future.result(timeout=5)

    assert left == ({"status": "PASS", "value": 1}, False)
    assert right == ({"status": "PASS", "value": 1}, True)
    assert calls == {"owner": 1, "follower": 0}
    assert store.cached_output(identity) == {"status": "PASS", "value": 1}

    directory = store._directory(identity)
    assert not (directory / ".write-lock").exists()
    assert not store._execution_claim_path(identity).exists()
    assert not list(directory.glob("*.tmp"))
    assert not list(directory.glob(".*.tmp"))


def test_stale_dag_lock_is_not_reclaimed_unsafely(tmp_path) -> None:
    store = DagStore(tmp_path)
    identity = _identity()
    lock = store._directory(identity) / ".write-lock"
    lock.mkdir(parents=True)

    with (
        pytest.raises(TimeoutError, match="timed out acquiring DAG write lock"),
        store._write_lock(identity, timeout_seconds=0.02),
    ):
        raise AssertionError("stale lock must never be stolen")

    assert lock.is_dir()


def test_expired_execution_claim_is_recovered_as_new_attempt(tmp_path) -> None:
    store = DagStore(tmp_path, execution_lease_seconds=0.05, follower_poll_seconds=0.001)
    identity = _identity()
    directory = store._directory(identity)
    directory.mkdir(parents=True, exist_ok=True)
    store._write_json(
        store._execution_claim_path(identity),
        {"owner_id": "dead-owner", "claimed_at": 0.0, "expires_at": 0.0},
    )
    store._write_json(
        store._record_path(identity),
        {
            "identity": identity.to_dict(),
            "status": "RUNNING",
            "attempt_count": 1,
            "started_at": "2026-08-29T00:00:00+00:00",
            "completed_at": None,
            "output_fingerprint": None,
            "usage": {},
            "cost_usd": 0.0,
            "error_type": None,
            "error": None,
        },
    )

    output, cached = store.execute(identity, lambda: {"value": 2})

    assert output == {"value": 2}
    assert cached is False
    record = store._read_record_payload(identity)
    assert record is not None
    assert record["status"] == "PASS"
    assert record["attempt_count"] == 2
    assert not store._execution_claim_path(identity).exists()


def test_private_dag_creation_resume_and_concurrent_writers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = StageIdentity(
        stage_name="private.ingest", source_hash="source-hash", contract_hash="contract-hash"
    )
    root = tmp_path / "private-dag"
    store = DagStore(root, private_stages=frozenset({identity.stage_name}))
    calls: list[int] = []
    original_replace = Path.replace

    def inspect_replace(path: Path, target: Path) -> Path:
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.parent.stat().st_mode & 0o777 == 0o700
        return original_replace(path, target)

    def operation() -> object:
        calls.append(1)
        time.sleep(0.03)
        return {"synthetic": "private"}

    previous = os.umask(0)
    try:
        monkeypatch.setattr(Path, "replace", inspect_replace)
        store.protect_stage(identity.stage_name)
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda _: store.execute(identity, operation), range(3)))
        assert len(calls) == 1
        assert sum(not reused for _, reused in results) == 1
        for directory in (root, root / "private-ingest", store._directory(identity)):
            assert directory.stat().st_mode & 0o777 == 0o700
        files = list(store._directory(identity).glob("*.json"))
        assert files
        hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        for path in files:
            assert path.stat().st_mode & 0o777 == 0o600
            path.chmod(0o644)
        store._directory(identity).chmod(0o755)
        store.protect_stage(identity.stage_name)
        assert store.execute(identity, operation)[1]
        assert len(calls) == 1
        for path, digest in hashes.items():
            assert path.stat().st_mode & 0o777 == 0o600
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        assert store._directory(identity).stat().st_mode & 0o777 == 0o700
    finally:
        os.umask(previous)


def test_private_dag_failure_keeps_record_private_and_no_temporary_files(tmp_path: Path) -> None:
    identity = _identity()
    store = DagStore(tmp_path / "private", private_stages=frozenset({identity.stage_name}))

    def fail() -> object:
        raise ValueError("synthetic failure")

    previous = os.umask(0)
    try:
        with pytest.raises(ValueError, match="synthetic failure"):
            store.execute(identity, fail)
        record = store._record_path(identity)
        assert record.stat().st_mode & 0o777 == 0o600
        assert not store._output_path(identity).exists()
        assert not list(record.parent.glob("*.tmp"))
        assert not store._execution_claim_path(identity).exists()
    finally:
        os.umask(previous)


def test_private_dag_output_rename_failure_removes_temporary_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    store = DagStore(tmp_path / "private", private_stages=frozenset({identity.stage_name}))
    original_replace = Path.replace

    def fail_output(path: Path, target: Path) -> Path:
        assert path.stat().st_mode & 0o777 == 0o600
        if target.name == "output.json":
            raise OSError("synthetic rename failure")
        return original_replace(path, target)

    previous = os.umask(0)
    try:
        monkeypatch.setattr(Path, "replace", fail_output)
        with pytest.raises(OSError, match="synthetic rename failure"):
            store.execute(identity, lambda: {"synthetic": 1})
        directory = store._directory(identity)
        assert not list(directory.glob(".*.tmp"))
        assert not store._output_path(identity).exists()
        assert not store._execution_claim_path(identity).exists()
        assert store._record_path(identity).stat().st_mode & 0o777 == 0o600
    finally:
        os.umask(previous)


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("replacement_owner", [False, True])
@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_initial_running_rename_failure_releases_only_owned_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    private: bool,
    replacement_owner: bool,
    failure_type: type[BaseException],
) -> None:
    identity = StageIdentity(
        stage_name="books.ingest", source_hash="source-hash", contract_hash="contract-hash"
    )
    store = DagStore(
        tmp_path / "dag",
        private_stages=frozenset({identity.stage_name}) if private else frozenset(),
    )
    original_replace = Path.replace
    calls: list[int] = []

    def operation() -> object:
        calls.append(1)
        return {"synthetic": True}

    def fail_running(path: Path, target: Path) -> Path:
        if target.name == "stage.json":
            assert store._execution_claim_path(identity).is_file()
            if private:
                assert path.stat().st_mode & 0o777 == 0o600
                assert store._execution_claim_path(identity).stat().st_mode & 0o777 == 0o600
            if replacement_owner:
                store._write_json(
                    store._execution_claim_path(identity),
                    {"owner_id": "other-owner", "expires_at": time.time() + 86_400},
                )
            raise failure_type("synthetic initial RUNNING rename failure")
        return original_replace(path, target)

    previous = os.umask(0)
    try:
        monkeypatch.setattr(Path, "replace", fail_running)
        with pytest.raises(failure_type, match="initial RUNNING rename failure"):
            store.execute(identity, operation)
        directory = store._directory(identity)
        assert calls == []
        assert not store._record_path(identity).exists()
        assert not store._output_path(identity).exists()
        assert not (directory / ".write-lock").exists()
        assert not list(directory.glob(".*.tmp"))
        if replacement_owner:
            claim = store._read_execution_claim(identity)
            assert claim is not None
            assert claim["owner_id"] == "other-owner"
            return
        assert not store._execution_claim_path(identity).exists()
        if private:
            assert directory.stat().st_mode & 0o777 == 0o700
        monkeypatch.setattr(Path, "replace", original_replace)
        # A stranded claim would enter the wait path; never wait for the lease.

        def unexpected_sleep(_seconds: float) -> None:
            raise AssertionError("retry waited on a stranded claim")

        monkeypatch.setattr(time, "sleep", unexpected_sleep)
        assert store.execute(identity, operation) == ({"synthetic": True}, False)
        assert calls == [1]
        assert not store._execution_claim_path(identity).exists()
    finally:
        os.umask(previous)


@pytest.mark.parametrize("private", [False, True])
@pytest.mark.parametrize("primary_at", ["operation", "output"])
@pytest.mark.parametrize("primary_type", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("secondary_type", [OSError, KeyboardInterrupt])
def test_local_double_fault_preserves_primary_and_releases_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    private: bool,
    primary_at: str,
    primary_type: type[BaseException],
    secondary_type: type[BaseException],
) -> None:
    identity = StageIdentity(
        stage_name="books.ingest", source_hash="source-hash", contract_hash="contract-hash"
    )
    store = DagStore(
        tmp_path / "dag",
        private_stages=frozenset({identity.stage_name}) if private else frozenset(),
    )
    primary = primary_type("synthetic primary")
    original_replace = Path.replace
    operations: list[int] = []
    reports: list[int] = []

    def operation() -> object:
        operations.append(1)
        if primary_at == "operation":
            raise primary
        return {"synthetic": True}

    def fail_publication(path: Path, target: Path) -> Path:
        if target.name == "output.json":
            raise primary
        if target.name == "stage.json" and json.loads(path.read_text())["status"] == "FAILED":
            reports.append(1)
            assert store._execution_claim_path(identity).exists()
            raise secondary_type("synthetic secondary report failure")
        return original_replace(path, target)

    previous = os.umask(0)
    try:
        with monkeypatch.context() as scoped:
            scoped.setattr(Path, "replace", fail_publication)
            with pytest.raises(primary_type) as error:
                store.execute(identity, operation)
        assert error.value is primary
        assert operations == reports == [1]
        assert store.attempt_count(identity) == 1
        assert store.cached_output(identity) is None
        assert not store._output_path(identity).exists()
        assert not store._execution_claim_path(identity).exists()
        directory = store._directory(identity)
        assert not (directory / ".write-lock").exists()
        assert not list(directory.glob(".*.tmp"))
        if private:
            assert directory.stat().st_mode & 0o777 == 0o700
            assert store._record_path(identity).stat().st_mode & 0o777 == 0o600

        def unexpected_sleep(_seconds: float) -> None:
            raise AssertionError("retry waited on a stranded claim")

        monkeypatch.setattr(time, "sleep", unexpected_sleep)
        assert store.execute(identity, lambda: {"retry": True}) == ({"retry": True}, False)
        assert store.attempt_count(identity) == 2
        assert not store._execution_claim_path(identity).exists()
    finally:
        os.umask(previous)


@pytest.mark.parametrize("cleanup", ["other-owner", "unlink", "lock"])
def test_local_failure_cleanup_does_not_mask_primary_or_remove_foreign_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup: str
) -> None:
    store = DagStore(tmp_path)
    identity = _identity()
    primary = KeyboardInterrupt("primary operation cancellation")
    original_unlink = Path.unlink
    original_rmdir = Path.rmdir
    armed = False

    def unlink(path: Path, *args, **kwargs):
        if armed and cleanup == "unlink" and path == store._execution_claim_path(identity):
            raise SystemExit("secondary unlink cancellation")
        return original_unlink(path, *args, **kwargs)

    def rmdir(path: Path):
        if armed and cleanup == "lock" and path.name == ".write-lock":
            original_rmdir(path)
            raise OSError("secondary lock cleanup failure")
        return original_rmdir(path)

    def operation() -> object:
        nonlocal armed
        armed = True
        if cleanup == "other-owner":
            with store._write_lock(identity):
                store._write_json(
                    store._execution_claim_path(identity),
                    {"owner_id": "other", "expires_at": time.time() + 86_400},
                )
        raise primary

    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(Path, "rmdir", rmdir)
    with pytest.raises(KeyboardInterrupt) as error:
        store.execute(identity, operation)
    assert error.value is primary
    assert store.cached_output(identity) is None
    assert not (store._directory(identity) / ".write-lock").exists()
    assert not list(store._directory(identity).glob(".*.tmp"))
    claim = store._read_execution_claim(identity)
    if cleanup == "other-owner":
        assert claim is not None and claim["owner_id"] == "other"
    elif cleanup == "unlink":
        assert claim is not None
    else:
        assert claim is None


@pytest.mark.parametrize("stage", ["initial", "terminal"])
def test_local_lock_or_release_cancellation_propagates_without_stranding_owned_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    identity = _identity()
    store = DagStore(tmp_path)
    cancellation = KeyboardInterrupt("new cleanup cancellation")
    original_rmdir = Path.rmdir
    original_unlink = Path.unlink
    operations: list[int] = []
    raised = False

    def rmdir(path: Path):
        nonlocal raised
        original_rmdir(path)
        if stage == "initial" and path.name == ".write-lock" and not raised:
            raised = True
            raise cancellation

    def unlink(path: Path, *args, **kwargs):
        nonlocal raised
        if stage == "terminal" and path == store._execution_claim_path(identity) and not raised:
            raised = True
            raise cancellation
        return original_unlink(path, *args, **kwargs)

    def operation() -> object:
        operations.append(1)
        return {"valid": True}

    monkeypatch.setattr(Path, "rmdir", rmdir)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(KeyboardInterrupt) as error:
        store.execute(identity, operation)
    assert error.value is cancellation
    assert not store._execution_claim_path(identity).exists()
    assert not (store._directory(identity) / ".write-lock").exists()
    if stage == "initial":
        assert not operations
        assert store.cached_output(identity) is None
        assert store.execute(identity, operation) == ({"valid": True}, False)
        assert store.attempt_count(identity) == 2
    else:
        assert store.cached_output(identity) == {"valid": True}
        assert store.execute(identity, operation) == ({"valid": True}, True)
    assert operations == [1]


def test_local_secondary_temp_cleanup_error_preserves_output_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = _identity()
    store = DagStore(tmp_path)
    primary = OSError("primary output rename failure")
    original_replace = Path.replace
    original_unlink = Path.unlink

    def replace(path: Path, target: Path) -> Path:
        if target == store._output_path(identity):
            raise primary
        return original_replace(path, target)

    def unlink(path: Path, *args, **kwargs):
        result = original_unlink(path, *args, **kwargs)
        if path.name.startswith(".output.json."):
            raise SystemExit("secondary temporary file cleanup cancellation")
        return result

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError) as error:
        store.execute(identity, lambda: {"synthetic": True})
    assert error.value is primary
    assert store.cached_output(identity) is None
    assert not store._execution_claim_path(identity).exists()
    assert not list(store._directory(identity).glob(".*.tmp"))
