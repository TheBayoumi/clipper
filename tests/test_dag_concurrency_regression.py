from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from clipper.dag import DagStore, StageResult
from clipper.stage_contracts import StageIdentity


def _identity() -> StageIdentity:
    return StageIdentity(
        stage_name="concurrent-stage",
        source_hash="source-hash",
        contract_hash="contract-hash",
        model_revision="model-revision",
    )


def test_concurrent_failure_cannot_overwrite_completed_pass(tmp_path) -> None:
    store = DagStore(tmp_path)
    identity = _identity()

    def succeed():
        time.sleep(0.03)
        return StageResult({"status": "PASS", "value": 1})

    def fail():
        time.sleep(0.08)
        raise RuntimeError("losing concurrent writer")

    with ThreadPoolExecutor(max_workers=2) as pool:
        success_future = pool.submit(store.execute, identity, succeed)
        failure_future = pool.submit(store.execute, identity, fail)
        left = success_future.result(timeout=5)
        right = failure_future.result(timeout=5)

    assert left[0] == {"status": "PASS", "value": 1}
    assert right[0] == {"status": "PASS", "value": 1}
    assert store.cached_output(identity) == {"status": "PASS", "value": 1}

    directory = store._directory(identity)
    assert not (directory / ".write-lock").exists()
    assert not list(directory.glob("*.tmp"))
    assert not list(directory.glob(".*.tmp"))
