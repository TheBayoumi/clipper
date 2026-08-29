import json
from pathlib import Path

import pytest

from clipper.dag import DagLeaseCoordinator, DagStore, StageRecord, StageResult
from clipper.stage_contracts import StageContract, content_fingerprint, stage_identity


def _identity(
    name: str,
    *,
    source_hash: str = "source",
    dependencies: tuple[str, ...] = (),
    contract_value: str = "v1",
    model_revision: str | None = None,
):
    contract = StageContract(name, {"behavior": contract_value})
    return stage_identity(
        contract,
        source_hash=source_hash,
        dependency_output_hashes=dependencies,
        model_revision=model_revision,
        decoding_parameters={"do_sample": False} if model_revision else {},
    )


def test_stage_contract_identity_changes_only_for_relevant_material() -> None:
    first = StageContract("render", {"codec": "h264"}, {"watermark": "a"})
    same = StageContract("render", {"codec": "h264"}, {"watermark": "a"})
    changed_policy = StageContract("render", {"codec": "h264"}, {"watermark": "b"})
    changed_contract = StageContract("render", {"codec": "hevc"}, {"watermark": "a"})

    assert first.contract_hash == same.contract_hash
    assert first.contract_hash != changed_policy.contract_hash
    assert first.contract_hash != changed_contract.contract_hash

    identity = stage_identity(first, source_hash="source", dependency_output_hashes=("dep",))
    same_identity = stage_identity(same, source_hash="source", dependency_output_hashes=("dep",))
    assert identity.cache_key == same_identity.cache_key
    assert (
        identity.cache_key
        != stage_identity(
            changed_policy,
            source_hash="source",
            dependency_output_hashes=("dep",),
        ).cache_key
    )
    assert (
        identity.cache_key
        != stage_identity(
            first,
            source_hash="source",
            dependency_output_hashes=("other",),
        ).cache_key
    )


def test_completed_stage_is_reused_without_invoking_operation_again(tmp_path: Path) -> None:
    store = DagStore(tmp_path)
    identity = _identity("transcription", model_revision="asr-rev")
    calls = 0

    def operation() -> StageResult:
        nonlocal calls
        calls += 1
        return StageResult({"words": ["hello"]}, {"gpu_seconds": 3.0}, 0.01)

    first, first_cached = store.execute(identity, operation)
    second, second_cached = store.execute(identity, operation)

    assert first == second == {"words": ["hello"]}
    assert first_cached is False
    assert second_cached is True
    assert calls == 1
    record = json.loads((store._record_path(identity)).read_text())
    assert record["status"] == "PASS"
    assert record["attempt_count"] == 1
    assert record["usage"] == {"gpu_seconds": 3.0}
    assert record["cost_usd"] == 0.01


def test_downstream_contract_change_does_not_rerun_unrelated_upstream_stages(
    tmp_path: Path,
) -> None:
    store = DagStore(tmp_path)
    calls = {"source": 0, "asr": 0, "align": 0, "multimodal": 0, "planning": 0}

    source_id = _identity("source")

    def source_op():
        calls["source"] += 1
        return {"source": "master"}

    source_output, _ = store.execute(source_id, source_op)
    source_fp = content_fingerprint(source_output)

    asr_id = _identity("transcription", dependencies=(source_fp,), model_revision="asr")

    def asr_op():
        calls["asr"] += 1
        return {"words": ["a", "b"]}

    asr_output, _ = store.execute(asr_id, asr_op)
    asr_fp = content_fingerprint(asr_output)

    align_id = _identity("alignment", dependencies=(asr_fp,), model_revision="align")

    def align_op():
        calls["align"] += 1
        return {"aligned": True}

    align_output, _ = store.execute(align_id, align_op)
    align_fp = content_fingerprint(align_output)

    multimodal_id = _identity("multimodal", dependencies=(align_fp,), model_revision="vlm")

    def multimodal_op():
        calls["multimodal"] += 1
        return {"events": [1]}

    multimodal_output, _ = store.execute(multimodal_id, multimodal_op)
    multimodal_fp = content_fingerprint(multimodal_output)

    planning_v1 = _identity(
        "planning",
        dependencies=(multimodal_fp,),
        contract_value="window-contract-a",
        model_revision="editor",
    )

    def planning_op():
        calls["planning"] += 1
        return {"windows": ["w1"]}

    store.execute(planning_v1, planning_op)

    # Simulate termination after planning and a later resume with only the planning
    # contract changed. Exact matching source/ASR/alignment/multimodal nodes must reuse.
    assert store.execute(source_id, source_op)[1] is True
    assert store.execute(asr_id, asr_op)[1] is True
    assert store.execute(align_id, align_op)[1] is True
    assert store.execute(multimodal_id, multimodal_op)[1] is True

    planning_v2 = _identity(
        "planning",
        dependencies=(multimodal_fp,),
        contract_value="window-contract-b",
        model_revision="editor",
    )
    _, planning_cached = store.execute(planning_v2, planning_op)
    assert planning_cached is False
    assert calls == {"source": 1, "asr": 1, "align": 1, "multimodal": 1, "planning": 2}


def test_failed_stage_retries_locally_and_preserves_attempt_count(tmp_path: Path) -> None:
    store = DagStore(tmp_path)
    identity = _identity("review", dependencies=("render-output",), model_revision="vlm")
    calls = 0

    def flaky():
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient review failure")
        return {"decision": "PASS"}

    with pytest.raises(RuntimeError, match="transient"):
        store.execute(identity, flaky)
    output, cached = store.execute(identity, flaky)
    assert output == {"decision": "PASS"}
    assert cached is False
    assert calls == 2
    record = json.loads(store._record_path(identity).read_text())
    assert record["status"] == "PASS"
    assert record["attempt_count"] == 2


def test_tampered_or_corrupt_cached_output_is_not_reused(tmp_path: Path) -> None:
    store = DagStore(tmp_path)
    identity = _identity("stage")
    calls = 0

    def operation():
        nonlocal calls
        calls += 1
        return {"value": calls}

    store.execute(identity, operation)
    store._output_path(identity).write_text('{"value": 999}')
    output, cached = store.execute(identity, operation)
    assert cached is False
    assert output == {"value": 2}
    assert calls == 2

    store._record_path(identity).write_text("not-json")
    output, cached = store.execute(identity, operation)
    assert cached is False
    assert output == {"value": 3}


def test_stage_result_and_contract_validation_are_strict() -> None:
    with pytest.raises(ValueError, match="negative"):
        StageResult({}, cost_usd=-1)
    with pytest.raises(ValueError, match="name"):
        StageContract("", {"x": 1})
    with pytest.raises(ValueError, match="cannot be empty"):
        StageContract("stage", {})


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"execution_lease_seconds": 0.0}, "execution lease"),
        ({"follower_poll_seconds": float("nan")}, "follower poll"),
    ],
)
def test_dag_store_rejects_invalid_execution_timing(
    tmp_path: Path,
    kwargs: dict[str, float],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        DagStore(tmp_path, **kwargs)


def test_execution_claim_validation_and_corruption_are_fail_closed(tmp_path: Path) -> None:
    store = DagStore(tmp_path)
    identity = _identity("claim")
    claim_path = store._execution_claim_path(identity)
    claim_path.parent.mkdir(parents=True, exist_ok=True)

    assert store._claim_expired(None, now=1.0) is True
    assert store._claim_expired({"expires_at": "bad"}, now=1.0) is True
    assert store._claim_expired({"expires_at": float("inf")}, now=1.0) is True
    valid = {"owner_id": "owner", "expires_at": 2.0}
    assert store._claim_owner(valid) == "owner"
    assert store._claim_expired(valid, now=1.0) is False

    claim_path.write_text("not-json", encoding="utf-8")
    assert store._read_execution_claim(identity) is None


def test_corrupt_cached_output_is_not_reused(tmp_path: Path) -> None:
    store = DagStore(tmp_path)
    identity = _identity("corrupt-output")
    store.execute(identity, lambda: {"value": 1})
    store._output_path(identity).write_text("not-json", encoding="utf-8")

    output, cached = store.execute(identity, lambda: {"value": 2})

    assert cached is False
    assert output == {"value": 2}


def test_owner_that_loses_execution_lease_cannot_publish_pass(tmp_path: Path) -> None:
    store = DagStore(tmp_path)
    identity = _identity("lost-lease")

    def operation() -> dict[str, int]:
        store._write_json(
            store._execution_claim_path(identity),
            {"owner_id": "replacement-owner", "claimed_at": 0.0, "expires_at": 9e9},
        )
        return {"value": 1}

    with pytest.raises(RuntimeError, match="execution lease lost"):
        store.execute(identity, operation)

    assert store.cached_output(identity) is None


@pytest.mark.parametrize(
    "record",
    [
        lambda identity: StageRecord(identity, "RUNNING", 0, "now", None, None, {}, 0.0),
        lambda identity: StageRecord(identity, "RUNNING", 1, "now", None, None, {}, -1.0),
        lambda identity: StageRecord(identity, "PASS", 1, "now", None, None, {}, 0.0),
        lambda identity: StageRecord(identity, "FAILED", 1, "now", "done", None, {}, 0.0),
    ],
)
def test_stage_record_rejects_invalid_terminal_evidence(record) -> None:
    identity = _identity("invalid-record")
    with pytest.raises(ValueError):
        record(identity)


def test_coordinated_dag_commits_terminal_state_before_releasing_owner(tmp_path: Path) -> None:
    events: list[str] = []
    active = {"owner": ""}

    def claim(_identity, owner: str, _ttl: float) -> bool:
        if active["owner"] and active["owner"] != owner:
            return False
        active["owner"] = owner
        events.append("claim")
        return True

    def renew(_identity, owner: str, _ttl: float) -> bool:
        return active["owner"] == owner

    def release(_identity, owner: str) -> bool:
        if active["owner"] != owner:
            return False
        events.append("release")
        active["owner"] = ""
        return True

    coordinator = DagLeaseCoordinator(
        claim=claim,
        renew=renew,
        release=release,
        commit=lambda: events.append("commit"),
        reload=lambda: events.append("reload"),
    )
    store = DagStore(
        tmp_path,
        execution_lease_seconds=10.0,
        follower_poll_seconds=0.001,
        coordinator=coordinator,
    )
    identity = _identity("distributed")

    output, cached = store.execute(identity, lambda: {"value": 4})

    assert output == {"value": 4}
    assert cached is False
    assert events.count("commit") == 2
    assert events.index("commit", events.index("claim")) < events.index("release")
    assert store.cached_output(identity) == {"value": 4}


def test_coordinated_dag_follower_reloads_and_reuses_terminal_cache(tmp_path: Path) -> None:
    identity = _identity("distributed-cache")
    first = DagStore(tmp_path)
    first.execute(identity, lambda: {"value": 5})
    calls = {"claim": 0, "reload": 0}

    coordinator = DagLeaseCoordinator(
        claim=lambda *_args: calls.__setitem__("claim", calls["claim"] + 1) or False,
        renew=lambda *_args: False,
        release=lambda *_args: False,
        commit=lambda: None,
        reload=lambda: calls.__setitem__("reload", calls["reload"] + 1),
    )
    follower = DagStore(tmp_path, coordinator=coordinator)

    output, cached = follower.execute(
        identity,
        lambda: (_ for _ in ()).throw(AssertionError("cached follower must not execute")),
    )

    assert output == {"value": 5}
    assert cached is True
    assert calls["reload"] >= 1
    assert calls["claim"] == 0
