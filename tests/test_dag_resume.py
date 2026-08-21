import json
from pathlib import Path

import pytest

from clipper.dag import DagStore, StageResult
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
