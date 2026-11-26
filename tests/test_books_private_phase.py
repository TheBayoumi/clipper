"""Private packet and configured phase preflight; no provider execution."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper.books import chapter, pipeline_plan
from clipper.books import private_phase as phase
from clipper.stage_contracts import content_fingerprint


def _brief():
    source = b"selected fixture"
    section = b"section fixture"
    config = pipeline_plan.template(phase.sha(source), "brief")
    manifest = {
        "schema": phase.MANIFEST_SCHEMA,
        "commit": "a" * 40,
        "workspace": phase.WORKSPACE,
        "phase": "brief",
        "configuration_sha256": content_fingerprint(config),
        "bundle_sha256": phase.sha(source),
        "section_sha256": phase.sha(section),
        "review_sha256": None,
        "expected_brief_identity": None,
        "checkpoint_sha256": None,
        "owner": "b" * 64,
        "cpu": 4,
        "memory_mib": 16384,
        "timeout_seconds": 900,
        "resource_approval": "EXPLICIT_PRIVATE_CPU_PHASE",
    }
    packet = phase.pack(
        {
            "selected.json": source,
            "section.json": section,
            "config.json": json.dumps(config).encode(),
        }
    )
    return manifest, packet


def test_packet_roundtrip_and_durable_checkpoint(tmp_path: Path) -> None:
    name = "book-candidates/hybrid_argument/" + "a" * 64 + "/raw-1.json"
    files = {name: b"private bytes"}
    packet = phase.pack(files)
    assert phase.unpack(packet) == files
    destination = tmp_path / "checkpoint.bin"
    phase.persist_checkpoint(destination, packet, phase.sha(packet))
    assert destination.read_bytes() == packet
    restored = tmp_path / "restored"
    restored.mkdir()
    phase.restore(restored, files)
    assert phase.collect(restored) == packet
    with pytest.raises(phase.PrivatePhaseError):
        phase.restore(restored, files)


@pytest.mark.parametrize("name", ["../escape", "/absolute", "book-candidates/other.json"])
def test_packet_rejects_paths(name: str) -> None:
    with pytest.raises(phase.PrivatePhaseError):
        phase.pack({name: b"x"})


def test_packet_corruption_and_trailing_bytes() -> None:
    packet = phase.pack({"selected.json": b"source"})
    for bad in (packet[:-1], packet + b"x", packet[:-1] + b"z"):
        with pytest.raises(phase.PrivatePhaseError):
            phase.unpack(bad)


@pytest.mark.parametrize(
    "field,value",
    [
        ("workspace", "wrong"),
        ("cpu", 8),
        ("memory_mib", 32768),
        ("timeout_seconds", 0),
        ("resource_approval", None),
        ("commit", "main"),
        ("review_sha256", "a" * 64),
        ("configuration_sha256", "c" * 64),
    ],
)
def test_manifest_preflight_rejects(field, value) -> None:
    manifest, packet = _brief()
    manifest[field] = value
    with pytest.raises(phase.PrivatePhaseError):
        phase.preflight(manifest, packet)


def test_worker_identity_precedes_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, packet = _brief()
    forbidden = Mock(side_effect=AssertionError("no provider"))
    monkeypatch.setattr(chapter, "evaluate_hybrid_chapter", forbidden)
    with pytest.raises(phase.PrivatePhaseError):
        phase.execute_private_phase(
            manifest,
            packet,
            None,
            verified_commit="0" * 40,
            verified_workspace=phase.WORKSPACE,
        )
    forbidden.assert_not_called()


def test_worker_passes_same_config_and_rejects_synthetic_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest, packet = _brief()
    evaluator = Mock(
        return_value={
            "status": "BRIEF_READY_UNREVIEWED",
            "automation_claim": "INJECTED_TEST",
        }
    )
    monkeypatch.setattr(chapter, "evaluate_hybrid_chapter", evaluator)
    report, output = phase.execute_private_phase(
        manifest,
        packet,
        None,
        verified_commit=manifest["commit"],
        verified_workspace=phase.WORKSPACE,
    )
    assert report["status"] == "PRIVATE_PHASE_FAILED_NO_RETRY"
    assert report["checkpoint_sha256"] == phase.sha(output)
    kwargs = evaluator.call_args.kwargs
    assert kwargs["pipeline_config"]["targets"] == ["brief"]
    assert kwargs["plan"].brief_cpu is True
    assert kwargs["plan"].kokoro_cpu is False
    assert kwargs["argument_completion"] is phase.scripting._ORIGINAL_LOCAL_CHAPTER_COMPLETION


@pytest.mark.parametrize("evaluation_fails", [False, True])
def test_finalization_failure_retains_partial(
    monkeypatch: pytest.MonkeyPatch, evaluation_fails: bool
) -> None:
    manifest, packet = _brief()
    if evaluation_fails:
        evaluator = Mock(side_effect=RuntimeError("private data"))
    else:
        evaluator = Mock(
            return_value={"status": "BRIEF_READY_UNREVIEWED", "automation_claim": "ACTUAL_CPU"}
        )
    monkeypatch.setattr(chapter, "evaluate_hybrid_chapter", evaluator)
    monkeypatch.setattr(
        phase.selected_scope, "import_chapter", Mock(return_value=Mock(chapter_number=1))
    )
    monkeypatch.setattr(phase.chapter_host, "_hybrid_review_packet", Mock())
    monkeypatch.setattr(phase, "collect", Mock(side_effect=OSError("private path")))
    report, raw = phase.execute_private_phase(
        manifest,
        packet,
        None,
        verified_commit=manifest["commit"],
        verified_workspace=phase.WORKSPACE,
    )
    assert report["status"] == "PRIVATE_PHASE_FAILED_NO_RETRY"
    assert report["retention"] == "PARTIAL_NONRESUMABLE"
    assert raw is not None
    assert report["checkpoint_sha256"] == phase.sha(raw)


def test_partial_excludes_unsafe_and_oversized(tmp_path: Path) -> None:
    native = tmp_path / "book-candidates" / "hybrid_argument" / ("a" * 64)
    native.mkdir(parents=True)
    (native / "raw-1.json").write_bytes(b"safe")
    (native / "raw-2.json").write_bytes(b"x" * (phase.MAX_MEMBER + 1))
    (native / "unknown.json").write_bytes(b"not allowed")
    (native / "started-1.json").symlink_to(native / "raw-1.json")
    packet, omitted, truncated = phase.partial_checkpoint(tmp_path)
    assert packet is not None
    assert omitted == 3
    assert not truncated
    assert list(phase.unpack(packet).values()) == [b"safe"]


def test_empty_partial_is_not_archive(tmp_path: Path) -> None:
    manifest, _ = _brief()
    report, raw = phase.failed_phase(tmp_path, manifest, {})
    assert raw is None
    assert report["retention"] == "OUTPUT_NOT_RETAINED"
    assert report["checkpoint_sha256"] is None
