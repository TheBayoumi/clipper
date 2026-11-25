"""End-to-end public synthetic chapter rehearsal tests; no Modal account/provider."""
from __future__ import annotations

import builtins
import hashlib
import json
import os
import socket
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

import book_public_rehearsal as rehearsal


SHA = "a" * 40


def test_two_distinct_public_tracks_and_adversarial_contracts(tmp_path: Path) -> None:
    root = tmp_path / "fresh-rehearsal"
    result = rehearsal.rehearse(root, SHA)
    assert result["schema"] == "clipper-public-book-model-free-rehearsal-v1"
    assert result["source_sha"] == SHA
    assert result["source_pin"] == "EXPLICIT_LOCAL_GIT_CHECK_REQUIRED_FOR_CLI"
    assert result["fixture_label"] == rehearsal.FIXTURE
    assert result["status"] == "SYNTHETIC_CONTRACT_REHEARSAL_PASSED"
    assert result["publish_eligible"] is False
    assert result["tracks"]["pause"]["status"] == "AWAITING_HUMAN_CHAPTER_BEAT_REVIEW"
    assert result["tracks"]["pause"]["review_supplied"] is False
    assert result["tracks"]["pause"]["units"] > 0
    assert result["tracks"]["pause"]["gaps"] > 0
    assert result["tracks"]["pause"]["provenance"] == "INJECTED_TEST"
    assert result["tracks"]["synthetic_test_review"]["review_label"] == rehearsal.REVIEW
    assert (
        result["tracks"]["synthetic_test_review"]["status"]
        == "TEST_DOUBLE_HYBRID_CHAPTER_TEXT_READY_UNREVIEWED"
    )
    assert result["tracks"]["synthetic_test_review"]["audio_provenance"] == "INJECTED_TEST_WAV"
    assert result["tracks"]["synthetic_test_review"]["publish_eligible"] is False
    assert result["injected_callback_calls"]["argument"] >= 2
    assert result["injected_callback_calls"]["audit"] >= 2
    assert result["injected_callback_calls"]["reconcile"] >= 2
    assert result["injected_callback_calls"]["adaptation"] == 1
    assert result["injected_callback_calls"]["chapter_script"] == 1
    assert result["injected_callback_calls"]["chapter_direction"] == 1
    assert result["injected_pcm_calls"] > 0
    assert result["malformed_calls"] == 1
    assert result["interrupted_calls"] == 1
    assert result["real_model_calls"] == 0
    assert result["real_provider_calls"] == 0
    assert result["network_calls"] == 0
    assert result["publication_calls"] == 0
    assert result["prohibited_attempts"] == []
    assert set(result["adversarial"]) == {
        "stale_review",
        "source_tamper",
        "producer_identity_tamper",
        "raw_artifact_tamper",
        "malformed_reply_no_replay",
        "interrupted_claim_no_replay",
    }
    assert (root / "pause").is_dir()
    assert (root / "continuation").is_dir()
    assert (root / "malformed").is_dir()
    assert (root / "interruption").is_dir()
    assert result["elapsed_seconds"] >= 0
    assert result["peak_rss_kib_platform"] > 0
    hashes = result["artifact_sha256"]
    assert hashes
    for relative, digest in hashes.items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("bad_sha", ["", "a" * 39, "g" * 40, "A" * 40])
def test_rehearsal_invalid_sha_rejects_before_writing(tmp_path: Path, bad_sha: str) -> None:
    path = tmp_path / "fresh"
    with pytest.raises(rehearsal.RehearsalStop, match="exact source SHA"):
        rehearsal.rehearse(path, bad_sha)
    assert not path.exists()


def test_existing_root_rejects_without_clobber(tmp_path: Path) -> None:
    root = tmp_path / "existing"
    root.mkdir()
    marker = root / "private-do-not-read.txt"
    marker.write_text("do not read")
    with pytest.raises(rehearsal.RehearsalStop, match="fresh absolute root"):
        rehearsal.rehearse(root, SHA)
    assert marker.read_text() == "do not read"


def test_guards_prohibit_network_subprocess_model_import_and_restore() -> None:
    events: list[str] = []
    real_import = builtins.__import__
    real_connect = socket.socket.connect
    real_popen = subprocess.Popen
    with rehearsal.offline_guard(events):
        for callback in (
            lambda: builtins.__import__("torch"),
            lambda: builtins.__import__("modal"),
            lambda: socket.create_connection(("127.0.0.1", 9)),
            lambda: subprocess.Popen(["echo", "forbidden"]),
            lambda: os.system("echo forbidden"),
        ):
            with pytest.raises(rehearsal.RehearsalStop, match="prohibited operation"):
                callback()
    assert len(events) == 5
    assert builtins.__import__ is real_import
    assert socket.socket.connect is real_connect
    assert subprocess.Popen is real_popen


def test_cli_requires_actual_checkout_pin_before_rehearsal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_git(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(stdout="b" * 40 + "\n")

    monkeypatch.setattr(rehearsal.subprocess, "run", fake_git)
    fresh = tmp_path / "never-created"
    code = rehearsal.main(["--fresh-root", str(fresh), "--expected-sha", SHA])
    assert code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "PUBLIC_SYNTHETIC_REHEARSAL_BLOCKED"
    assert not fresh.exists()


def test_cli_explicit_git_match_reports_verified_pin(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        rehearsal.subprocess, "run", lambda *_a, **_kw: SimpleNamespace(stdout=SHA + "\n")
    )
    monkeypatch.setattr(
        rehearsal,
        "rehearse",
        lambda _root, _sha: {"status": "SYNTHETIC_CONTRACT_REHEARSAL_PASSED"},
    )
    assert rehearsal.main(["--fresh-root", str(tmp_path / "new"), "--expected-sha", SHA]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source_pin"] == "VERIFIED_LOCAL_GIT_HEAD"
