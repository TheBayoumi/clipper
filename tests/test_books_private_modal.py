"""Offline launcher contract tests; never authenticate or allocate a resource."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper.books import private_modal, private_phase
from clipper.stage_contracts import content_fingerprint

from test_books_private_phase import _brief


def test_preflight_does_not_authenticate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, packet = _brief()
    forbidden = Mock(side_effect=AssertionError("no provider"))
    monkeypatch.setattr(private_modal, "identify", forbidden)
    result = private_modal.launch(
        manifest,
        packet,
        None,
        approved_manifest_sha256=content_fingerprint(manifest),
        execute=False,
        checkout=tmp_path,
        journal_root=tmp_path,
        output=tmp_path / "result.bin",
    )
    assert result["status"] == "PREFLIGHT_ONLY"
    forbidden.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_duplicate_phase_claim(tmp_path: Path) -> None:
    manifest, _ = _brief()
    journal = private_modal.Journal(tmp_path, manifest)
    assert (journal.directory / "0001.json").is_file()
    manifest["owner"] = "c" * 64
    with pytest.raises(FileExistsError):
        private_modal.Journal(tmp_path, manifest)


@pytest.mark.parametrize("tasks", ["0", "1", 0])
def test_exact_terminal_observation(monkeypatch: pytest.MonkeyPatch, tasks: object) -> None:
    import json

    app_id = "ap-" + "a" * 22
    run = Mock(return_value=Mock(stdout=json.dumps([
        {"app_id": app_id, "state": "stopped", "tasks": tasks}
    ])))
    monkeypatch.setattr(private_modal.subprocess, "run", run)
    if isinstance(tasks, int):
        with pytest.raises(private_phase.PrivatePhaseError):
            private_modal.observe(app_id)
    else:
        assert private_modal.observe(app_id) is (tasks == "0")
