"""Offline configured replay CLI policy, no model or remote execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from clipper.books import chapter, configured_replay, pipeline_plan, scripting
from clipper.books.cli import configure_books_parser, dispatch_books


def setup_config(tmp_path: Path, target: str = "brief") -> tuple[Path, dict]:
    config = pipeline_plan.template("a" * 64, target)
    for name in config["stages"]:
        if name not in {"source", "human_review"}:
            config["stages"][name]["enabled"] = False
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    return path, config


@pytest.mark.parametrize("target", ["brief", "script"])
def test_replay_entrypoint_consumes_actual_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    path, config = setup_config(tmp_path, target)
    expected = {
        "status": "BRIEF_READY_UNREVIEWED",
        "configured_execution_receipt_identity": "f" * 64,
        "configured_stage_results": {"argument": "REUSED"},
    }
    evaluate = Mock(return_value=expected)
    monkeypatch.setattr(chapter, "evaluate_hybrid_chapter", evaluate)
    report = configured_replay.replay_configured(
        config_path=path,
        selected_source=tmp_path / "source.json",
        section_manifest=tmp_path / "sections.json",
        artifact_root=tmp_path,
    )
    kwargs = evaluate.call_args.kwargs
    assert kwargs["pipeline_config"] == config
    assert kwargs["plan"] == chapter.ChapterPlan()
    assert kwargs["argument_completion"] is scripting._ORIGINAL_LOCAL_CHAPTER_COMPLETION
    assert kwargs["script_completion"] is scripting._ORIGINAL_LOCAL_CHAPTER_COMPLETION
    assert report["inference_calls"] == 0
    assert report["mode"] == "NATIVE_CACHE_REPLAY_ONLY"


@pytest.mark.parametrize("fault", ["enabled", "unsupported", "source_off"])
def test_replay_cannot_start_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    path, config = setup_config(tmp_path)
    if fault == "enabled":
        config["stages"]["argument"]["enabled"] = True
    elif fault == "unsupported":
        config["targets"] = ["video"]
    else:
        config["stages"]["source"]["enabled"] = False
    path.write_text(json.dumps(config))
    forbidden = Mock(side_effect=AssertionError("inference forbidden"))
    monkeypatch.setattr(scripting, "_ORIGINAL_LOCAL_CHAPTER_COMPLETION", forbidden)
    monkeypatch.setattr(scripting, "_local_chapter_completion", forbidden)
    with pytest.raises(chapter.ChapterStop):
        configured_replay.replay_configured(
            config_path=path,
            selected_source=tmp_path / "no-source.json",
            section_manifest=tmp_path / "no-sections.json",
            artifact_root=tmp_path,
        )
    forbidden.assert_not_called()


def test_replay_cli_sanitizes_failures(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    parser = argparse.ArgumentParser()
    configure_books_parser(parser.add_subparsers())
    args = parser.parse_args(
        [
            "books",
            "pipeline-replay",
            "--config",
            str(tmp_path / "missing.json"),
            "--selected-source",
            str(tmp_path / "source.json"),
            "--section-manifest",
            str(tmp_path / "section.json"),
            "--artifact-root",
            str(tmp_path),
        ]
    )
    assert dispatch_books(args) == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "CONFIGURED_REPLAY_FAILED_CLOSED",
        "inference_calls": 0,
    }


def test_replay_cli_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        configured_replay,
        "replay_configured",
        lambda **kw: {
            "mode": "NATIVE_CACHE_REPLAY_ONLY",
            "inference_calls": 0,
        },
    )
    parser = argparse.ArgumentParser()
    configure_books_parser(parser.add_subparsers())
    args = parser.parse_args(
        [
            "books",
            "pipeline-replay",
            "--config",
            str(tmp_path / "config.json"),
            "--selected-source",
            str(tmp_path / "source.json"),
            "--section-manifest",
            str(tmp_path / "section.json"),
            "--artifact-root",
            str(tmp_path),
        ]
    )
    assert dispatch_books(args) == 0
    assert json.loads(capsys.readouterr().out)["inference_calls"] == 0
