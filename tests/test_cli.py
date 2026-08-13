import json
from pathlib import Path
from unittest.mock import patch

from clipper.cli import main
from clipper.models import VideoCandidate


def make_brief(tmp_path: Path) -> Path:
    path = tmp_path / "brief.json"
    path.write_text(
        json.dumps(
            {
                "campaign_id": "c",
                "title": "AI",
                "objective": "Goal",
                "keywords": ["automation"],
                "allowed_video_ids": ["v1"],
                "rights_confirmed": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_cli_validate(tmp_path: Path, capsys) -> None:
    path = make_brief(tmp_path)
    assert main(["--verbose", "validate", "--brief", str(path)]) == 0
    assert '"campaign_id": "c"' in capsys.readouterr().out


def test_cli_discover(tmp_path: Path, capsys) -> None:
    path = make_brief(tmp_path)
    video = VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")
    with patch("clipper.cli.YouTubeClient") as client_cls:
        client_cls.return_value.discover.return_value = [video]
        assert main(["discover", "--brief", str(path)]) == 0
    assert '"video_id": "v1"' in capsys.readouterr().out


def test_cli_discover_filters_video_outside_allow_list(tmp_path: Path, capsys) -> None:
    path = make_brief(tmp_path)
    video = VideoCandidate("v2", "Other", "UC2", "Other Channel", "https://youtu.be/v2")
    with patch("clipper.cli.YouTubeClient") as client_cls:
        client_cls.return_value.discover.return_value = [video]
        assert main(["discover", "--brief", str(path)]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_cli_run_and_error(tmp_path: Path, capsys, monkeypatch) -> None:
    path = make_brief(tmp_path)
    monkeypatch.setenv("CLIPPER_WHISPER_MODEL", "base.en")
    with patch("clipper.cli.run_pipeline", return_value=tmp_path / "run") as run:
        assert (
            main(
                [
                    "run",
                    "--brief",
                    str(path),
                    "--artifact-root",
                    str(tmp_path / "artifacts"),
                    "--no-render",
                ]
            )
            == 0
        )
        assert run.call_args.kwargs["render"] is False
        assert run.call_args.kwargs["settings"].whisper_model == "base.en"
    assert str(tmp_path / "run") in capsys.readouterr().out

    with patch("clipper.cli.load_brief", side_effect=RuntimeError("boom")):
        assert main(["validate", "--brief", str(path)]) == 1


def test_cli_run_returns_failure_for_failed_production_manifest(tmp_path: Path) -> None:
    path = make_brief(tmp_path)
    run_dir = tmp_path / "failed-run"
    run_dir.mkdir()
    (run_dir / "manifest.json").write_text('{"status":"FAILED"}')
    with patch("clipper.cli.run_pipeline", return_value=run_dir):
        assert main(["run", "--brief", str(path), "--artifact-root", str(tmp_path / "out")]) == 1


def test_cli_benchmark_writes_report_and_returns_threshold_status(tmp_path: Path, capsys) -> None:
    domains = [
        "gaming",
        "business",
        "comedy_conversational",
        "science_education",
        "interview_personal",
    ]
    episodes = []
    for index, domain in enumerate(domains):
        stories = tmp_path / f"s{index}.json"
        concepts = tmp_path / f"c{index}.json"
        stories.write_text(json.dumps([{"start": 10, "end": 20}]))
        concepts.write_text(json.dumps([{"source_start": 10, "source_end": 20}]))
        episodes.append(
            {
                "episode_id": str(index),
                "domain": domain,
                "references": [
                    {
                        "reference_id": str(index),
                        "start": 10,
                        "end": 20,
                        "semantic_group": str(index),
                    }
                ],
                "predictions": {"story_moments": stories.name, "concepts": concepts.name},
            }
        )
    manifest = tmp_path / "corpus.json"
    manifest.write_text(
        json.dumps({"schema_version": "clipper-benchmark-corpus-v1", "episodes": episodes})
    )
    output = tmp_path / "report.json"
    assert main(["benchmark", "--manifest", str(manifest), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["status"] == "PASS"
    assert '"status": "PASS"' in capsys.readouterr().out
