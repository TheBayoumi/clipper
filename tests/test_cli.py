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


def test_mw4_cli_routes_to_installed_runtime() -> None:
    with patch("clipper.cli.run_mw4", return_value=0) as run_mw4:
        assert main(["mw4", "self-test"]) == 0

    args = run_mw4.call_args.args[0]
    assert args.command == "mw4"
    assert args.mw4_command == "self-test"


def test_mw4_cli_reports_runtime_failure() -> None:
    with patch("clipper.cli.run_mw4", side_effect=RuntimeError("mw4 runtime unavailable")):
        assert main(["mw4", "self-test"]) == 1
