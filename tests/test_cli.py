import json
from pathlib import Path
from unittest.mock import patch

from clipper.cli import _print_mw4_diagnostics, _repo_root, _run_script, main
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


def test_mw4_self_test_uses_canonical_modules(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    with (
        patch("clipper.cli._repo_root", return_value=tmp_path),
        patch("clipper.cli.subprocess.run") as run,
        patch("clipper.cli._run_script") as run_script,
    ):
        assert main(["mw4", "self-test"]) == 0
    compile_command = run.call_args.args[0]
    assert compile_command[1:3] == ["-m", "py_compile"]
    assert any(
        str(item).endswith("mw4_semantic_gameplay_v3_1_payoff_terminal.py")
        for item in compile_command
    )
    assert run_script.call_count == 2

    forbidden = scripts / "mw4_semantic_gameplay_v3_1_refined.py"
    forbidden.write_text("", encoding="utf-8")
    with patch("clipper.cli._repo_root", return_value=tmp_path):
        assert main(["mw4", "self-test"]) == 1


def test_mw4_discover_routes_through_cli_contract(tmp_path: Path) -> None:
    output = tmp_path / "catalog.json"
    github_output = tmp_path / "github-output"
    with (
        patch("clipper.cli._repo_root", return_value=tmp_path),
        patch("clipper.cli._run_script") as run_script,
    ):
        assert (
            main(
                [
                    "mw4",
                    "discover",
                    "--review-url",
                    "https://example.test/review",
                    "--output",
                    str(output),
                    "--github-output",
                    str(github_output),
                    "--expected-count",
                    "4",
                ]
            )
            == 0
        )
    arguments = run_script.call_args.args
    assert arguments[1] == "mw4_v3_1_source_catalog.py"
    assert "--expected-count" in arguments
    assert "4" in arguments


def test_mw4_analyze_keeps_signed_catalog_local(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    source_dir = tmp_path / "raw_sources"
    output_dir = tmp_path / "analysis" / "r1"
    with (
        patch("clipper.cli._repo_root", return_value=tmp_path),
        patch("clipper.cli._run_script") as run_script,
        patch("clipper.cli._print_mw4_diagnostics") as diagnostics,
    ):
        assert (
            main(
                [
                    "mw4",
                    "analyze",
                    "--review-url",
                    "https://example.test/review",
                    "--expected-count",
                    "4",
                    "--source-key",
                    "r1",
                    "--config",
                    str(config),
                    "--source-dir",
                    str(source_dir),
                    "--output-dir",
                    str(output_dir),
                ]
            )
            == 0
        )
    called_scripts = [call.args[1] for call in run_script.call_args_list]
    assert called_scripts == [
        "mw4_v3_1_source_catalog.py",
        "mw4_v3_1_source_catalog.py",
        "mw4_v3_1_mediasilo_source.py",
        "mw4_v3_1_source_qa.py",
        "mw4_fullframe_retention_v3_1.py",
    ]
    first_call = run_script.call_args_list[0].args
    assert first_call[2] == "discover"
    assert "--review-url" in first_call
    assert "--expected-count" in first_call
    diagnostics.assert_called_once_with(output_dir / "r1_analysis_v3_1.json")


def test_mw4_allocate_render_and_batch_route_through_cli(tmp_path: Path) -> None:
    common = ["--config", str(tmp_path / "config.json")]
    with (
        patch("clipper.cli._repo_root", return_value=tmp_path),
        patch("clipper.cli._run_script") as run_script,
    ):
        assert (
            main(
                [
                    "mw4",
                    "allocate",
                    *common,
                    "--analysis-root",
                    str(tmp_path / "analysis"),
                    "--allocation-out",
                    str(tmp_path / "allocation.json"),
                    "--rejection-out",
                    str(tmp_path / "rejection.json"),
                    "--github-output",
                    str(tmp_path / "github-output"),
                ]
            )
            == 0
        )
        assert [call.args[1] for call in run_script.call_args_list] == [
            "mw4_v3_1_dynamic_contract.py",
            "mw4_v3_1_dynamic_workflow_support.py",
        ]

    run_script.reset_mock()
    with (
        patch("clipper.cli._repo_root", return_value=tmp_path),
        patch("clipper.cli._run_script", run_script),
    ):
        assert (
            main(
                [
                    "mw4",
                    "render-one",
                    "--source-key",
                    "r1",
                    "--source-dir",
                    str(tmp_path / "raw_sources"),
                    *common,
                    "--allocation",
                    str(tmp_path / "allocation.json"),
                    "--plan-key",
                    "plan",
                    "--ordinal",
                    "1",
                    "--output-dir",
                    str(tmp_path / "one"),
                    "--mode",
                    "production",
                ]
            )
            == 0
        )
        assert [call.args[1] for call in run_script.call_args_list] == [
            "mw4_v3_1_source_qa.py",
            "mw4_v3_1_render_one.py",
        ]

    run_script.reset_mock()
    with (
        patch("clipper.cli._repo_root", return_value=tmp_path),
        patch("clipper.cli._run_script", run_script),
    ):
        assert (
            main(
                [
                    "mw4",
                    "batch-contract",
                    *common,
                    "--allocation",
                    str(tmp_path / "allocation.json"),
                    "--clip-meta",
                    str(tmp_path / "clip-meta"),
                    "--output-dir",
                    str(tmp_path / "batch"),
                ]
            )
            == 0
        )
        assert [call.args[1] for call in run_script.call_args_list] == [
            "mw4_v3_1_dynamic_workflow_support.py",
            "mw4_v3_1_dynamic_contract.py",
        ]


def test_mw4_diagnostics_and_script_runner(tmp_path: Path, capsys) -> None:
    manifest = tmp_path / "r1_analysis_v3_1.json"
    manifest.write_text(
        json.dumps(
            {
                "source_key": "r1",
                "diagnostics": {
                    "local_interaction_verifier": {
                        "events": [
                            {"confirmed": True, "kinds": ["outcome_like"], "time": 10.0},
                            {"confirmed": False, "kinds": ["outcome_like"], "time": 20.0},
                        ]
                    }
                },
                "candidate_pool": [
                    {
                        "effect_events": [{"kind": "outcome_like", "time": 10.0}],
                        "engagements": [],
                    },
                    {
                        "effect_events": [],
                        "engagements": [
                            {"events": [{"kinds": ["impact"], "time": 30.0}]}
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    _print_mw4_diagnostics(manifest)
    output = capsys.readouterr().out
    assert '"kill_like_outcome_events": 1' in output
    assert '"distinct_terminal_payoff_scenes_in_pool": 2' in output

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "tool.py").write_text("", encoding="utf-8")
    with patch("clipper.cli.subprocess.run") as run:
        _run_script(tmp_path, "tool.py", "--flag")
    assert run.call_args.args[0][-2:] == [str(scripts / "tool.py"), "--flag"]

    assert _repo_root().name == "clipper"
