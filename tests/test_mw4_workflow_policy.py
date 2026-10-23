"""Production and neural qualification stay validation-only at immutable source SHA."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_complete_production_is_deliberately_dispatched_and_not_cancelled() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/mw4-qualification.yml").read_text())
    trigger = workflow.get("on", workflow.get(True))
    assert "workflow_dispatch" in trigger
    assert "push" not in trigger  # CI still runs on push; expensive acceptance is frozen.
    concurrency = workflow["concurrency"]
    assert "github.sha" in concurrency["group"]
    assert concurrency["cancel-in-progress"] is False
    jobs = workflow["jobs"]
    assert jobs["analyze"]["strategy"]["max-parallel"] == 2
    assert jobs["render-one"]["strategy"]["max-parallel"] == 2
    assert jobs["allocate"]["needs"] == "analyze"
    assert jobs["render-one"]["needs"] == "allocate"
    assert "independent-source-review" in jobs
    assert any(
        "review-contract" in step.get("run", "")
        for step in jobs["independent-source-review"]["steps"]
    )
    assert jobs["batch-contract"]["needs"] == ["allocate", "render-one"]


def test_gpu_analysis_uses_profile_and_exact_sha_cached_batches() -> None:
    workflow = yaml.safe_load((ROOT / ".github/workflows/mw4-qualification.yml").read_text())
    analyze = workflow["jobs"]["analyze"]
    assert analyze["env"]["CLIPPER_NEURAL_MODAL_APP"] == "clipper-mw4-vision-run"
    assert "github.sha" in analyze["env"]["CLIPPER_DEPLOYED_GIT_SHA"]
    assert "MODAL_TOKEN_ID" in analyze["env"]
    steps = analyze["steps"]
    assert any("settings.enabled" in step.get("run", "") for step in steps)
    assert any("clipper mw4 analyze" in step.get("run", "") for step in steps)
    cache = [step for step in steps if "actions/cache/" in str(step.get("uses", ""))]
    assert {step["uses"] for step in cache} == {"actions/cache/restore@v4", "actions/cache/save@v4"}
    for step in cache:
        assert "github.sha" in step["with"]["key"]
        assert "matrix.source" in step["with"]["key"]
    assert any(
        step.get("if") == "always()" and "neural diagnostic" in step.get("name", "").lower()
        for step in steps
    )


def test_source_wide_gpu_qualification_matches_production_sampling() -> None:
    profile = __import__("json").loads((ROOT / "src/clipper_engine/profiles/mw4.json").read_text())
    settings = profile["capabilities"]["neural_visual_events"]
    assert settings["enabled"] is True and settings["backend"] == "modal"
    assert settings["batch_frames"] == 2
    assert settings["max_sample_frames"] == 144
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/modal-neural-qualification.yml").read_text()
    )
    job = workflow["jobs"]["neural-source-wide-review"]
    assert job["env"]["CLIPPER_NEURAL_MODAL_APP"] == settings["app_name"]
    gpu_steps = [step.get("run", "") for step in job["steps"]]
    command = next(text for text in gpu_steps if "::source_scout" in text)
    assert "--batch-frames 2" in command
    assert "--max-sample-frames 144" in command
    assert "--max-refinement-anchors 20" in command


def test_gpu_source_wide_can_validate_one_source_before_six_source_expense() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/modal-neural-qualification.yml").read_text()
    )
    inputs = workflow.get("on", workflow.get(True))["workflow_dispatch"]["inputs"]
    assert inputs["source_wide"]["default"] is False
    selection = inputs["source_key"]
    assert selection["default"] == "all"
    assert set(selection["options"]) == {
        "all",
        "ghost_fight",
        "convenience_store",
        "traffic_chase",
        "nightclub",
        "makarov_chase",
        "refusing_orders",
    }
    source_job = workflow["jobs"]["neural-source-wide-review"]
    assert source_job["needs"] == "neural-source-gpu-smoke"
    matrix = source_job["strategy"]["matrix"]["source"]
    assert "inputs.source_key" in matrix
    assert "fromJSON" in matrix
    assert "convenience_store" in matrix
    artifact_step = next(
        s
        for s in source_job["steps"]
        if s.get("name") == "Retain source-wide model output and independent review requirements"
    )
    assert artifact_step["if"] == "always()"
    assert "neural-source-${{ matrix.source }}/" in artifact_step["with"]["path"]


def test_gpu_qualification_uses_signed_official_ubuntu_archive_not_slow_azure_mirror() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/modal-neural-qualification.yml").read_text()
    )
    for job_name in ("neural-source-gpu-smoke", "neural-source-wide-review"):
        steps = workflow["jobs"][job_name]["steps"]
        installer = next(
            step["run"] for step in steps if step.get("name", "").startswith("Install FFmpeg")
        )
        assert "https://archive.ubuntu.com/ubuntu" in installer
        assert "/etc/apt/sources.list.d/*.sources" in installer
        # Hosted runner apt uses a mirror+file list, not only deb822 sources.
        assert "/etc/apt/apt-mirrors.txt" in installer
        assert "for source_file in /etc/apt/apt-mirrors.txt" in installer
        assert "Acquire::Retries=2" in installer
        assert "Acquire::https::Timeout=20" in installer
        assert "DEBIAN_FRONTEND=noninteractive" in installer
        assert "command -v ffprobe" in installer


def test_gpu_qualification_resolves_unexpired_certified_source_without_stale_run_ids() -> None:
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/modal-neural-qualification.yml").read_text()
    )
    for job_name in ("neural-source-gpu-smoke", "neural-source-wide-review"):
        steps = workflow["jobs"][job_name]["steps"]
        resolver = next(s for s in steps if s.get("id") == "certified-source")
        script = resolver["run"]
        assert "actions/artifacts?name=" in script
        assert ".expired == false" in script
        assert ".workflow_run.head_branch" in script
        assert "MW4 production qualification" in script
        assert "CERTIFIED_SOURCE_ARTIFACT_UNAVAILABLE" in script
        assert "CERTIFIED_SOURCE_RUN_INVALID" in script
        assert "GH_TOKEN" in resolver["env"]
        download = next(s for s in steps if s.get("uses") == "actions/download-artifact@v4")
        assert download["with"]["run-id"] == "${{ steps.certified-source.outputs.run_id }}"
        assert "39749647361" not in str(download)
        assert "39827390899" not in str(download)
