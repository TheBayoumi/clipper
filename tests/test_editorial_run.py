"""Explicit run configuration for the established contextual editor."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from clipper.cli import main
from clipper.editorial_run import (
    EditorialRunConfig,
    load_editorial_run_config,
    write_editorial_run_config,
)

BRIEF = Path(__file__).resolve().parents[1] / "campaigns/reach-double-coverage-dedicated.yaml"
CHANNEL = "UCf1q6dhccWr6eQEcFFnJSbA"


@pytest.mark.parametrize("suffix", ["json", "yaml"])
def test_editorial_run_file_resolves_paths_and_preserves_source_policy(tmp_path, suffix):
    data = {
        "brief": str(BRIEF),
        "artifact_root": "drafts",
        "source_video_id": "_kDrxucOx9g",
        "target_channel_id": CHANNEL,
        "editorial_cache_root": "existing-cache",
        "render_safety_limit": 20,
        "caption_style": "B2",
    }
    config_file = tmp_path / f"run.{suffix}"
    if suffix == "json":
        config_file.write_text(json.dumps(data), encoding="utf-8")
    else:
        import yaml

        config_file.write_text(yaml.safe_dump(data), encoding="utf-8")
    loaded = load_editorial_run_config(config_file)
    assert loaded.artifact_root == tmp_path / "drafts"
    assert loaded.editorial_cache_root == tmp_path / "existing-cache"
    assert loaded.source_video_id == "_kDrxucOx9g"
    with patch("scripts.tjr_youtube_preview.render_youtube_previews") as render:
        render.return_value = tmp_path / "drafts" / "completed"
        assert main(["editorial", "--config", str(config_file)]) == 0
        render.assert_called_once_with(loaded.artifact_root, loaded.brief, run_config=loaded)


@pytest.mark.parametrize(
    "change,error",
    [
        ({"target_channel_id": "UCnot_authorized_channel_1"}, "target_channel_id"),
        ({"source_video_id": "wrong"}, "source_video_id"),
        ({"caption_style": "A"}, "B2 captions"),
        ({"render_safety_limit": 21}, "safety bound"),
        ({"unknown_editorial_rule": True}, "unknown fields"),
        ({"require_staged_original": True}, "browser_capture_file"),
    ],
)
def test_invalid_run_file_cannot_override_campaign_or_safety(tmp_path, change, error):
    data = {"brief": str(BRIEF), "artifact_root": "drafts", **change}
    config_file = tmp_path / "run.json"
    config_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        load_editorial_run_config(config_file)


def test_json_run_does_not_inherit_legacy_editorial_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "wrong")
    monkeypatch.setenv("TJR_CAPTION_STYLE", "A")
    config_file = tmp_path / "run.json"
    config_file.write_text(
        json.dumps({"brief": str(BRIEF), "artifact_root": "drafts"}), encoding="utf-8"
    )
    loaded = load_editorial_run_config(config_file)
    assert loaded.source_video_id == "" and loaded.caption_style == "B2"


def test_workflow_exports_one_validated_config_with_absolute_cache_paths(tmp_path, monkeypatch):
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "_kDrxucOx9g")
    monkeypatch.setenv("TJR_MODAL_CHANNEL_ID", CHANNEL)
    monkeypatch.setenv("TJR_CAPTION_STYLE", "B2")
    monkeypatch.setenv("TJR_EDITORIAL_CACHE_ROOT", "existing-editorial-cache")
    monkeypatch.setenv("TJR_TRANSCRIPT_CACHE_ROOT", "existing-transcript-cache")
    monkeypatch.setenv("TJR_RENDER_SAFETY_LIMIT", "20")
    config = EditorialRunConfig.from_legacy_environment(BRIEF, tmp_path / "drafts")
    output = tmp_path / "probe" / "run.json"
    write_editorial_run_config(config, output)
    loaded = load_editorial_run_config(output)
    assert loaded.source_video_id == "_kDrxucOx9g"
    assert loaded.target_channel_id == CHANNEL
    assert loaded.artifact_root == (tmp_path / "drafts").resolve()
    assert loaded.editorial_cache_root == Path("existing-editorial-cache").resolve()
    assert loaded.transcript_cache_root == Path("existing-transcript-cache").resolve()
    assert (
        main(
            [
                "editorial-config",
                "--brief",
                str(BRIEF),
                "--artifact-root",
                str(tmp_path / "drafts"),
                "--output",
                str(tmp_path / "cli-run.json"),
            ]
        )
        == 0
    )
    assert load_editorial_run_config(tmp_path / "cli-run.json") == loaded


def test_legacy_channel_alias_fails_on_conflict(tmp_path, monkeypatch):
    monkeypatch.setenv("TJR_TARGET_CHANNEL_ID", CHANNEL)
    monkeypatch.setenv("TJR_MODAL_CHANNEL_ID", "UCf1q6dhccWr6eQEcFFnJSbB")
    with pytest.raises(ValueError, match="conflicting"):
        EditorialRunConfig.from_legacy_environment(BRIEF, tmp_path)


def test_tracked_run_file_validates_without_media_acquisition(capsys):
    config_file = BRIEF.with_name("reach-double-coverage-editorial.yaml")
    with patch("scripts.tjr_youtube_preview.render_youtube_previews") as render:
        assert main(["editorial", "--config", str(config_file), "--check-config"]) == 0
        render.assert_not_called()
    assert "reach-double-coverage-dedicated.yaml" in capsys.readouterr().out


def test_renderer_uses_explicit_config_not_legacy_environment(tmp_path, monkeypatch):
    from scripts.tjr_youtube_preview import OfficialVideo, render_youtube_previews

    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "wrong")
    monkeypatch.setenv("TJR_TARGET_CHANNEL_ID", "wrong")
    monkeypatch.setenv("TJR_REQUIRE_STAGED_ORIGINAL", "0")
    official = OfficialVideo("_kDrxucOx9g", CHANNEL, "Official", "2026-10-01T00:00:00Z")
    config = EditorialRunConfig(
        brief=BRIEF,
        artifact_root=tmp_path / "drafts",
        source_video_id=official.video_id,
        target_channel_id=CHANNEL,
        browser_capture_file=tmp_path / "missing-stage.json",
        require_staged_original=True,
    )
    seen = []

    def constrain(candidates, requested_id, *, published_after, target_channel_id):
        seen.append((requested_id, target_channel_id))
        return candidates

    with (
        patch.dict(
            render_youtube_previews.__globals__,
            {
                "discover_official_uploads": lambda: ([official], []),
                "_download_asset": lambda _url, path, **_kwargs: path,
                "constrain_official_sources": constrain,
            },
        ),
        pytest.raises(RuntimeError, match="required approved staged original"),
    ):
        render_youtube_previews(config.artifact_root, config.brief, run_config=config)
    assert seen == [(official.video_id, CHANNEL)]
