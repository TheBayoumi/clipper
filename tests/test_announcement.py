"""Warzone policy checks exercise reusable Clipper capabilities, not a parallel editor."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from clipper.cli import main
from clipper_engine import montage
from clipper_engine.profiles import CampaignProfile, load_profile
from clipper_engine.rendering import montage as renderer
from clipper_engine.sources import qa
from clipper_engine.workflow import plan as plan_campaign
from clipper_engine.workflow import qualify as qualify_campaign
from clipper_engine.workflow import render as render_campaign

CAMPAIGN = "warzone-operator-toggle"
VIDEO_NAME = "MW4_MTX_Toggle_v08_16x9_1.mp4"


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    workspace = tmp_path_factory.mktemp("montage-source")
    target = workspace / VIDEO_NAME
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x180:rate=30:duration=5.933333",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=700:sample_rate=48000:duration=5.933333",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-frames:v",
            "178",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-video_track_timescale",
            "15360",
            "-c:a",
            "aac",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-b:a",
            "160k",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    return target


@pytest.fixture()
def profile(tmp_path: Path) -> CampaignProfile:
    values = copy.deepcopy(load_profile(CAMPAIGN).config)
    values["output"]["width"] = 320
    values["output"]["height"] = 180
    values["editorial"]["minimum_state_difference"] = 0.005
    values["editorial"]["minimum_changed_pixel_fraction"] = 0.05
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return load_profile(CAMPAIGN, path)


@pytest.fixture()
def certificate(
    tmp_path: Path, source: Path, profile: CampaignProfile
) -> tuple[Path, dict[str, Any]]:
    resolved = tmp_path / "resolved.json"
    resolved.write_text(
        json.dumps(
            {
                "title": VIDEO_NAME,
                "file_name": VIDEO_NAME,
                "review_url": profile.source_review_url,
                "derivative_type": "source",
                "width": 320,
                "height": 180,
            }
        ),
        encoding="utf-8",
    )
    path = tmp_path / "source_qa.json"
    result = qa.certify(source, "toggle", resolved, path)
    return path, result


def test_profile_loader_is_declarative() -> None:
    assert load_profile("mw4").capability("player_death_detection")["enabled"] is True
    p = load_profile(CAMPAIGN)
    assert p.capability("gameplay_analysis")["enabled"] is False
    assert p.config["kind"] == "announcement"
    assert p.config["editorial"]["selected_text"] in p.config["editorial"]["approved_text"]
    with pytest.raises(ValueError, match="unknown campaign"):
        load_profile("random")


def test_campaign_cli_stage_routes_to_shared_workflow() -> None:
    with patch("clipper.cli.run_campaign", return_value=0) as mocked:
        assert main(["campaign", CAMPAIGN, "discover", "--output", "catalog.json"]) == 0
        assert mocked.call_args.args[0].campaign_command == "discover"
        assert mocked.call_args.args[0].profile == CAMPAIGN


def test_plan_frame_grid_preview(source: Path, profile: CampaignProfile) -> None:
    plan = montage.build_plan(source, profile)
    assert plan["status"] == "PREVIEW_ONLY"
    assert plan["montage"]["output_frames"] == 330
    assert plan["montage"]["full_source_frames"] == 178
    assert plan["montage"]["comparison_frames"] == 82
    assert plan["montage"]["ending_frames"] == 70
    assert plan["audio"]["added_music"] is False
    assert plan["evidence"]["verified_transformation"] is True
    montage.validate_plan(plan, source, profile)


@pytest.mark.parametrize(
    ("field", "value", "code"),
    [
        ("sha256", "incorrect", "source_hash_mismatch"),
        ("derivative_type", "proxy", "source_proxy"),
        ("review_url", "https://other.example/review", "wrong_review"),
    ],
)
def test_plan_rejects_wrong_source(
    source: Path,
    profile: CampaignProfile,
    certificate: tuple[Path, dict[str, Any]],
    field: str,
    value: str,
    code: str,
) -> None:
    _, manifest = certificate
    tampered = {**manifest, field: value}
    with pytest.raises(montage.MontageRejection) as failure:
        montage.build_plan(source, profile, tampered)
    assert failure.value.to_json()["code"] == code


def test_plan_rejects_unapproved_text_and_uncalibrated_source(
    source: Path,
    profile: CampaignProfile,
) -> None:
    config = copy.deepcopy(profile.config)
    config["editorial"]["selected_text"] = "Breaking: Operators BANNED"
    with pytest.raises(montage.MontageRejection, match="unapproved_text"):
        montage.build_plan(source, CampaignProfile(name=CAMPAIGN, config=config))
    config["editorial"]["selected_text"] = config["editorial"]["approved_text"][0]
    config["editorial"]["verified_visual_states"] = {}
    with pytest.raises(montage.MontageRejection, match="uncalibrated_source"):
        montage.build_plan(source, CampaignProfile(name=CAMPAIGN, config=config))


def test_plan_rejects_off_grid_and_unverified_visual_evidence(
    source: Path,
    profile: CampaignProfile,
) -> None:
    config = copy.deepcopy(profile.config)
    config["editorial"]["preferred_output_seconds"] = 11.01
    with pytest.raises(montage.MontageRejection, match="off_frame_grid"):
        montage.build_plan(source, CampaignProfile(name=CAMPAIGN, config=config))
    config["editorial"]["preferred_output_seconds"] = 11
    config["editorial"]["minimum_changed_pixel_fraction"] = 1.01
    with pytest.raises(montage.MontageRejection, match="no_visual_state_change"):
        montage.build_plan(source, CampaignProfile(name=CAMPAIGN, config=config))


def test_plan_validation_detects_tampering(
    source: Path,
    profile: CampaignProfile,
) -> None:
    base = montage.build_plan(source, profile)
    tampered = copy.deepcopy(base)
    tampered["audio"]["added_music"] = True
    with pytest.raises(montage.MontageRejection, match="audio_policy"):
        montage.validate_plan(tampered, source, profile)
    tampered = copy.deepcopy(base)
    tampered["montage"]["approved_on_screen_text"] = "unapproved"
    with pytest.raises(montage.MontageRejection, match="unapproved_text"):
        montage.validate_plan(tampered, source, profile)
    tampered = copy.deepcopy(base)
    tampered["montage"]["output_frames"] += 1
    with pytest.raises(montage.MontageRejection, match="frame_grid_mismatch"):
        montage.validate_plan(tampered, source, profile)
    tampered = copy.deepcopy(base)
    tampered["source"]["sha256"] = "altered"
    with pytest.raises(montage.MontageRejection, match="source_changed"):
        montage.validate_plan(tampered, source, profile)


def test_workflow_rejects_forged_review(
    source: Path,
    profile: CampaignProfile,
    certificate: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    manifest_path, original = certificate
    tampered = {**original, "review_url": "https://unapproved.example/review"}
    manifest_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(montage.MontageRejection, match="source_outside_campaign"):
        plan_campaign(profile, source, manifest_path, tmp_path / "plan.json")


@pytest.fixture(scope="module")
def full_render(
    tmp_path_factory: pytest.TempPathFactory, source: Path
) -> tuple[CampaignProfile, dict[str, Any]]:
    workspace = tmp_path_factory.mktemp("montage-e2e")
    cfg = copy.deepcopy(load_profile(CAMPAIGN).config)
    cfg["output"]["width"] = 320
    cfg["output"]["height"] = 180
    cfg["editorial"]["minimum_state_difference"] = 0.005
    cfg["editorial"]["minimum_changed_pixel_fraction"] = 0.05
    profile = CampaignProfile(name=CAMPAIGN, config=cfg)
    resolved = workspace / "resolved.json"
    resolved.write_text(
        json.dumps(
            {
                "title": VIDEO_NAME,
                "file_name": VIDEO_NAME,
                "derivative_type": "source",
                "review_url": profile.source_review_url,
                "width": 320,
                "height": 180,
            }
        ),
        encoding="utf-8",
    )
    qa_path = workspace / "certified.json"
    qa.certify(source, "toggle", resolved, qa_path)
    plan_path = workspace / "plan.json"
    plan_campaign(profile, source, qa_path, plan_path)
    rendered = render_campaign(profile, source, qa_path, plan_path, workspace / "renders")
    return profile, rendered


def test_certified_montage_full_stack(
    full_render: tuple[CampaignProfile, dict[str, Any]], tmp_path: Path
) -> None:
    profile, result = full_render
    assert result["status"] == "PASS"
    assert result["qa"]["encoded_video_frames"] == 330
    assert 10 <= result["qa"]["encoded_duration_seconds"] <= 12
    assert all(result["staging"]["checks"].values())
    assert all(result["qa"]["checks"].values())
    assert result["qa"]["ssim"] >= 0.96
    assert result["qa"]["psnr_db"] >= 35
    qualified = qualify_campaign(
        profile,
        Path(result["file"]).with_name("render_manifest.json"),
        tmp_path / "acceptance.json",
    )
    assert qualified["status"] == "PASS"
    assert qualified["encoded_frames"] == 330


def test_uncertified_preview_cannot_qualify(
    full_render: tuple[CampaignProfile, dict[str, Any]], tmp_path: Path
) -> None:
    profile, payload = full_render
    forged = {**payload, "status": "PREVIEW_ONLY"}
    path = tmp_path / "preview.json"
    path.write_text(json.dumps(forged), encoding="utf-8")
    with pytest.raises(montage.MontageRejection, match="uncertified_preview"):
        qualify_campaign(profile, path, tmp_path / "qa.json")


def test_qualification_rechecks_delivery_hash(
    full_render: tuple[CampaignProfile, dict[str, Any]], tmp_path: Path
) -> None:
    profile, payload = full_render
    tampered = {**payload, "sha256": "not-a-real-video-hash"}
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(montage.MontageRejection, match="delivery_hash"):
        qualify_campaign(profile, path, tmp_path / "qa.json")


def test_composite_filter_uses_only_source_and_approved_copy(
    source: Path,
    profile: CampaignProfile,
    tmp_path: Path,
) -> None:
    approved = montage.build_plan(source, profile)
    text = tmp_path / "title.txt"
    text.write_text(approved["montage"]["approved_on_screen_text"])
    with patch.object(renderer, "_fontfile", return_value=Path("/etc/hosts")):
        graph = renderer.filter_graph(approved, profile, text, renderer._fontfile())
    assert "[0:v]setpts=PTS-STARTPTS[vfull]" in graph
    assert "[1:v]setpts=PTS-STARTPTS[vcompare]" in graph
    assert "[2:v]setpts=PTS-STARTPTS[vfinal]" in graph
    assert "[vfull][vcompare][vfinal]concat=n=3:v=1:a=0" in graph
    assert "concat=n=3:v=0:a=1" in graph
    assert "[0:a]asplit=3" in graph
    assert "amovie=" not in graph
    assert "drawtext=" in graph
    assert "textfile=" in graph
    assert "[outv]" in graph and "[outa]" in graph
