"""Warzone policy checks exercise reusable Clipper capabilities, not a parallel editor."""

from __future__ import annotations

import copy
import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from clipper.cli import main
from clipper_engine import montage
from clipper_engine.profiles import CampaignProfile, load_profile
from clipper_engine.rendering import montage as renderer
from clipper_engine.rendering import panel_compositor, portrait_matte
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
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_trc",
            "bt709",
            "-color_primaries",
            "bt709",
            "-chroma_sample_location",
            "left",
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
    values["output"]["portrait_matte"]["enabled"] = False
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


def test_campaign_cli_accepts_profile_configured_cascade_mode() -> None:
    with patch("clipper.cli.run_campaign", return_value=0) as mocked:
        assert (
            main(
                [
                    "campaign",
                    CAMPAIGN,
                    "plan",
                    "--source",
                    "source.mp4",
                    "--output",
                    "cascade.json",
                    "--comparison-mode",
                    "cascade",
                ]
            )
            == 0
        )
    args = mocked.call_args.args[0]
    assert args.profile == CAMPAIGN
    assert args.campaign_command == "plan"
    assert args.comparison_mode == "cascade"


def test_plan_frame_grid_preview(source: Path, profile: CampaignProfile) -> None:
    plan = montage.build_plan(source, profile)
    assert plan["status"] == "PREVIEW_ONLY"
    assert plan["montage"]["output_frames"] == 315
    assert plan["montage"]["full_source_frames"] == 178
    assert plan["montage"]["comparison_frames"] == 60
    assert plan["montage"]["ending_frames"] == 62
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
    cfg["output"]["portrait_matte"]["enabled"] = False
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
    assert result["qa"]["encoded_video_frames"] == 315
    assert 10 <= result["qa"]["encoded_duration_seconds"] <= 12
    assert all(result["staging"]["checks"].values())
    # FFmpeg/x264 builds differ in which color fields they emit for synthetic
    # lavfi input; all explicitly emitted source fields must survive exactly.
    expected_colors = result["staging"]["source_color_metadata"]
    assert expected_colors["color_range"] == "tv"
    assert expected_colors["color_space"] == "bt709"
    assert result["staging"]["stage_color_tags"] == expected_colors
    assert result["qa"]["canonical_color_metadata_tags"] == expected_colors
    assert result["qa"]["delivery_color_metadata"] == expected_colors
    assert all(result["qa"]["checks"].values())
    assert result["qa"]["ssim"] >= 0.96
    assert result["qa"]["psnr_db"] >= 35
    qualified = qualify_campaign(
        profile,
        Path(result["file"]).with_name("render_manifest.json"),
        tmp_path / "acceptance.json",
    )
    assert qualified["status"] == "PASS"
    assert qualified["encoded_frames"] == 315


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
    assert "[3:v]setpts=PTS-STARTPTS[vhook]" in graph
    assert "[0:v]setpts=PTS-STARTPTS[vfull]" in graph
    assert "[1:v]setpts=PTS-STARTPTS[vcompare]" in graph
    assert "[2:v]setpts=PTS-STARTPTS[vfinal]" in graph
    assert "[vhook][vfull][vcompare][vfinal]concat=n=4:v=1:a=0" in graph
    assert "[0:a]asplit=4" in graph
    assert "concat=n=4:v=0:a=1" in graph
    assert "enable='between(n,20,111)'" in graph
    assert graph.count("afade=t=in") == 4
    assert graph.count("afade=t=out") == 4
    assert "amovie=" not in graph
    assert "drawtext=" in graph
    assert "textfile=" in graph
    assert "[outv]" in graph and "[outa]" in graph


@pytest.mark.parametrize(
    ("mode", "text_choice"),
    [("wipe", 1), ("cuts", 2)],
)
def test_editorial_modes_preserve_approved_text_and_legal_windows(
    source: Path, profile: CampaignProfile, mode: str, text_choice: int
) -> None:
    p = montage.build_plan(source, profile, comparison_mode=mode, approved_text_index=text_choice)
    montage.validate_plan(p, source, profile)
    edit = p["montage"]
    assert edit["comparison_mode"] == mode
    assert edit["hook"] == {"start_frame": 55, "frames": 15}
    assert edit["full_source_frames"] == 178
    assert edit["comparison_frames"] == 60
    assert edit["ending_frames"] == 62
    assert edit["output_frames"] == 315
    assert " ".join(edit["title"]["lines"]) == edit["approved_on_screen_text"]
    assert edit["title"]["position"] == "upper_right"
    assert edit["title"]["start_frame"] == 20
    assert edit["title"]["end_frame"] == 112
    assert sum(x["frames"] for x in edit["ending_shots"]) == 62
    assert (
        edit["approved_on_screen_text"]
        == profile.config["editorial"]["approved_text"][text_choice - 1]
    )


def test_reject_invalid_mode_and_tampered_approval(source: Path, profile: CampaignProfile) -> None:
    with pytest.raises(montage.MontageRejection, match="invalid_comparison_mode"):
        montage.build_plan(source, profile, comparison_mode="half_strip")
    p = montage.build_plan(source, profile)
    p["montage"]["title"]["lines"] = ["IMPROVISED HOOK"]
    with pytest.raises(montage.MontageRejection, match="title_changed"):
        montage.validate_plan(p, source, profile)


def test_alternate_comparison_render_fills_frame(
    source: Path, profile: CampaignProfile, tmp_path: Path
) -> None:
    plan = montage.build_plan(source, profile, comparison_mode="cuts", approved_text_index=2)
    result = renderer.render(source, profile, plan, tmp_path / "cuts")
    assert result["status"] == "PREVIEW_ONLY"
    assert result["qa"]["encoded_video_frames"] == 315
    assert result["qa"]["checks"]["full_frame_comparison"]
    assert result["qa"]["checks"]["hook_source_hashes_exact"]
    assert result["qa"]["checks"]["reveal_source_hashes_exact"]
    assert result["staging"]["comparison"]["comparison_mode"] == "cuts"
    frame = (
        plan["montage"]["hook"]["frames"]
        + plan["montage"]["full_source_frames"]
        + plan["montage"]["comparison_frames"] // 2
    )
    # The old strip had completely black upper and lower thirds.
    completed = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(result["file"]),
            "-vf",
            f"select=eq(n\\,{frame}),format=gray",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    image = np.frombuffer(completed.stdout, dtype=np.uint8).reshape(180, 320)
    assert float(np.mean(image[:25, :])) > 12.0
    assert float(np.mean(image[-25:, :])) > 12.0


def test_toggle_word_follows_exact_visual_state_timeline(
    source: Path, profile: CampaignProfile
) -> None:
    plan = montage.build_plan(source, profile)
    assert plan["evidence"]["toggle_motion_start_frame"] == 57
    assert plan["evidence"]["toggle_motion_end_frame"] == 64
    progress = portrait_matte.toggle_progress
    expected = {
        0: 0.0,
        9: 1.0,
        15: 0.0,
        72: 0.0,
        79: 1.0,
        193: 0.0,
        203: 0.0,
        223: 0.5,
        243: 1.0,
        253: 0.0,
        263: 0.0,
        264: 1.0,
        274: 0.0,
        281: 1.0,
        314: 1.0,
    }
    for output_frame, visual_state in expected.items():
        assert progress(output_frame, plan, profile) == pytest.approx(visual_state)
    other = montage.build_plan(source, profile, comparison_mode="cuts")
    assert progress(206, other, profile) == 0.0
    assert progress(207, other, profile) == 1.0
    assert progress(224, other, profile) == 1.0
    assert progress(225, other, profile) == 0.0
    assert progress(233, other, profile) == 0.0
    assert progress(234, other, profile) == 1.0


@pytest.mark.parametrize(("mode", "text_choice"), [("wipe", 1), ("cuts", 2)])
def test_certified_portrait_delivery_sync_and_full_duration_copy(
    source: Path,
    profile: CampaignProfile,
    certificate: tuple[Path, dict[str, Any]],
    tmp_path: Path,
    mode: str,
    text_choice: int,
) -> None:
    values = copy.deepcopy(profile.config)
    values["output"]["portrait_matte"]["enabled"] = True
    values["output"]["portrait_matte"]["width"] = 180
    values["output"]["portrait_matte"]["height"] = 320
    values["output"]["portrait_matte"]["target_size_mb"] = 2.0
    values["output"]["portrait_matte"]["size_tolerance_mb"] = 1.0
    p = CampaignProfile(name=CAMPAIGN, config=values)
    manifest_path, _ = certificate
    plan_path = tmp_path / "portrait_plan.json"
    plan = plan_campaign(
        p,
        source,
        manifest_path,
        plan_path,
        comparison_mode=mode,
        approved_text_index=text_choice,
    )
    result = render_campaign(p, source, manifest_path, plan_path, tmp_path / "portrait")
    portrait = result["portrait"]
    assert portrait is not None
    assert portrait["qa"]["frame_count"] == 315
    assert portrait["qa"]["encoded_duration"] == pytest.approx(10.5, abs=0.055)
    assert portrait["title"]["text_visible_frames"] == 315
    assert portrait["title"]["approved_copy"] == plan["montage"]["approved_on_screen_text"]
    assert portrait["title"]["progress_samples"]["0"] == 0.0
    assert portrait["title"]["progress_samples"]["314"] == 1.0
    assert all(portrait["qa"]["checks"].values())
    assert portrait["qa"]["matte_rgb_max_error"] <= 2
    assert 1.0 <= portrait["qa"]["file_size_mb"] <= 3.0
    accepted = qualify_campaign(
        p,
        tmp_path / "portrait" / "render_manifest.json",
        tmp_path / "portrait_acceptance.json",
    )
    assert accepted["status"] == "PASS"
    assert accepted["primary_delivery"] == portrait["file"]
    assert accepted["portrait_text_visible_frames"] == 315

    manifest = json.loads((tmp_path / "portrait" / "render_manifest.json").read_text())
    manifest["portrait"]["qa"]["checks"] = {}
    bad = tmp_path / "portrait_qa_bad.json"
    bad.write_text(json.dumps(manifest))
    with pytest.raises(montage.MontageRejection, match="portrait_qa"):
        qualify_campaign(p, bad, tmp_path / "wrong_qa.json")
    manifest["portrait"] = None
    missing = tmp_path / "portrait_missing.json"
    missing.write_text(json.dumps(manifest))
    with pytest.raises(montage.MontageRejection, match="portrait_missing"):
        qualify_campaign(p, missing, tmp_path / "missing_qa.json")


def test_plan_rejects_tampered_verified_toggle_event(
    source: Path, profile: CampaignProfile
) -> None:
    wrong = copy.deepcopy(profile.config)
    wrong["output"]["portrait_matte"]["enabled"] = True
    p = CampaignProfile(name=CAMPAIGN, config=wrong)
    plan = montage.build_plan(source, p)
    assert plan["status"] == "PREVIEW_ONLY"
    with pytest.raises(montage.MontageRejection, match="toggle_event_changed"):
        plan["evidence"]["toggle_motion_start_frame"] = 1
        montage.validate_plan(plan, source, p)


def _cascade_test_profile(profile: CampaignProfile) -> CampaignProfile:
    values = copy.deepcopy(profile.config)
    values["output"]["portrait_matte"]["enabled"] = True
    values["output"]["portrait_matte"]["width"] = 180
    values["output"]["portrait_matte"]["height"] = 320
    values["output"]["portrait_matte"]["target_size_mb"] = 1.0
    values["output"]["portrait_matte"]["size_tolerance_mb"] = 0.5
    return CampaignProfile(name=CAMPAIGN, config=values)


def test_cascade_is_configured_by_profile_not_a_second_pipeline(
    source: Path, profile: CampaignProfile
) -> None:
    p = _cascade_test_profile(profile)
    assert p.config["editorial"]["minimum_output_seconds"] == 5
    plan = montage.build_plan(source, p, comparison_mode="cascade")
    edit = plan["montage"]
    assert edit["output_frames"] == 165
    assert edit["output_seconds"] == 5.5
    assert edit["full_source_frames"] == 178  # Original master, certified in full.
    assert edit["source_window"] == {"start_frame": 14, "frames": 96}
    assert edit["hook"] == {"start_frame": 55, "frames": 10}
    assert edit["comparison_frames"] == 39
    assert [shot["frames"] for shot in edit["ending_shots"]] == [5, 5, 5, 5]
    assert edit["comparison_mode"] == "cascade"
    start = edit["hook"]["frames"] + edit["source_window"]["frames"]
    expected = {
        start: [0, 0, 0],
        start + 3: [0, 0, 0],
        start + 6: [1, 0, 0],
        start + 18: [1, 1, 0],
        start + 30: [1, 1, 1],
        145: [0, 0, 0],
        150: [1, 1, 1],
        155: [0, 0, 0],
        160: [1, 1, 1],
        164: [1, 1, 1],
    }
    for frame, expected_states in expected.items():
        fill = portrait_matte.toggle_progress(frame, plan, p)
        actual = panel_compositor.states_for_output(frame, plan, p, fill)
        assert actual == pytest.approx(expected_states)
        assert fill == pytest.approx(sum(expected_states) / 3)
    for local in range(edit["comparison_frames"]):
        frame = start + local
        fill = portrait_matte.toggle_progress(frame, plan, p)
        assert fill == pytest.approx(sum(panel_compositor.stage_progress(local, plan, p)) / 3)
    original = montage.build_plan(source, p, comparison_mode="wipe")
    assert original["montage"]["output_frames"] == 315
    assert original["montage"]["source_window"] == {"start_frame": 0, "frames": 178}
    assert original["montage"]["comparison_frames"] == 60


@pytest.mark.parametrize(
    "bad_window",
    [
        {"start_frame": 21, "frames": 96},  # Excludes verified BEFORE anchor.
        {"start_frame": 14, "frames": 94},  # Excludes verified AFTER anchor.
        {"start_frame": -1, "frames": 96},
        {"start_frame": 14, "frames": 165},
        {"start_frame": 14, "frames": 0},
        {"start_frame": 14},
    ],
)
def test_source_excerpt_must_retain_both_verified_visual_states(
    source: Path, profile: CampaignProfile, bad_window: dict[str, int]
) -> None:
    p = _cascade_test_profile(profile)
    p.config["editorial"]["mode_timing"]["cascade"]["source_window"] = bad_window
    with pytest.raises(montage.MontageRejection, match="source_window_invalid"):
        montage.build_plan(source, p, comparison_mode="cascade")


@pytest.mark.parametrize(
    ("key", "bad_value"),
    [
        ("hook_frames", 6),
        ("ending_shot_frames", [5, 5, 8, 5]),
        ("ending_shot_frames", [5, 5, 5]),
        ("comparison_seconds", 1.01),
        ("maximum_output_seconds", 5),
    ],
)
def test_cascade_rejects_invalid_short_edit_timing(
    source: Path, profile: CampaignProfile, key: str, bad_value: object
) -> None:
    p = _cascade_test_profile(profile)
    p.config["editorial"]["mode_timing"]["cascade"][key] = bad_value
    with pytest.raises(
        montage.MontageRejection,
        match=r"mode_timing_invalid|off_frame_grid|invalid_duration_config",
    ):
        montage.build_plan(source, p, comparison_mode="cascade")


def test_campaign_five_second_minimum_is_not_confused_with_editing_target(
    source: Path, profile: CampaignProfile
) -> None:
    p = _cascade_test_profile(profile)
    p.config["editorial"]["mode_timing"]["cascade"]["preferred_output_seconds"] = 4
    with pytest.raises(montage.MontageRejection, match="invalid_duration_config"):
        montage.build_plan(source, p, comparison_mode="cascade")


@pytest.mark.parametrize(
    ("bad_key", "bad_value"),
    [
        ("operator_rois", [[0, 0, 2, 1]] * 3),
        ("switch_start_frames", [40, 20, 8]),
        ("switch_order", [0, 0, 2]),
        ("transition_frames", 40),
    ],
)
def test_cascade_rejects_invalid_calibration(
    source: Path, profile: CampaignProfile, bad_key: str, bad_value: object
) -> None:
    p = _cascade_test_profile(profile)
    p.config["output"]["portrait_matte"]["cascade"][bad_key] = bad_value
    with pytest.raises(montage.MontageRejection, match="cascade_"):
        montage.build_plan(source, p, comparison_mode="cascade")


def test_cascade_produces_full_duration_original_source_portrait(
    source: Path,
    profile: CampaignProfile,
    certificate: tuple[Path, dict[str, Any]],
    tmp_path: Path,
) -> None:
    p = _cascade_test_profile(profile)
    plan_file = tmp_path / "plan_cascade.json"
    plan = plan_campaign(
        p,
        source,
        certificate[0],
        plan_file,
        comparison_mode="cascade",
        approved_text_index=1,
    )
    assert plan["montage"]["comparison_mode"] == "cascade"
    output = tmp_path / "cascade"
    rendered = render_campaign(p, source, certificate[0], plan_file, output)
    qa = rendered["portrait"]["qa"]
    panel_qa = rendered["portrait"]["cascade"]
    assert rendered["staging"]["comparison"]["comparison_mode"] == "cascade"
    assert rendered["staging"]["comparison"]["frame_count_exact"]
    assert rendered["staging"]["comparison"]["full_frame"]
    assert rendered["staging"]["source_excerpt"]["windows"] == [{"start_frame": 14, "frames": 96}]
    assert rendered["staging"]["source_excerpt"]["source_to_piece_hashes_exact"]
    assert rendered["staging"]["source_excerpt"]["frame_count"] == 96
    assert rendered["qa"]["checks"]["source_excerpt_hashes_exact"]
    assert panel_qa["frame_count"] == 165
    assert panel_qa["panel_count"] == 3
    assert panel_qa["text_synced"] is True
    assert panel_qa["switch_frames"] == [109, 121, 133]
    assert panel_qa["sampled_states"]["106"] == [0, 0, 0]
    assert panel_qa["sampled_states"]["144"] == [1, 1, 1]
    assert (
        panel_qa["source_still_sha256"]
        == (rendered["staging"]["comparison"]["source_still_sha256"])
    )
    assert all(qa["checks"].values())
    assert len(qa["cascade_panel_pixel_differences"]) == 3
    assert all(delta > 2.5 for delta in qa["cascade_panel_pixel_differences"])
    assert qa["frame_count"] == 165
    assert qa["encoded_duration"] == pytest.approx(5.5, abs=0.055)
    assert qa["file_size_mb"] == pytest.approx(1.0, abs=0.5)
    assert len(qa["matte_sampled_frames"]) == 3
    assert all(sample["max_error"] <= 2 for sample in qa["matte_sampled_frames"])
    assert rendered["portrait"]["title"]["text_visible_frames"] == 165
    assert rendered["portrait"]["title"]["progress_samples"]["0"] == 0
    assert rendered["portrait"]["title"]["progress_samples"]["164"] == 1
    accepted = qualify_campaign(
        p, output / "render_manifest.json", tmp_path / "cascade_acceptance.json"
    )
    assert accepted["status"] == "PASS"
    assert accepted["primary_delivery"] == rendered["portrait"]["file"]
    manifest = json.loads((output / "render_manifest.json").read_text())
    manifest["portrait"]["cascade"]["switch_frames"] = [0, 1, 2]
    tampered = tmp_path / "invalid_cascade.json"
    tampered.write_text(json.dumps(manifest))
    with pytest.raises(montage.MontageRejection, match="cascade_schedule"):
        qualify_campaign(p, tampered, tmp_path / "rejected_acceptance.json")
    manifest["portrait"]["cascade"]["source_still_sha256"]["before"] = "tampered"
    tampered.write_text(json.dumps(manifest))
    with pytest.raises(montage.MontageRejection, match="cascade_stills"):
        qualify_campaign(p, tampered, tmp_path / "rejected_stills.json")
    manifest["portrait"]["cascade"] = copy.deepcopy(rendered["portrait"]["cascade"])
    manifest["staging"]["source_excerpt"]["source_to_piece_hashes_exact"] = False
    tampered.write_text(json.dumps(manifest))
    with pytest.raises(montage.MontageRejection, match="source_excerpt"):
        qualify_campaign(p, tampered, tmp_path / "rejected_excerpt.json")


def test_cascade_is_covered_by_existing_official_qualification_workflow() -> None:
    workflow = Path(".github/workflows/warzone-qualification.yml").read_text()
    assert "--comparison-mode cascade" in workflow
    assert "delivery_cascade/render_manifest.json" in workflow
    assert "delivery_cascade/*.mp4" in workflow
    assert workflow.index("Plan certified-source three-Operator cascade") < workflow.index(
        "Plan source-grounded legal visual-state comparison"
    )
    assert "warzone-cascade-" + chr(36) + "{{ github.sha }}" in workflow
