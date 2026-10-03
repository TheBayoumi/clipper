"""Campaign-specific technical and encoding guards."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from clipper.models import ClipCandidate
from clipper.render import RenderError, build_ffmpeg_command

# pytest's entrypoint does not always include the repository root in sys.path.
# Load the standalone workflow script from its explicit repository-relative path.
_tjr_qa = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "tjr_quality.py"))
QualityError = _tjr_qa["QualityError"]
check_campaign_brief = _tjr_qa["check_campaign_brief"]
probe_video = _tjr_qa["probe_video"]
probe_original = _tjr_qa["probe_original"]
prepare_staged_brief = _tjr_qa["prepare_staged_brief"]
validate_artifacts = _tjr_qa["validate_artifacts"]
_resolve_approved_youtube_video = _tjr_qa["_resolve_approved_youtube_video"]


@pytest.fixture
def campaign_brief(tmp_path: Path) -> Path:
    path = Path("campaigns/reach-double-coverage-dedicated.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = tmp_path / "campaign.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


def test_campaign_accepts_verified_video_ids(campaign_brief: Path) -> None:
    data = check_campaign_brief(campaign_brief)
    assert data["allowed_video_ids"] == []
    assert data["watermark_url"] == (
        "https://drive.google.com/file/d/1wn3gL5h7cUZn-8jzTfwkAvLhbleXgbL6/view?usp=sharing"
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_channel_ids", ["UC-not-double-coverage"]),
        ("allowed_video_ids", ["not-a-video-id"]),
        ("allowed_video_ids", ["8PYgFVB0GHE", "8PYgFVB0GHE"]),
        ("watermark_text", "Branding"),
        ("rights_confirmed", False),
        ("required_hashtags", ["#other"]),
        ("source_media_urls", {"8PYgFVB0GHE": "https://example.com/video.mp4"}),
    ],
)
def test_campaign_rejects_unsafe_changes(campaign_brief: Path, field: str, value: object) -> None:
    data = yaml.safe_load(campaign_brief.read_text(encoding="utf-8"))
    data[field] = value
    campaign_brief.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(QualityError):
        check_campaign_brief(campaign_brief)


def test_tjr_hd_render_profile(tmp_path: Path) -> None:
    clip = ClipCandidate("8PYgFVB0GHE", 4, 30, "text", 2.0)
    with patch.dict(
        "os.environ",
        {
            "CLIPPER_RENDER_PRESET": "slow",
            "CLIPPER_RENDER_CRF": "18",
            "CLIPPER_RENDER_THREADS": "2",
        },
    ):
        command = build_ffmpeg_command("source.mp4", "out.mp4", clip, tmp_path / "captions.srt")
    assert command[command.index("-preset") + 1] == "slow"
    assert command[command.index("-crf") + 1] == "18"
    assert command[command.index("-threads") + 1] == "2"
    assert "scale=1080:1920" in " ".join(command)
    assert "setsar=1" in " ".join(command)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CLIPPER_RENDER_PRESET", "malicious"),
        ("CLIPPER_RENDER_CRF", "text"),
        ("CLIPPER_RENDER_CRF", "51"),
        ("CLIPPER_RENDER_THREADS", "0"),
        ("CLIPPER_RENDER_THREADS", "nope"),
    ],
)
def test_render_fails_closed(tmp_path: Path, name: str, value: str) -> None:
    clip = ClipCandidate("v", 0, 23, "text", 1.0)
    with patch.dict("os.environ", {name: value}), pytest.raises(RenderError):
        build_ffmpeg_command("src.mp4", "out.mp4", clip, tmp_path / "captions.srt")


def test_probe_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(QualityError, match="empty or missing"):
        probe_video(tmp_path / "not-created.mp4")


def test_staged_brief_requires_verified_budget_source_and_hash(
    campaign_brief: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "runtime" / "brief.yaml"
    url = "https://drive.google.com/file/d/ApprovedFile123/view?usp=sharing"
    monkeypatch.setenv("TJR_SOURCE_MEDIA_URL", url)
    monkeypatch.setenv("TJR_SOURCE_MEDIA_SHA256", "a" * 64)
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "p2LU37eat70")
    monkeypatch.setitem(
        prepare_staged_brief.__globals__,
        "_resolve_approved_youtube_video",
        lambda video_id: "UCf1q6dhccWr6eQEcFFnJSbA",
    )
    with pytest.raises(QualityError, match="confirm live campaign budget"):
        prepare_staged_brief(campaign_brief, output)
    monkeypatch.setenv("TJR_BUDGET_CONFIRMED", "true")
    monkeypatch.setenv("TJR_SOURCE_VERIFIED", "true")
    assert prepare_staged_brief(campaign_brief, output) == output
    parsed = check_campaign_brief(output)
    assert parsed["source_media_urls"] == {"p2LU37eat70": url}
    assert parsed["source_channel_ids"] == ["UCf1q6dhccWr6eQEcFFnJSbA"]
    monkeypatch.setenv("TJR_SOURCE_MEDIA_SHA256", "invalid-hash")
    with pytest.raises(QualityError, match="64-character"):
        prepare_staged_brief(campaign_brief, tmp_path / "invalid.yaml")
    monkeypatch.setenv("TJR_SOURCE_MEDIA_SHA256", "a" * 64)
    monkeypatch.setenv("TJR_SOURCE_MEDIA_URL", "https://untrusted.invalid/asset.mp4")
    with pytest.raises(QualityError, match="Google Drive"):
        prepare_staged_brief(campaign_brief, tmp_path / "rejected.yaml")
    assert not (tmp_path / "rejected.yaml").exists()


def test_staged_original_must_have_real_hd_resolution(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"test source bytes")
    hd = Mock(
        stdout=json.dumps({"streams": [{"codec_type": "video", "width": 1920, "height": 1080}]})
    )
    sd = Mock(
        stdout=json.dumps({"streams": [{"codec_type": "video", "width": 640, "height": 480}]})
    )
    with patch("subprocess.run", return_value=hd):
        assert probe_original(source) == {"width": 1920, "height": 1080}
    with (
        patch("subprocess.run", return_value=sd),
        pytest.raises(QualityError, match="below production"),
    ):
        probe_original(source)


def test_staged_original_hash_must_match(
    campaign_brief: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "TJR_SOURCE_MEDIA_URL",
        "https://drive.google.com/file/d/ApprovedFile123/view",
    )
    monkeypatch.setenv("TJR_SOURCE_MEDIA_SHA256", "a" * 64)
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "p2LU37eat70")
    monkeypatch.setitem(
        prepare_staged_brief.__globals__,
        "_resolve_approved_youtube_video",
        lambda video_id: "UCf1q6dhccWr6eQEcFFnJSbA",
    )
    monkeypatch.setenv("TJR_BUDGET_CONFIRMED", "true")
    monkeypatch.setenv("TJR_SOURCE_VERIFIED", "true")
    runtime = prepare_staged_brief(campaign_brief, tmp_path / "runtime.yaml")
    run = tmp_path / "artifacts" / "sample"
    original = run / "work" / "p2LU37eat70" / "source.mp4"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"this does not match the pinned SHA-256")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "errors": [],
                "discovered_videos": [{"channel_id": "UCf1q6dhccWr6eQEcFFnJSbA"}],
                "planned_clips": [{"video_id": "p2LU37eat70"}],
                "rendered_clips": [{"video_id": "8PYgFVB0GHE"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(QualityError, match="hash differs"):
        validate_artifacts(runtime, tmp_path / "artifacts")


def test_staged_source_rejects_non_campaign_video_id(
    campaign_brief: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "TJR_SOURCE_MEDIA_URL", "https://drive.google.com/file/d/ApprovedFile123/view"
    )
    monkeypatch.setenv("TJR_SOURCE_MEDIA_SHA256", "a" * 64)
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "invalid")
    monkeypatch.setenv("TJR_BUDGET_CONFIRMED", "true")
    monkeypatch.setenv("TJR_SOURCE_VERIFIED", "true")
    output = tmp_path / "rejected-brief.yaml"
    with pytest.raises(QualityError, match="11-character YouTube"):
        _resolve_approved_youtube_video("invalid")
    with pytest.raises(QualityError, match="11-character YouTube"):
        prepare_staged_brief(campaign_brief, output)
    assert not output.exists()


def test_known_tjr_livestream_editorial_layout_crops_source_sponsor_area(tmp_path: Path) -> None:
    clip = ClipCandidate("p2LU37eat70", 8, 37, "authentic quote", 10.0)
    command = build_ffmpeg_command(
        "original-1920x1080.mp4",
        "review.mp4",
        clip,
        tmp_path / "captions.srt",
        editorial_layout="tjr-trading-logo-safe",
    )
    graph = command[command.index("-filter_complex") + 1]
    assert "crop=1460:600:280:20" in graph
    assert "crop=565:335:20:710" in graph
    assert "subtitles=" in graph
    assert "gblur" not in graph
    with pytest.raises(RenderError, match="rejects logos"):
        build_ffmpeg_command(
            "original.mp4",
            "review.mp4",
            clip,
            tmp_path / "captions.srt",
            editorial_layout="tjr-trading-logo-safe",
            watermark_path=tmp_path / "brand.png",
        )


def test_native_23976_and_25fps_match_original_qa(tmp_path: Path) -> None:
    from fractions import Fraction

    for value in ("24000/1001", "25/1"):
        good = Mock(
            stdout=json.dumps(
                {
                    "streams": [
                        {
                            "codec_type": "video",
                            "codec_name": "h264",
                            "pix_fmt": "yuv420p",
                            "sample_aspect_ratio": "1:1",
                            "width": 1080,
                            "height": 1920,
                            "avg_frame_rate": value,
                        },
                        {"codec_type": "audio", "codec_name": "aac"},
                    ],
                    "format": {"duration": "28.3"},
                }
            )
        )
        clip = tmp_path / "clip.mp4"
        clip.write_bytes(b"authentic fixture data")
        with patch("subprocess.run", return_value=good):
            assert probe_video(clip)["fps"] == pytest.approx(float(Fraction(value)))


def test_mirror_staging_uses_exact_hash_and_a_shared_source_manifest(
    campaign_brief: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    stage_verified_mirror = _tjr_qa["stage_verified_mirror"]
    original = b"test original video bytes"
    expected = hashlib.sha256(original).hexdigest()
    mirror_url = "https://drive.google.com/file/d/ApprovedFile123/view"
    monkeypatch.setenv("TJR_SOURCE_MEDIA_URL", mirror_url)
    monkeypatch.setenv("TJR_SOURCE_MEDIA_SHA256", expected)
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "p2LU37eat70")
    monkeypatch.setenv("TJR_BUDGET_CONFIRMED", "true")
    monkeypatch.setenv("TJR_SOURCE_VERIFIED", "true")
    monkeypatch.setitem(
        prepare_staged_brief.__globals__,
        "_resolve_approved_youtube_video",
        lambda video_id: "UCf1q6dhccWr6eQEcFFnJSbA",
    )
    brief = prepare_staged_brief(campaign_brief, tmp_path / "approved.yaml")
    output = tmp_path / "stage" / "original.mp4"
    manifest = tmp_path / "stage" / "manifest.json"

    def download(url: str, destination: Path, **kwargs: object) -> Path:
        assert url == mirror_url and kwargs["expected_kind"] == "media"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(original)
        return destination

    probe = Mock(
        stdout=json.dumps({"streams": [{"codec_type": "video", "width": 1920, "height": 1080}]})
    )
    duration = Mock(stdout="125.5\n")
    with (
        patch("clipper.pipeline._download_asset", side_effect=download),
        patch("subprocess.run", side_effect=[probe, duration]),
    ):
        assert stage_verified_mirror(brief, output, manifest) == manifest
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["source_sha256"] == expected
    assert record["video_id"] == "p2LU37eat70"
    assert record["channel_id"] == "UCf1q6dhccWr6eQEcFFnJSbA"
    assert record["source_transport"] == "approved_sha256_mirror"
    assert record["duration"] == 125.5
    assert "drive.google.com" not in manifest.read_text(encoding="utf-8")


def test_staging_rejects_wrong_hash_and_deletes_untrusted_media(
    campaign_brief: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage_verified_mirror = _tjr_qa["stage_verified_mirror"]
    monkeypatch.setenv(
        "TJR_SOURCE_MEDIA_URL", "https://drive.google.com/file/d/ApprovedFile123/view"
    )
    monkeypatch.setenv("TJR_SOURCE_MEDIA_SHA256", "a" * 64)
    monkeypatch.setenv("TJR_SOURCE_VIDEO_ID", "p2LU37eat70")
    monkeypatch.setenv("TJR_BUDGET_CONFIRMED", "true")
    monkeypatch.setenv("TJR_SOURCE_VERIFIED", "true")
    monkeypatch.setitem(
        prepare_staged_brief.__globals__,
        "_resolve_approved_youtube_video",
        lambda video_id: "UCf1q6dhccWr6eQEcFFnJSbA",
    )
    brief = prepare_staged_brief(campaign_brief, tmp_path / "approved.yaml")
    output = tmp_path / "stage" / "original.mp4"
    manifest = tmp_path / "stage" / "manifest.json"

    def untrusted_download(url: str, destination: Path, **kwargs: object) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"wrong mirror bytes")
        return destination

    with (
        patch("clipper.pipeline._download_asset", side_effect=untrusted_download),
        pytest.raises(QualityError, match="SHA-256 differs"),
    ):
        stage_verified_mirror(brief, output, manifest)
    assert not output.exists()
    assert not manifest.exists()


def test_production_workflow_validates_inputs_and_avoids_duplicate_renders() -> None:
    from yaml.nodes import MappingNode, SequenceNode

    workflow_path = Path(".github/workflows/tjr-weekly-hd.yml")
    source = workflow_path.read_text(encoding="utf-8")
    root = yaml.compose(source, Loader=yaml.BaseLoader)
    assert root is not None

    def check_unique_keys(node: yaml.Node) -> None:
        if isinstance(node, MappingNode):
            keys = [str(key.value) for key, _ in node.value]
            assert len(keys) == len(set(keys)), f"duplicate YAML key: {keys}"
            for _, value in node.value:
                check_unique_keys(value)
        elif isinstance(node, SequenceNode):
            for child in node.value:
                check_unique_keys(child)

    check_unique_keys(root)
    config = yaml.safe_load(source)
    events = config.get("on", config.get(True))
    inputs = events["workflow_dispatch"]["inputs"]
    assert inputs["source_mode"]["default"] == "validate_only"
    assert inputs["source_video_id"]["required"] is False
    assert inputs["target_channel_id"]["required"] is False
    assert "clip_limit" not in inputs
    assert inputs["source_verified"]["required"] is False
    assert inputs["budget_confirmed"]["required"] is False
    assert set(inputs["source_mode"]["options"]) == {
        "validate_only",
        "modal_direct",
        "youtube_direct",
        "verified_mirror",
        "feedback_replay",
    }
    jobs = config["jobs"]
    preflight = next(
        item
        for item in jobs["tests"]["steps"]
        if item.get("name") == "Validate production inputs before any real source acquisition"
    )
    assert "TJR_BUDGET_CONFIRMED" in preflight["run"]
    target_channel_error = "Direct production requires exactly one Reach-listed target_channel_id"
    assert target_channel_error in preflight["run"]
    code = preflight["run"].split("python - <<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
    channel = "UCf1q6dhccWr6eQEcFFnJSbA"
    for video_id, target, expected in (
        ("2Y4LP85PTak", channel, 0),
        ("malformed", channel, 1),
        ("2Y4LP85PTak", "unapproved", 1),
    ):
        result = subprocess.run(
            [sys.executable, "-c", code],
            env={
                **os.environ,
                "TJR_SOURCE_MODE": "youtube_direct",
                "TJR_BUDGET_CONFIRMED": "false",
                "TJR_SOURCE_VIDEO_ID": video_id,
                "TJR_TARGET_CHANNEL_ID": target,
            },
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == expected, result.stderr
    assert "clip_limit" not in preflight["run"]
    browser = jobs["youtube_preview"]
    assert browser["needs"] == "tests"
    assert "inputs.source_mode == 'youtube_direct'" in browser["if"]
    assert "youtube_modal_egress" not in browser["if"]
    modal = jobs["youtube_modal_egress"]
    assert set(modal["outputs"]) == {"cache_run_id"}
    assert "inputs.source_mode == 'modal_direct'" in modal["if"]
    modal_steps = {item.get("name"): item for item in modal["steps"] if item.get("name")}
    acquisition = modal_steps["Reuse completed pipeline or acquire missing verified source"]
    assert "continue-on-error" not in acquisition
    assert "Classify Modal acquisition outcome" not in modal_steps
    assert "youtube_alternate_egress" not in jobs
    for name in ("youtube_preview", "youtube_modal_egress", "render"):
        assert jobs[name]["env"]["TJR_CAPTION_STYLE"] == "B2"
    assert jobs["youtube_modal_egress"]["env"]["TJR_RENDER_SAFETY_LIMIT"] == "20"
    assert jobs["youtube_modal_egress"]["env"]["TJR_MODAL_CHANNEL_ID"] == (
        "${{ inputs.target_channel_id }}"
    )
    mirror_script = " ".join(item.get("run", "") for item in jobs["render"]["steps"])
    assert "scripts.tjr_quality --stage" in mirror_script
    assert "scripts.tjr_youtube_preview" in mirror_script
    assert "clipper run" not in mirror_script
