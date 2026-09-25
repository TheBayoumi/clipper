"""Campaign-specific technical and encoding guards."""

import json
import runpy
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


@pytest.fixture
def campaign_brief(tmp_path: Path) -> Path:
    path = Path("campaigns/reach-tjr-weekly.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = tmp_path / "campaign.yaml"
    out.write_text(yaml.safe_dump(data), encoding="utf-8")
    return out


def test_campaign_accepts_verified_video_ids(campaign_brief: Path) -> None:
    data = check_campaign_brief(campaign_brief)
    assert data["allowed_video_ids"] == ["8PYgFVB0GHE"]
    assert data["watermark_url"] is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_channel_ids", ["UC-not-tjr"]),
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
    with pytest.raises(QualityError, match="confirm live campaign budget"):
        prepare_staged_brief(campaign_brief, output)
    monkeypatch.setenv("TJR_BUDGET_CONFIRMED", "true")
    monkeypatch.setenv("TJR_SOURCE_VERIFIED", "true")
    assert prepare_staged_brief(campaign_brief, output) == output
    parsed = check_campaign_brief(output)
    assert parsed["source_media_urls"] == {"8PYgFVB0GHE": url}
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
    hd = Mock(stdout=json.dumps({"streams": [
        {"codec_type": "video", "width": 1920, "height": 1080}
    ]}))
    sd = Mock(stdout=json.dumps({"streams": [
        {"codec_type": "video", "width": 640, "height": 480}
    ]}))
    with patch("subprocess.run", return_value=hd):
        assert probe_original(source) == {"width": 1920, "height": 1080}
    with patch("subprocess.run", return_value=sd), pytest.raises(
        QualityError, match="below 720p"
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
    monkeypatch.setenv("TJR_BUDGET_CONFIRMED", "true")
    monkeypatch.setenv("TJR_SOURCE_VERIFIED", "true")
    runtime = prepare_staged_brief(campaign_brief, tmp_path / "runtime.yaml")
    run = tmp_path / "artifacts" / "sample"
    original = run / "work" / "8PYgFVB0GHE" / "source.mp4"
    original.parent.mkdir(parents=True)
    original.write_bytes(b"this does not match the pinned SHA-256")
    (run / "manifest.json").write_text(
        json.dumps({
            "errors": [],
            "discovered_videos": [{"channel_id": "UCGHBUXjDCeiIXNdKR0HUZnA"}],
            "planned_clips": [{"video_id": "8PYgFVB0GHE"}],
            "rendered_clips": [{"video_id": "8PYgFVB0GHE"}],
        }),
        encoding="utf-8",
    )
    with pytest.raises(QualityError, match="hash differs"):
        validate_artifacts(runtime, tmp_path / "artifacts")
