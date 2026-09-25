"""Campaign-specific technical and encoding guards."""

import runpy
from pathlib import Path
from unittest.mock import patch

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
