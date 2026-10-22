from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from clipper.models import VideoCandidate
from clipper.pipeline import _source_media


def test_authoritative_source_client_wins_over_campaign_media_url(tmp_path: Path) -> None:
    canonical = tmp_path / "canonical-source.mkv"
    canonical.write_bytes(b"sha-verified-canonical-source")
    source = Mock()
    source.download_media.return_value = canonical
    brief = SimpleNamespace(
        source_media_urls={"v1": "https://drive.google.com/file/d/supplemental/view"}
    )
    video = VideoCandidate(
        video_id="v1",
        title="explicit target",
        channel_id="UC1",
        channel_title="authorized channel",
        url="https://www.youtube.com/watch?v=v1",
    )

    with patch("clipper.pipeline._download_asset") as supplemental_download:
        result = _source_media(
            brief,
            source,
            video,
            tmp_path,
            source_is_authoritative=True,
        )

    assert result == canonical
    source.download_media.assert_called_once_with(video, tmp_path)
    supplemental_download.assert_not_called()


def test_run_pipeline_marks_supplied_source_client_authoritative() -> None:
    source = Path("src/clipper/pipeline.py").read_text(encoding="utf-8")
    assert "source_is_authoritative = source_client is not None" in source
    assert "source_is_authoritative=source_is_authoritative" in source
