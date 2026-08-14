from pathlib import Path
from unittest.mock import patch

from clipper.models import VideoCandidate
from clipper.source_cache import PersistentYouTubeClient


def _video() -> VideoCandidate:
    return VideoCandidate("v1", "Title", "UC1", "Channel", "https://youtu.be/v1")


def test_persistent_youtube_client_reuses_cached_master(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    cached_dir = cache_root / "v1"
    cached_dir.mkdir(parents=True)
    cached_media = cached_dir / "v1.mkv"
    cached_media.write_bytes(b"cached-master")

    client = PersistentYouTubeClient(media_cache_root=cache_root)
    with patch("clipper.youtube.YouTubeClient._video_info") as video_info:
        result = client.download_media(_video(), tmp_path / "different-run" / "work" / "v1")

    assert result == cached_media
    video_info.assert_not_called()


def test_persistent_youtube_client_routes_downloads_to_shared_cache(tmp_path: Path) -> None:
    cache_root = tmp_path / "cache"
    expected_dir = cache_root / "v1"
    expected_media = expected_dir / "v1.mkv"
    client = PersistentYouTubeClient(media_cache_root=cache_root)

    def fake_download(_self, _video_candidate, work_dir):
        assert work_dir == expected_dir
        work_dir.mkdir(parents=True, exist_ok=True)
        expected_media.write_bytes(b"master")
        return expected_media

    with patch("clipper.source_cache.YouTubeClient.download_media", new=fake_download):
        result = client.download_media(_video(), tmp_path / "run" / "work" / "v1")

    assert result == expected_media
