from __future__ import annotations

from pathlib import Path

from .models import VideoCandidate
from .youtube import YouTubeClient


class PersistentYouTubeClient(YouTubeClient):
    """Reuse downloaded authorized source masters across independent pipeline runs."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_height: int | None = None,
        media_cache_root: Path,
    ) -> None:
        super().__init__(api_key, max_height=max_height)
        self.media_cache_root = Path(media_cache_root)

    def download_media(self, video: VideoCandidate, work_dir: Path) -> Path:
        del work_dir
        cache_dir = self.media_cache_root / video.video_id
        cache_dir.mkdir(parents=True, exist_ok=True)
        return super().download_media(video, cache_dir)
