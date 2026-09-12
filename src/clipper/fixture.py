from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import CampaignBrief, VideoCandidate


class FixtureError(RuntimeError):
    """Raised when a private live-acceptance source fixture is invalid or incomplete."""


@dataclass(frozen=True, slots=True)
class SpanMedia:
    path: Path
    source_origin: float
    source_end: float
    sha256: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FixtureError(f"invalid fixture manifest: {path}") from exc
    if not isinstance(payload, dict):
        raise FixtureError("fixture manifest must be a JSON object")
    return {str(key): value for key, value in payload.items()}


class FixtureSourceClient:
    def __init__(self, fixture_dir: str | Path) -> None:
        self.root = Path(fixture_dir).resolve()
        self.manifest = _object(self.root / "fixture.json")
        video = self.manifest.get("video")
        spans = self.manifest.get("spans")
        if not isinstance(video, dict) or not isinstance(spans, list):
            raise FixtureError("fixture manifest requires video and spans")
        self.video = VideoCandidate(
            video_id=str(video.get("video_id") or ""),
            title=str(video.get("title") or ""),
            channel_id=str(video.get("channel_id") or ""),
            channel_title=str(video.get("channel_title") or ""),
            url=str(video.get("url") or ""),
            duration_seconds=float(video["duration_seconds"])
            if video.get("duration_seconds")
            else None,
        )
        if not self.video.video_id or not self.video.channel_id or not self.video.url:
            raise FixtureError("fixture video identity is incomplete")
        self.spans = [item for item in spans if isinstance(item, dict)]
        if not self.spans:
            raise FixtureError("fixture contains no source spans")
        self._verify_file("transcript", self.manifest.get("transcript"))
        full_media = self.manifest.get("full_media")
        if full_media is not None:
            self._verify_file("full_media", full_media)
        watermark = self.manifest.get("watermark")
        if watermark is not None:
            self._verify_file("watermark", watermark)
        for index, item in enumerate(self.spans):
            self._verify_file(f"span[{index}]", item)

    def _resolve(self, relative: object) -> Path:
        if not isinstance(relative, str) or not relative.strip():
            raise FixtureError("fixture file path is missing")
        path = (self.root / relative).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise FixtureError("fixture path escapes fixture directory") from exc
        return path

    def _verify_file(self, label: str, entry: object) -> Path:
        if not isinstance(entry, dict):
            raise FixtureError(f"fixture {label} entry is invalid")
        path = self._resolve(entry.get("file"))
        if not path.is_file() or path.stat().st_size <= 0:
            raise FixtureError(f"fixture {label} file is missing: {path}")
        expected = str(entry.get("sha256") or "")
        if not expected or _sha256(path) != expected:
            raise FixtureError(f"fixture {label} checksum mismatch")
        return path

    def discover(self, brief: CampaignBrief) -> list[VideoCandidate]:
        if self.video.video_id not in brief.allowed_video_ids:
            raise FixtureError("fixture video is not authorized by the campaign")
        if self.video.channel_id not in brief.source_channel_ids:
            raise FixtureError("fixture channel is not authorized by the campaign")
        return [self.video]

    def download_subtitles(
        self, video: VideoCandidate, work_dir: Path, language: str
    ) -> Path | None:
        del language
        if video.video_id != self.video.video_id:
            raise FixtureError("fixture subtitle request targets the wrong video")
        return self._verify_file("transcript", self.manifest["transcript"])

    def download_media(self, video: VideoCandidate, work_dir: Path) -> Path:
        del work_dir
        if video.video_id != self.video.video_id:
            raise FixtureError("fixture media request targets the wrong video")
        entry = self.manifest.get("full_media")
        if not isinstance(entry, dict):
            raise FixtureError("fixture source has no full media for open grounding")
        return self._verify_file("full_media", entry)

    def _full_media_span(self, start: float, end: float) -> SpanMedia | None:
        entry = self.manifest.get("full_media")
        if not isinstance(entry, dict):
            return None
        source_end = self.video.duration_seconds
        if source_end is None:
            source_end = max(float(item.get("source_end") or 0.0) for item in self.spans)
        if start < -1e-6 or source_end < end - 1e-6:
            return None
        path = self._verify_file("full_media", entry)
        return SpanMedia(path, 0.0, source_end, str(entry["sha256"]))

    def download_media_span(
        self, video: VideoCandidate, start: float, end: float, work_dir: Path
    ) -> SpanMedia:
        del work_dir
        if video.video_id != self.video.video_id:
            raise FixtureError("fixture media request targets the wrong video")
        master = self._full_media_span(start, end)
        if master is not None:
            return master
        covering: list[tuple[float, dict[str, Any]]] = []
        for item in self.spans:
            origin = float(item.get("source_origin") or 0.0)
            source_end = float(item.get("source_end") or 0.0)
            if origin <= start + 1e-6 and source_end >= end - 1e-6:
                covering.append((source_end - origin, item))
        if not covering:
            raise FixtureError(f"fixture has no source span covering {start:.3f}-{end:.3f}")
        _, item = min(covering, key=lambda pair: pair[0])
        path = self._verify_file("span", item)
        return SpanMedia(
            path=path,
            source_origin=float(item["source_origin"]),
            source_end=float(item["source_end"]),
            sha256=str(item["sha256"]),
        )

    def campaign_watermark(self, brief: CampaignBrief) -> Path | None:
        entry = self.manifest.get("watermark")
        if not brief.watermark_url:
            return None
        if not isinstance(entry, dict):
            raise FixtureError("campaign requires watermark but fixture does not provide it")
        expected_url = str(entry.get("source_url") or "")
        if expected_url and expected_url != brief.watermark_url:
            raise FixtureError("fixture watermark does not match campaign watermark URL")
        return self._verify_file("watermark", entry)
