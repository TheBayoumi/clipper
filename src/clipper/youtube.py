from __future__ import annotations

import json
import os
import shutil
import subprocess
import urllib.parse
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .models import CampaignBrief, VideoCandidate


class YouTubeError(RuntimeError):
    """Raised for YouTube discovery or media acquisition failures."""


def _json_object(text: str) -> dict[str, Any]:
    value: object = json.loads(text)
    if not isinstance(value, dict):
        raise YouTubeError("expected a JSON object from YouTube")
    if not all(isinstance(key, str) for key in value):
        raise YouTubeError("YouTube JSON object contains a non-string key")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _object_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    objects: list[dict[str, Any]] = []
    for item in value:
        if isinstance(item, dict) and all(isinstance(key, str) for key in item):
            objects.append({key: child for key, child in item.items() if isinstance(key, str)})
    return objects


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}


def _run(command: Sequence[str], *, timeout: int = 900) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise YouTubeError(f"required executable not found: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise YouTubeError(f"command failed: {' '.join(command[:3])}: {detail[-1200:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise YouTubeError(f"command timed out after {timeout}s: {command[0]}") from exc


class YouTubeClient:
    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")

    def discover(self, brief: CampaignBrief) -> list[VideoCandidate]:
        if self.api_key:
            return self._discover_api(brief)
        return self._discover_ytdlp(brief)

    def _api_get(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        if self.api_key is None:
            raise YouTubeError("YouTube API key is required for API discovery")
        query = urllib.parse.urlencode({**params, "key": self.api_key})
        request = urllib.request.Request(
            f"https://www.googleapis.com/youtube/v3/{endpoint}?{query}",
            headers={"User-Agent": "clipper/0.1"},
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
                return _json_object(response.read().decode("utf-8"))
        except Exception as exc:
            raise YouTubeError(f"YouTube API request failed for {endpoint}: {exc}") from exc

    def _discover_api(self, brief: CampaignBrief) -> list[VideoCandidate]:
        ids: list[str] = list(brief.allowed_video_ids)
        per_channel = max(1, min(50, brief.source_limit))
        for channel_id in brief.source_channel_ids:
            params: dict[str, Any] = {
                "part": "snippet",
                "q": brief.search_query,
                "type": "video",
                "order": "relevance",
                "maxResults": per_channel,
                "regionCode": brief.region_code,
                "relevanceLanguage": brief.language,
                "channelId": channel_id,
            }
            if brief.published_after:
                params["publishedAfter"] = brief.published_after
            payload = self._api_get("search", params)
            for item in _object_list(payload.get("items")):
                identifier = _object(item.get("id")).get("videoId")
                if identifier:
                    ids.append(str(identifier))
        ids = list(dict.fromkeys(ids))[: brief.source_limit]
        if not ids:
            return []
        details = self._api_get(
            "videos",
            {"part": "snippet,statistics,contentDetails", "id": ",".join(ids)},
        )
        return [self._candidate_from_api(item) for item in _object_list(details.get("items"))]

    @staticmethod
    def _candidate_from_api(item: dict[str, Any]) -> VideoCandidate:
        snippet = _object(item.get("snippet"))
        statistics = _object(item.get("statistics"))
        video_id = str(item.get("id", ""))
        published_at = snippet.get("publishedAt")
        return VideoCandidate(
            video_id=video_id,
            title=str(snippet.get("title", "")),
            channel_id=str(snippet.get("channelId", "")),
            channel_title=str(snippet.get("channelTitle", "")),
            url=f"https://www.youtube.com/watch?v={video_id}",
            description=str(snippet.get("description", "")),
            published_at=str(published_at) if published_at else None,
            view_count=int(statistics["viewCount"]) if statistics.get("viewCount") else None,
        )

    @staticmethod
    def _candidate_from_ytdlp(entry: dict[str, Any]) -> VideoCandidate:
        video_id = str(entry.get("id", ""))
        return VideoCandidate(
            video_id=video_id,
            title=str(entry.get("title", "")),
            channel_id=str(entry.get("channel_id", "")),
            channel_title=str(entry.get("channel", "")),
            url=str(entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"),
            description=str(entry.get("description") or ""),
            published_at=str(entry.get("upload_date") or "") or None,
            duration_seconds=float(entry["duration"]) if entry.get("duration") else None,
            view_count=int(entry["view_count"]) if entry.get("view_count") else None,
        )

    def _discover_ytdlp(self, brief: CampaignBrief) -> list[VideoCandidate]:
        if not shutil.which("yt-dlp"):
            raise YouTubeError("YOUTUBE_API_KEY is unset and yt-dlp is not installed")
        results: list[VideoCandidate] = []
        for video_id in brief.allowed_video_ids[: brief.source_limit]:
            command = [
                "yt-dlp",
                "--dump-single-json",
                "--skip-download",
                "--no-warnings",
                f"https://www.youtube.com/watch?v={video_id}",
            ]
            entry = _json_object(_run(command, timeout=180).stdout)
            results.append(self._candidate_from_ytdlp(entry))

        remaining = brief.source_limit - len(results)
        if remaining > 0 and brief.source_channel_ids:
            command = [
                "yt-dlp",
                "--dump-single-json",
                "--skip-download",
                "--no-warnings",
                f"ytsearch{remaining}:{brief.search_query}",
            ]
            payload = _json_object(_run(command, timeout=180).stdout)
            results.extend(
                self._candidate_from_ytdlp(entry)
                for entry in _object_list(payload.get("entries"))
            )
        return list({item.video_id: item for item in results}.values())[: brief.source_limit]

    def download_subtitles(
        self,
        video: VideoCandidate,
        work_dir: Path,
        language: str,
    ) -> Path | None:
        work_dir.mkdir(parents=True, exist_ok=True)
        template = str(work_dir / f"{video.video_id}.%(ext)s")
        command = [
            "yt-dlp",
            "--skip-download",
            "--write-subs",
            "--write-auto-subs",
            "--sub-format",
            "vtt",
            "--sub-langs",
            f"{language}.*,en.*",
            "--no-warnings",
            "-o",
            template,
            video.url,
        ]
        try:
            _run(command, timeout=180)
        except YouTubeError:
            return None
        candidates = sorted(work_dir.glob(f"{video.video_id}*.vtt"))
        return candidates[0] if candidates else None

    def download_media(self, video: VideoCandidate, work_dir: Path) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        output = work_dir / f"{video.video_id}.mp4"
        if output.is_file() and output.stat().st_size > 0:
            return output
        command = [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "-f",
            "bv*[height<=1080]+ba/b[height<=1080]",
            "--merge-output-format",
            "mp4",
            "-o",
            str(output),
            video.url,
        ]
        _run(command)
        if not output.is_file() or output.stat().st_size == 0:
            raise YouTubeError(f"yt-dlp completed without creating {output}")
        return output
