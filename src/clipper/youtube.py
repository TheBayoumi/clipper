from __future__ import annotations

import codecs
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import urllib.parse
import urllib.request
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import VideoCandidate


class YouTubeError(RuntimeError):
    """Raised for YouTube discovery or media acquisition failures."""


@dataclass(frozen=True, slots=True)
class DiscoveryRequest:
    """Input for the separate discovery workflow; never used by ``clipper run``."""

    query: str = ""
    channel_ids: tuple[str, ...] = ()
    limit: int = 10
    language: str = "en"
    region_code: str = "US"
    published_after: str | None = None
    video_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 1 <= self.limit <= 50:
            raise ValueError("discovery limit must be between 1 and 50")
        if not self.video_ids and not self.channel_ids:
            raise ValueError("discovery requires at least one channel_id or video_id")
        if self.channel_ids and not self.query.strip():
            raise ValueError("channel discovery requires a non-empty query")
        if not self.language.strip() or not self.region_code.strip():
            raise ValueError("discovery language and region_code cannot be empty")


def _youtube_video_id(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    video_id = ""
    if parsed.scheme == "https" and host == "youtu.be":
        video_id = parsed.path.strip("/").split("/", 1)[0]
    elif parsed.scheme == "https" and host in {"youtube.com", "m.youtube.com", "music.youtube.com"}:
        if parsed.path == "/watch":
            video_id = (urllib.parse.parse_qs(parsed.query).get("v") or [""])[0]
        else:
            for prefix in ("/shorts/", "/live/", "/embed/"):
                if parsed.path.startswith(prefix):
                    video_id = parsed.path[len(prefix) :].split("/", 1)[0]
                    break
    if not video_id:
        raise YouTubeError("authorized source URL must identify a YouTube video")
    return video_id


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_source_identity(
    video: VideoCandidate,
    info: dict[str, Any],
) -> dict[str, str]:
    requested_id = _youtube_video_id(video.url)
    actual_id = str(info.get("id") or "")
    actual_channel_id = str(info.get("channel_id") or "")
    canonical_url = str(info.get("webpage_url") or info.get("original_url") or "")
    if requested_id != video.video_id:
        raise YouTubeError(
            "authorized candidate URL video ID mismatch: "
            f"expected={video.video_id} url={requested_id}"
        )
    if actual_id != video.video_id:
        raise YouTubeError(
            "YouTube extractor video ID does not match authorized candidate: "
            f"expected={video.video_id} actual={actual_id}"
        )
    if actual_channel_id != video.channel_id:
        raise YouTubeError(
            "YouTube extractor channel ID does not match authorized candidate: "
            f"expected={video.channel_id} actual={actual_channel_id}"
        )
    if _youtube_video_id(canonical_url) != video.video_id:
        raise YouTubeError(
            "YouTube extractor canonical URL does not match authorized candidate"
        )
    return {
        "video_id": actual_id,
        "channel_id": actual_channel_id,
        "webpage_url": canonical_url,
    }


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


def _run_visible(command: Sequence[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    """Stream a subprocess verbatim while retaining a bounded tail for structured errors."""
    process = subprocess.Popen(
        list(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    reader = process.stdout
    if reader is None:  # pragma: no cover - defensive Popen contract guard
        process.kill()
        raise YouTubeError(f"unable to capture live output from {command[0]}")

    tail: deque[str] = deque(maxlen=96)

    def pump_output() -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        while True:
            chunk = reader.read(512)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if text:
                sys.stderr.write(text)
                sys.stderr.flush()
                tail.append(text)
        final = decoder.decode(b"", final=True)
        if final:
            sys.stderr.write(final)
            sys.stderr.flush()
            tail.append(final)

    pump = threading.Thread(target=pump_output, name="clipper-live-subprocess", daemon=True)
    pump.start()
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        pump.join(timeout=2)
        raise
    pump.join(timeout=2)
    detail = "".join(tail)[-12000:]
    if returncode:
        raise subprocess.CalledProcessError(
            returncode,
            list(command),
            output=detail,
            stderr=detail,
        )
    return subprocess.CompletedProcess(list(command), returncode, stdout=detail, stderr=detail)


def _run(
    command: Sequence[str], *, timeout: int = 900, visible: bool = False
) -> subprocess.CompletedProcess[str]:
    try:
        if visible:
            return _run_visible(command, timeout=timeout)
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


def _is_http_403(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "http error 403" in text or "403: forbidden" in text or "403 forbidden" in text


def _retry_player_clients() -> tuple[str, ...]:
    configured = os.getenv("CLIPPER_YTDLP_RETRY_CLIENTS", "android_vr,web_embedded")
    clients = tuple(item.strip() for item in configured.split(",") if item.strip())
    return clients or ("android_vr", "web_embedded")


class YouTubeClient:
    def __init__(self, api_key: str | None = None, *, max_height: int | None = None) -> None:
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        if max_height is not None and max_height < 360:
            raise YouTubeError("max_height must be at least 360 when configured")
        self.max_height = max_height

    def discover(self, request: DiscoveryRequest) -> list[VideoCandidate]:
        if self.api_key:
            return self._discover_api(request)
        return self._discover_ytdlp(request)

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

    def _discover_api(self, request: DiscoveryRequest) -> list[VideoCandidate]:
        ids: list[str] = list(request.video_ids)
        per_channel = max(1, min(50, request.limit))
        for channel_id in request.channel_ids:
            params: dict[str, Any] = {
                "part": "snippet",
                "q": request.query,
                "type": "video",
                "order": "relevance",
                "maxResults": per_channel,
                "regionCode": request.region_code,
                "relevanceLanguage": request.language,
                "channelId": channel_id,
            }
            if request.published_after:
                params["publishedAfter"] = request.published_after
            payload = self._api_get("search", params)
            for item in _object_list(payload.get("items")):
                identifier = _object(item.get("id")).get("videoId")
                if identifier:
                    ids.append(str(identifier))
        ids = list(dict.fromkeys(ids))[: request.limit]
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

    def _discover_ytdlp(self, request: DiscoveryRequest) -> list[VideoCandidate]:
        if not shutil.which("yt-dlp"):
            raise YouTubeError("YOUTUBE_API_KEY is unset and yt-dlp is not installed")
        results: list[VideoCandidate] = []
        for video_id in request.video_ids[: request.limit]:
            command = [
                "yt-dlp",
                "--dump-single-json",
                "--skip-download",
                "--no-warnings",
                f"https://www.youtube.com/watch?v={video_id}",
            ]
            entry = _json_object(_run(command, timeout=180).stdout)
            results.append(self._candidate_from_ytdlp(entry))

        remaining = request.limit - len(results)
        if remaining > 0 and request.channel_ids:
            command = [
                "yt-dlp",
                "--dump-single-json",
                "--skip-download",
                "--no-warnings",
                f"ytsearch{remaining}:{request.query}",
            ]
            payload = _json_object(_run(command, timeout=180).stdout)
            results.extend(
                self._candidate_from_ytdlp(entry) for entry in _object_list(payload.get("entries"))
            )
        return list({item.video_id: item for item in results}.values())[: request.limit]

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

    @staticmethod
    def _select_video_format(
        payload: dict[str, Any], max_height: int | None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        available: list[dict[str, Any]] = []
        for item in _object_list(payload.get("formats")):
            height = int(item.get("height") or 0)
            width = int(item.get("width") or 0)
            codec = str(item.get("vcodec") or "none")
            if height <= 0 or codec == "none":
                continue
            available.append(
                {
                    "format_id": str(item.get("format_id") or ""),
                    "width": width,
                    "height": height,
                    "fps": float(item.get("fps") or 0.0),
                    "codec": codec,
                    "audio_codec": str(item.get("acodec") or "none"),
                    "container": str(item.get("ext") or ""),
                    "bitrate_kbps": float(item.get("tbr") or 0.0),
                }
            )
        eligible = (
            available
            if max_height is None
            else [item for item in available if int(item["height"]) <= max_height]
        )
        if not eligible:
            limit = "available source quality" if max_height is None else f"{max_height}p"
            raise YouTubeError(f"no video format is available for {limit}")
        selected = max(
            eligible,
            key=lambda item: (
                int(item["height"]),
                int(item["width"]),
                float(item["fps"]),
                float(item["bitrate_kbps"]),
            ),
        )
        available.sort(
            key=lambda item: (
                int(item["height"]),
                int(item["width"]),
                float(item["fps"]),
                float(item["bitrate_kbps"]),
            ),
            reverse=True,
        )
        return selected, available

    @staticmethod
    def _same_quality_formats(
        available: Sequence[dict[str, Any]], selected: dict[str, Any]
    ) -> list[dict[str, Any]]:
        target_height = int(selected["height"])
        target_width = int(selected["width"])
        matching = [
            item
            for item in available
            if int(item["height"]) == target_height and int(item["width"]) == target_width
        ]
        selected_id = str(selected["format_id"])
        matching.sort(
            key=lambda item: (
                str(item["format_id"]) == selected_id,
                float(item["fps"]),
                float(item["bitrate_kbps"]),
            ),
            reverse=True,
        )
        return matching

    @staticmethod
    def _extractor_args(player_client: str | None) -> list[str]:
        if not player_client:
            return []
        return ["--extractor-args", f"youtube:player_client={player_client}"]

    def _video_info(self, url: str, player_client: str | None = None) -> dict[str, Any]:
        return _json_object(
            _run(
                [
                    "yt-dlp",
                    "--dump-single-json",
                    "--skip-download",
                    "--no-warnings",
                    *self._extractor_args(player_client),
                    url,
                ],
                timeout=180,
            ).stdout
        )

    @staticmethod
    def _format_selector(selected: dict[str, Any]) -> str:
        format_id = str(selected["format_id"])
        if selected["audio_codec"] != "none":
            return format_id
        return f"{format_id}+bestaudio/{format_id}"

    def _download_command(
        self,
        video: VideoCandidate,
        output: Path,
        selected: dict[str, Any],
        player_client: str | None = None,
    ) -> list[str]:
        return [
            "yt-dlp",
            "--no-playlist",
            "--no-warnings",
            "--retries",
            "3",
            "--fragment-retries",
            "3",
            *self._extractor_args(player_client),
            "-f",
            self._format_selector(selected),
            "--merge-output-format",
            "mkv",
            "--remux-video",
            "mkv",
            "-o",
            str(output),
            video.url,
        ]

    def download_media(self, video: VideoCandidate, work_dir: Path) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)
        output = work_dir / f"{video.video_id}.mkv"
        metadata_path = output.with_suffix(".source.json")
        if output.is_file() and output.stat().st_size > 0:
            if not metadata_path.is_file():
                raise YouTubeError("cached source is missing identity evidence")
            try:
                cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise YouTubeError("cached source identity evidence is invalid") from exc
            if not isinstance(cached_metadata, dict):
                raise YouTubeError("cached source identity evidence is invalid")
            identity = cached_metadata.get("canonical_identity")
            if not isinstance(identity, dict):
                raise YouTubeError("cached source has no canonical identity")
            _validated_source_identity(video, identity)
            expected_sha = str(cached_metadata.get("sha256") or "")
            if not expected_sha or _source_sha256(output) != expected_sha:
                raise YouTubeError("cached source media hash does not match identity evidence")
            return output

        initial_info = self._video_info(video.url)
        final_identity = _validated_source_identity(video, initial_info)
        initial_selected, initial_available = self._select_video_format(initial_info, None)
        attempts: list[dict[str, object]] = []
        final_selected = initial_selected
        final_available = initial_available
        final_client: str | None = None

        def attempt(selected: dict[str, Any], player_client: str | None) -> None:
            command = self._download_command(video, output, selected, player_client)
            label = player_client or "default"
            try:
                _run(command, timeout=7200, visible=True)
            except YouTubeError as exc:
                attempts.append(
                    {
                        "player_client": label,
                        "format_id": str(selected["format_id"]),
                        "height": int(selected["height"]),
                        "width": int(selected["width"]),
                        "status": "FAILED",
                        "error": str(exc)[-1200:],
                    }
                )
                raise
            attempts.append(
                {
                    "player_client": label,
                    "format_id": str(selected["format_id"]),
                    "height": int(selected["height"]),
                    "width": int(selected["width"]),
                    "status": "SUCCESS",
                }
            )

        try:
            attempt(initial_selected, None)
        except YouTubeError as first_error:
            if not _is_http_403(first_error):
                raise
            last_error: YouTubeError = first_error
            recovered = False
            for player_client in _retry_player_clients():
                try:
                    refreshed_info = self._video_info(video.url, player_client)
                    refreshed_identity = _validated_source_identity(video, refreshed_info)
                    _, refreshed_available = self._select_video_format(refreshed_info, None)
                except YouTubeError as refresh_error:
                    attempts.append(
                        {
                            "player_client": player_client,
                            "status": "METADATA_REFRESH_FAILED",
                            "error": str(refresh_error)[-1200:],
                        }
                    )
                    last_error = refresh_error
                    continue
                equivalent = self._same_quality_formats(refreshed_available, initial_selected)
                if not equivalent:
                    attempts.append(
                        {
                            "player_client": player_client,
                            "status": "NO_EQUIVALENT_QUALITY_FORMAT",
                            "required_height": int(initial_selected["height"]),
                            "required_width": int(initial_selected["width"]),
                        }
                    )
                    continue
                for candidate in equivalent[:2]:
                    try:
                        attempt(candidate, player_client)
                    except YouTubeError as retry_error:
                        last_error = retry_error
                        continue
                    final_selected = candidate
                    final_available = refreshed_available
                    final_client = player_client
                    final_identity = refreshed_identity
                    recovered = True
                    break
                if recovered:
                    break
            if not recovered:
                raise YouTubeError(
                    "YouTube media download exhausted same-quality HTTP 403 recovery attempts; "
                    "quality remained locked to "
                    f"{initial_selected['width']}x{initial_selected['height']}; "
                    f"last error: {last_error}"
                ) from last_error

        if not output.is_file() or output.stat().st_size == 0:
            raise YouTubeError(f"yt-dlp completed without creating {output}")
        metadata_path.write_text(
            json.dumps(
                {
                    "quality_policy": "highest_available_no_transcode",
                    "canonical_identity": final_identity,
                    "sha256": _source_sha256(output),
                    "legacy_requested_max_height_ignored": self.max_height,
                    "initial_selected": initial_selected,
                    "selected": final_selected,
                    "selected_player_client": final_client or "default",
                    "recovered_after_http_403": len(attempts) > 1,
                    "download_attempts": attempts,
                    "available_formats": final_available,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return output
