"""Original niche-video production, independent of third-party clipping campaigns.

Usage: python scripts/niche_video.py --config niches/ai-workflows.yaml --topic "..."
All external calls are explicit; generated videos remain unpublished drafts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

SCRIPT_MODEL = "gpt-4.1-mini"
VOICE_MODEL = "gpt-4o-mini-tts"
SCRIPT_CONTRACT = "niche-scenes-v1"
VOICES = frozenset(
    {
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "fable",
        "onyx",
        "nova",
        "sage",
        "shimmer",
        "verse",
        "marin",
        "cedar",
    }
)


class NicheVideoError(RuntimeError):
    """A validation or production step failed closed."""


def _text(value: object, field: str, limit: int = 500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise NicheVideoError(f"{field} must be nonempty text of at most {limit} characters")
    return value.strip()


@dataclass(frozen=True)
class NicheConfig:
    niche: str
    audience: str
    brand: str
    voice: str
    facts: tuple[str, ...]
    topic: str

    @classmethod
    def load(cls, path: Path, topic_override: str | None = None) -> NicheConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise NicheVideoError("niche config must be an object")
        allowed = {"niche", "audience", "brand", "voice", "facts", "topic"}
        if set(raw) - allowed:
            raise NicheVideoError(
                f"unsupported niche configuration keys: {sorted(set(raw) - allowed)}"
            )
        facts = raw.get("facts", [])
        if not isinstance(facts, list) or len(facts) > 12:
            raise NicheVideoError("facts must be a list of at most twelve verified statements")
        if not all(isinstance(fact, str) and 1 <= len(fact.strip()) <= 400 for fact in facts):
            raise NicheVideoError("every fact must be a nonempty string of at most 400 characters")
        voice = _text(raw.get("voice", "marin"), "voice", 30).lower()
        if voice not in VOICES:
            raise NicheVideoError("unsupported OpenAI TTS voice")
        return cls(
            niche=_text(raw.get("niche"), "niche", 150),
            audience=_text(raw.get("audience"), "audience", 200),
            brand=_text(raw.get("brand"), "brand", 60),
            voice=voice,
            facts=tuple(fact.strip() for fact in facts),
            topic=_text(topic_override or raw.get("topic"), "topic", 200),
        )


@dataclass(frozen=True)
class Scene:
    headline: str
    narration: str


@dataclass(frozen=True)
class Episode:
    title: str
    scenes: tuple[Scene, ...]

    @classmethod
    def from_json(cls, raw: object) -> Episode:
        if not isinstance(raw, dict) or set(raw) != {"title", "scenes"}:
            raise NicheVideoError("episode schema mismatch")
        scenes_raw = raw["scenes"]
        if not isinstance(scenes_raw, list) or not 3 <= len(scenes_raw) <= 5:
            raise NicheVideoError("episode requires 3-5 distinct scenes")
        scenes: list[Scene] = []
        for item in scenes_raw:
            if not isinstance(item, dict) or set(item) != {"headline", "narration"}:
                raise NicheVideoError("scene schema mismatch")
            headline = _text(item["headline"], "headline", 65)
            narration = _text(item["narration"], "narration", 400)
            if not 6 <= len(narration.split()) <= 45:
                raise NicheVideoError("scene narration must contain 6-45 words")
            scenes.append(Scene(headline=headline, narration=narration))
        if not 45 <= sum(len(scene.narration.split()) for scene in scenes) <= 125:
            raise NicheVideoError("episode narration must contain 45-125 words")
        return cls(title=_text(raw["title"], "title", 100), scenes=tuple(scenes))

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "scenes": [
                {"headline": scene.headline, "narration": scene.narration} for scene in self.scenes
            ],
        }


def _fingerprint(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)


def _openai_post(path: str, payload: dict[str, object], *, audio: bool = False) -> bytes:
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise NicheVideoError("OPENAI_API_KEY is required for paid inference")
    request = urllib.request.Request(
        f"https://api.openai.com/v1/{path}",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "audio/mpeg" if audio else "application/json",
        },
        method="POST",
    )
    try:
        # The URL is always prefixed with the fixed HTTPS OpenAI API host.
        with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
            result = response.read()
    except urllib.error.HTTPError as exc:
        raise NicheVideoError(f"OpenAI {path} returned HTTP {exc.code}") from exc
    except (OSError, TimeoutError) as exc:
        raise NicheVideoError(f"OpenAI {path} request failed") from exc
    if not result:
        raise NicheVideoError(f"OpenAI {path} returned an empty response")
    return result


def _script_request(config: NicheConfig) -> dict[str, object]:
    scene_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "headline": {"type": "string"},
            "narration": {"type": "string"},
        },
        "required": ["headline", "narration"],
        "additionalProperties": False,
    }
    return {
        "model": SCRIPT_MODEL,
        "temperature": 0.4,
        "max_completion_tokens": 1100,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Create one original, concise, useful vertical-video episode. "
                    "Return only the requested JSON. Write a surprising but honest first "
                    "line, substantive middle and a resolved ending. 3-5 scenes, 45-125 "
                    "narration words total, 6-45 words per scene. Never include invented "
                    "statistics, quotations, product promises, or news. Treat the supplied "
                    "facts as the only approved source for concrete factual claims. "
                    "If no facts are supplied, teach general methods without unverifiable "
                    "claims. Avoid filler, emojis, hashtags, and calls for engagement."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "niche": config.niche,
                        "audience": config.audience,
                        "topic": config.topic,
                        "approved_facts": config.facts,
                    },
                    ensure_ascii=False,
                ),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "niche_episode",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "scenes": {"type": "array", "items": scene_schema},
                    },
                    "required": ["title", "scenes"],
                    "additionalProperties": False,
                },
            },
        },
    }


def generate_episode(config: NicheConfig, cache_root: Path) -> tuple[Episode, str]:
    request = _script_request(config)
    digest = _fingerprint({"contract": SCRIPT_CONTRACT, "request": request})
    path = cache_root / "scripts" / f"{digest}.json"
    if path.is_file():
        return Episode.from_json(json.loads(path.read_text(encoding="utf-8"))), digest
    response = json.loads(_openai_post("chat/completions", request))
    choices = response.get("choices") or []
    if not choices or choices[0].get("finish_reason") != "stop":
        raise NicheVideoError("script generation did not finish cleanly")
    message = choices[0].get("message") or {}
    if message.get("refusal") or not isinstance(message.get("content"), str):
        raise NicheVideoError("script generation returned a refusal or no content")
    episode = Episode.from_json(json.loads(message["content"]))
    _write_atomic(path, json.dumps(episode.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"))
    return episode, digest


def synthesize_scene(scene: Scene, config: NicheConfig, cache_root: Path) -> Path:
    request: dict[str, object] = {
        "model": VOICE_MODEL,
        "voice": config.voice,
        "input": scene.narration,
        "instructions": "Warm, energetic, clear natural narration. No exaggerated announcer voice.",
        "response_format": "mp3",
    }
    digest = _fingerprint(request)
    path = cache_root / "tts" / f"{digest}.mp3"
    if not path.is_file():
        audio = _openai_post("audio/speech", request, audio=True)
        if len(audio) < 512:
            raise NicheVideoError("TTS result is too small to be credible audio")
        _write_atomic(path, audio)
    return path


def _process(command: list[str], *, timeout: int = 240) -> str:
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise NicheVideoError(f"media tool invocation failed: {command[0]}") from exc
    if result.returncode:
        raise NicheVideoError(
            f"media tool {command[0]} failed ({result.returncode}): {result.stderr[-900:]}"
        )
    return result.stdout


def _probe(path: Path) -> dict[str, Any]:
    return json.loads(
        _process(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration:stream=codec_type,codec_name,width,height",
                "-of",
                "json",
                str(path),
            ]
        )
    )


def _duration(path: Path) -> float:
    seconds = float(_probe(path).get("format", {}).get("duration", 0))
    if not math.isfinite(seconds) or not 1.0 <= seconds <= 90.0:
        raise NicheVideoError("TTS duration is outside accepted bounds")
    return seconds


def _render_scene(scene: Scene, audio: Path, target: Path, brand: str) -> float:
    duration = math.ceil((_duration(audio) + 0.20) * 30) / 30.0
    headline_path = target.with_suffix(".headline.txt")
    brand_path = target.with_suffix(".brand.txt")
    headline_path.write_text(scene.headline, encoding="utf-8")
    brand_path.write_text(brand.upper(), encoding="utf-8")
    # All text goes through drawtext textfile=, never through a shell or text= expression.
    filters = (
        "drawbox=x=72:y=315:w=936:h=10:color=0x55dbbf:t=fill,"
        f"drawtext=font='DejaVu Sans':textfile='{brand_path.as_posix()}':"
        "fontcolor=0x55dbbf:fontsize=46:x=80:y=215:expansion=none,"
        f"drawtext=font='DejaVu Sans':textfile='{headline_path.as_posix()}':"
        "fontcolor=white:fontsize=64:x=80:y=470:expansion=none,"
        "drawbox=x=72:y=1540:w=936:h=4:color=0x55dbbf:t=fill"
    )
    _process(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"color=c=0x0d172c:s=1080x1920:r=30:d={duration:.3f}",
            "-i",
            str(audio),
            "-vf",
            filters,
            "-af",
            "apad",
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
            "-r",
            "30",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            str(target),
        ]
    )
    return duration


def _verify_video(path: Path, expected: float) -> dict[str, object]:
    probe = _probe(path)
    streams = probe.get("streams", [])
    video = [item for item in streams if item.get("codec_type") == "video"]
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    if len(video) != 1 or len(audio) != 1:
        raise NicheVideoError("video must have exactly one video and one audio stream")
    if (
        video[0].get("codec_name") != "h264"
        or (video[0].get("width"), video[0].get("height")) != (1080, 1920)
        or audio[0].get("codec_name") != "aac"
    ):
        raise NicheVideoError("media codecs or 9:16 geometry mismatch")
    duration = float(probe.get("format", {}).get("duration", 0))
    if abs(duration - expected) > 0.9:
        raise NicheVideoError("output duration differs from the authored timeline")
    _process(["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"], timeout=360)
    return {"status": "PASS", "duration_seconds": duration, "full_decode": True}


def _concat(clips: list[Path], output: Path) -> None:
    # Resolve filenames into the same work directory; no user-provided concat directives.
    listing = output.parent / "concat.txt"
    listing.write_text(
        "".join(f"file '{path.name}'\n" for path in clips),
        encoding="utf-8",
    )
    _process(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(listing),
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ]
    )


def produce(config: NicheConfig, output_root: Path) -> Path:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise NicheVideoError("ffmpeg and ffprobe are required")
    cache_root = output_root / "_cache"
    episode, script_hash = generate_episode(config, cache_root)
    run_dir = output_root / f"draft-{script_hash[:16]}"
    run_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []
    durations: list[float] = []
    for i, scene in enumerate(episode.scenes):
        audio = synthesize_scene(scene, config, cache_root)
        clip = run_dir / f"scene-{i:02d}.mp4"
        durations.append(_render_scene(scene, audio, clip, config.brand))
        clips.append(clip)
    draft = run_dir / "draft.mp4"
    _concat(clips, draft)
    qc = _verify_video(draft, sum(durations))
    manifest = {
        "status": "DRAFT_REQUIRES_HUMAN_REVIEW",
        "published": False,
        "source_revision": os.getenv("GITHUB_SHA", "untracked"),
        "niche": config.niche,
        "topic": config.topic,
        "title": episode.title,
        "approved_facts": list(config.facts),
        "script_fingerprint": script_hash,
        "script_model": SCRIPT_MODEL,
        "tts_model": VOICE_MODEL,
        "voice": config.voice,
        "disclosure": "Includes AI-generated narration",
        "editorial_review": "PENDING",
        "visual_review": "PENDING",
        "audio_listening_review": "PENDING",
        "qc": qc,
        "video_sha256": hashlib.sha256(draft.read_bytes()).hexdigest(),
        "episode": episode.to_dict(),
    }
    _write_atomic(
        run_dir / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    return draft


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Original niche video -> TTS -> draft MP4")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--topic", default=None)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/niche"))
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args(argv)
    try:
        config = NicheConfig.load(arguments.config, arguments.topic)
        if arguments.validate_only:
            print(json.dumps({"valid": True, "niche": config.niche, "topic": config.topic}))
            return 0
        print(f"Draft generated: {produce(config, arguments.output_root)}")
        return 0
    except (NicheVideoError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"niche-video failed: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
