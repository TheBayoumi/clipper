"""Original niche-video CLI; script stage delegates to clipper.books.scripting.

The legacy TTS and prototype renderer remain unchanged, outside the script stage.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import wave
from functools import lru_cache
from pathlib import Path
from typing import Any

from clipper.books import scripting as _scripting

NicheVideoError = _scripting.NicheVideoError
NicheConfig = _scripting.NicheConfig
Scene = _scripting.Scene
Episode = _scripting.Episode
SCRIPT_MODEL = _scripting.SCRIPT_MODEL
SCRIPT_REVISION = _scripting.SCRIPT_REVISION
VOICE_MODEL = _scripting.VOICE_MODEL
VOICE_REVISION = _scripting.VOICE_REVISION
SCRIPT_CONTRACT = _scripting.SCRIPT_CONTRACT
BOOK_EDIT_CONTRACT = _scripting.BOOK_EDIT_CONTRACT
MAX_SCRIPT_ATTEMPTS = _scripting.MAX_SCRIPT_ATTEMPTS
VOICE_CONTRACT = _scripting.VOICE_CONTRACT
VOICES = _scripting.VOICES
_text = _scripting._text
_fingerprint = _scripting._fingerprint
_write_atomic = _scripting._write_atomic
_script_request = _scripting._script_request
_script_pipeline = _scripting._script_pipeline
_local_script_completion = _scripting._local_script_completion
_script_feedback = _scripting._script_feedback
_revision_request = _scripting._revision_request
_compact_book_sentences = _scripting._compact_book_sentences
_script_fingerprint = _scripting._script_fingerprint
_recover_book_script = _scripting._recover_book_script


def generate_episode(config: NicheConfig, cache_root: Path) -> tuple[Episode, str]:
    """Preserve legacy model monkeypatch boundaries and exact cache key semantics."""
    return _scripting.generate_episode(
        config, cache_root, completion=_local_script_completion
    )


@lru_cache(maxsize=1)
def _kokoro_pipeline() -> Any:
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise NicheVideoError("Install the [niche-models] runtime for Kokoro TTS") from exc
    return KPipeline(lang_code="a", repo_id=VOICE_MODEL)


def _wav_metadata(payload: bytes) -> tuple[int, int, int]:
    try:
        with wave.open(io.BytesIO(payload), "rb") as wav:
            return wav.getnchannels(), wav.getframerate(), wav.getnframes()
    except (wave.Error, EOFError) as exc:
        raise NicheVideoError("Kokoro produced invalid WAV audio") from exc


def _kokoro_wav(narration: str, voice: str) -> bytes:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise NicheVideoError("Install soundfile and numpy for open-source TTS") from exc
    chunks = [
        np.asarray(audio, dtype=np.float32).reshape(-1)
        for _, _, audio in _kokoro_pipeline()(narration, voice=voice, speed=1.0)
    ]
    if not chunks:
        raise NicheVideoError("Kokoro returned no audio segments")
    speech = np.concatenate(chunks)
    if len(speech) < 8000 or not np.isfinite(speech).all():
        raise NicheVideoError("Kokoro returned empty or non-finite audio")
    buffer = io.BytesIO()
    sf.write(buffer, np.clip(speech, -1.0, 1.0), 24000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def synthesize_scene(scene: Scene, config: NicheConfig, cache_root: Path) -> Path:
    request = {
        "contract": VOICE_CONTRACT,
        "model": VOICE_MODEL,
        "model_revision": VOICE_REVISION,
        "voice": config.voice,
        "language": "en-us",
        "sample_rate": 24000,
        "speed": 1.0,
        "text": scene.narration,
    }
    digest = _fingerprint(request)
    path = cache_root / "tts" / f"{digest}.wav"
    if path.is_file():
        nch, rate, frames = _wav_metadata(path.read_bytes())
        if nch == 1 and rate == 24000 and frames >= 8000:
            return path
        raise NicheVideoError("Cached TTS WAV has invalid geometry or duration")
    audio = _kokoro_wav(scene.narration, config.voice)
    nch, rate, frames = _wav_metadata(audio)
    if nch != 1 or rate != 24000 or frames < 8000:
        raise NicheVideoError("Kokoro WAV must be mono 24 kHz with audible duration")
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


@lru_cache(maxsize=1)
def _book_artwork_creator() -> Any:
    """Load a sibling artwork module without relying on cwd or sys.path."""
    import importlib.util

    source = Path(__file__).with_name("book_cartoon.py")
    spec = importlib.util.spec_from_file_location("clipper_book_cartoon", source)
    if spec is None or spec.loader is None:
        raise NicheVideoError("book cartoon artwork module could not be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_assets


def _render_scene(
    scene: Scene,
    audio: Path,
    target: Path,
    brand: str,
    config: NicheConfig | None = None,
    scene_index: int = 0,
    total_scenes: int = 1,
) -> float:
    duration = math.ceil((_duration(audio) + 0.20) * 30) / 30.0
    if config is not None and config.format == "book_lesson":
        bg, talking, idle = _book_artwork_creator()(
            target.parent,
            config.brand,
            config.book_title,
            config.book_author,
            scene.headline,
            scene_index,
            total_scenes,
        )
        # Rhythmic speaking-pose animation, not phoneme-accurate lip sync.
        motion = "x='(W-w)/2+9*sin(t*1.4)':y='735+10*sin(t*2.2)'"
        filters = (
            rf"[0:v][2:v]overlay={motion}:enable='lt(mod(t\,0.44)\,0.23)'[a];"
            rf"[a][3:v]overlay={motion}:enable='gte(mod(t\,0.44)\,0.23)'[v]"
        )
        _process(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-loop",
                "1",
                "-framerate",
                "30",
                "-i",
                str(bg),
                "-i",
                str(audio),
                "-loop",
                "1",
                "-framerate",
                "30",
                "-i",
                str(talking),
                "-loop",
                "1",
                "-framerate",
                "30",
                "-i",
                str(idle),
                "-filter_complex",
                filters,
                "-map",
                "[v]",
                "-map",
                "1:a",
                "-af",
                "apad",
                "-t",
                f"{duration:.3f}",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-r",
                "30",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(target),
            ],
            timeout=360,
        )
        return duration
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


def produce(
    config: NicheConfig, output_root: Path, *, allow_prototype_visuals: bool = False
) -> Path:
    if config.format == "book_lesson" and not allow_prototype_visuals:
        raise NicheVideoError(
            "Book visual production is creatively BLOCKED: the two-pose Pillow "
            "cartoon and title-card layout were rejected in editorial review. "
            "Use --allow-prototype-visuals only for explicit archived technical "
            "reproduction, never as publication-quality acceptance."
        )
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
        durations.append(
            _render_scene(scene, audio, clip, config.brand, config, i, len(episode.scenes))
        )
        clips.append(clip)
    draft = run_dir / "draft.mp4"
    _concat(clips, draft)
    qc = _verify_video(draft, sum(durations))
    manifest = {
        "status": "DRAFT_REQUIRES_HUMAN_REVIEW",
        "published": False,
        "publish_eligible": False,
        "creative_review_status": (
            "REJECTED_PROTOTYPE" if config.format == "book_lesson" else "NOT_REVIEWED"
        ),
        "source_revision": os.getenv("GITHUB_SHA", "untracked"),
        "niche": config.niche,
        "topic": config.topic,
        "format": config.format,
        "pillar": config.pillar,
        "book": {
            "title": config.book_title,
            "author": config.book_author,
            "source_url": config.source_url,
        }
        if config.format == "book_lesson"
        else None,
        "character": "Pip" if config.format == "book_lesson" else None,
        "animation": (
            "pip-cartoon-v1-approximate-mouth-motion" if config.format == "book_lesson" else None
        ),
        "title": episode.title,
        "approved_facts": list(config.facts),
        "script_fingerprint": script_hash,
        "script_recovery": (
            json.loads(
                (cache_root / "script_recoveries" / f"{script_hash}.json").read_text(
                    encoding="utf-8"
                )
            )
            if (cache_root / "script_recoveries" / f"{script_hash}.json").is_file()
            else None
        ),
        "script_model": SCRIPT_MODEL,
        "script_model_revision": SCRIPT_REVISION,
        "tts_model": VOICE_MODEL,
        "tts_model_revision": VOICE_REVISION,
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
    parser.add_argument("--allow-prototype-visuals", action="store_true")
    parser.add_argument("--recover-audit", type=Path, default=None)
    parser.add_argument("--recover-run-id", default=None)
    arguments = parser.parse_args(argv)
    try:
        config = NicheConfig.load(arguments.config, arguments.topic)
        if arguments.validate_only:
            print(json.dumps({"valid": True, "niche": config.niche, "topic": config.topic}))
            return 0
        if (arguments.recover_audit is None) != (arguments.recover_run_id is None):
            raise NicheVideoError("--recover-audit requires --recover-run-id and vice versa")
        if arguments.recover_audit is not None:
            _recover_book_script(
                config,
                arguments.output_root / "_cache",
                arguments.recover_audit,
                arguments.recover_run_id,
            )
        draft = produce(
            config,
            arguments.output_root,
            allow_prototype_visuals=arguments.allow_prototype_visuals,
        )
        print(f"Technical prototype generated: {draft}")
        return 0
    except (NicheVideoError, OSError, ValueError, json.JSONDecodeError) as exc:
        parser.exit(1, f"niche-video failed: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
