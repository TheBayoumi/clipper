"""Original niche-video production, independent of third-party clipping campaigns.

Usage: python scripts/niche_video.py --config niches/ai-workflows.yaml --topic "..."
All external calls are explicit; generated videos remain unpublished drafts.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
import shutil
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Pin the script weights; keep TTS identity versioned to invalidate cache on upgrades.
SCRIPT_MODEL = "Qwen/Qwen3-1.7B"
SCRIPT_REVISION = "8d4744f9e13072f4920c326350fa81eedb74eae9"
VOICE_MODEL = "hexgrad/Kokoro-82M"
VOICE_REVISION = "v1.0"
SCRIPT_CONTRACT = "niche-scenes-open-v3"
MAX_SCRIPT_ATTEMPTS = 3
VOICE_CONTRACT = "kokoro-24khz-wav-v1"
VOICES = frozenset({"af_heart", "af_bella", "am_adam", "am_michael", "bf_emma"})


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
        voice = _text(raw.get("voice", "af_heart"), "voice", 30).lower()
        if voice not in VOICES:
            raise NicheVideoError("unsupported Kokoro voice")
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


def _script_request(config: NicheConfig) -> dict[str, object]:
    return {
        "model": SCRIPT_MODEL,
        "revision": SCRIPT_REVISION,
        "seed": 42,
        "max_new_tokens": 960,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Write exactly one valid JSON object, no markdown or commentary. "
                    'Schema: {"title": string, "scenes": '
                    '[{"headline": string, "narration": string}]}. '
                     "Write four distinct scenes, aiming for 80-100 narration words "
                    "in total, around 20-25 words each. Hard limits: 3-5 scenes, "
                    "45-125 total words, 6-45 per scene, headline <= 65 chars. "
                    "Start with a clear honest hook, develop useful original "
                    "ideas, finish with a complete payoff. Never invent statistics, "
                    "quotes, or news. Concrete claims may use approved_facts only. "
                    "With no facts, use general reproducible methods instead."
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
    }


@lru_cache(maxsize=1)
def _script_pipeline() -> tuple[Any, Any]:
    """Load Qwen3 weights only once, even when a revision is required."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise NicheVideoError(
            "Install the [niche-models] runtime for open-weight inference"
        ) from exc
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    tokenizer = AutoTokenizer.from_pretrained(SCRIPT_MODEL, revision=SCRIPT_REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        SCRIPT_MODEL, revision=SCRIPT_REVISION, dtype="auto"
    ).eval()
    return model, tokenizer


def _local_script_completion(request: dict[str, object]) -> str:
    """Generate locally; keep the weights resident for bounded repair attempts."""
    import torch

    model, tokenizer = _script_pipeline()
    torch.manual_seed(int(request["seed"]))
    inputs = tokenizer.apply_chat_template(
        request["messages"],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(request["max_new_tokens"]),
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            pad_token_id=tokenizer.eos_token_id,
        )
    output_ids = output[0][inputs["input_ids"].shape[-1] :]
    result = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
    if result.startswith(chr(96) * 3):
        result = "\n".join(result.splitlines()[1:-1]).strip()
    return result


def _script_feedback(raw: object, error: Exception) -> str:
    """Give the local model measured word counts, not only a generic error."""
    message = str(error)
    if isinstance(raw, dict) and isinstance(raw.get("scenes"), list):
        counts = [
            len(scene["narration"].split())
            if isinstance(scene, dict) and isinstance(scene.get("narration"), str)
            else None
            for scene in raw["scenes"]
        ]
        total = sum(count for count in counts if count is not None)
        message += f"; total_narration_words={total}; scene_words={counts}"
    return message[:900]


def _revision_request(
    original: dict[str, object], previous: str, error: str
) -> dict[str, object]:
    messages = original["messages"]
    if not isinstance(messages, list):
        raise NicheVideoError("internal request lost original script context")
    return {
        **original,
        "messages": [
            *messages,
            {"role": "assistant", "content": previous[:6000]},
            {
                "role": "user",
                "content": (
                    "Revise your previous JSON to resolve this exact violation: "
                    f"{error}. Preserve the original niche, topic, approved facts, "
                    "honest hook and complete payoff. Do not invent claims. "
                    "Return JSON only; aim for four scenes of 20-25 narration "
                    "words each, totalling 80-100 narration words."
                ),
            },
        ],
    }


def generate_episode(config: NicheConfig, cache_root: Path) -> tuple[Episode, str]:
    original = _script_request(config)
    digest = _fingerprint({"contract": SCRIPT_CONTRACT, "request": original})
    path = cache_root / "scripts" / f"{digest}.json"
    if path.is_file():
        return Episode.from_json(json.loads(path.read_text(encoding="utf-8"))), digest

    request = original
    audit_path = cache_root / "script_diagnostics" / f"{digest}.json"
    attempts: list[dict[str, object]] = []
    for attempt in range(1, MAX_SCRIPT_ATTEMPTS + 1):
        response = _local_script_completion(request)
        raw: object = None
        try:
            if not response:
                raise NicheVideoError("local script model returned no content")
            raw = json.loads(response)
            episode = Episode.from_json(raw)
        except (json.JSONDecodeError, NicheVideoError) as exc:
            error = _script_feedback(raw, exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "accepted": False,
                    "feedback": error,
                    "completion": response[:8000],
                }
            )
            _write_atomic(
                audit_path,
                json.dumps(attempts, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            print(
                f"Qwen3 validation attempt {attempt}/{MAX_SCRIPT_ATTEMPTS}: {error}",
                file=sys.stderr,
            )
            if attempt == MAX_SCRIPT_ATTEMPTS:
                raise NicheVideoError(
                    f"script rejected after {MAX_SCRIPT_ATTEMPTS} attempts: {error}"
                ) from exc
            request = _revision_request(original, response, error)
            continue
        attempts.append({"attempt": attempt, "accepted": True})
        _write_atomic(
            audit_path,
            json.dumps(attempts, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        _write_atomic(
            path,
            json.dumps(episode.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
        )
        print(f"Qwen3 script accepted on attempt {attempt}/{MAX_SCRIPT_ATTEMPTS}")
        return episode, digest

    raise NicheVideoError("bounded script generation unexpectedly exhausted")


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
