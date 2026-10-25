"""Source-locked 30-60s Atomic Habits book episode preparation and MP4 assembly.

The previous original Qwen3/Kokoro *draft* supplies immutable source narration.
All visual shots must be rendered independently by Wan; the archived Pip video
is NEVER passed through as production footage. Technical review != publication.
"""

from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import subprocess
import wave
from pathlib import Path

FPS = 24
RATE = 24000
RENDERED_SHOT_NAMES = ("hook", "lesson_example", "book_intro", "hook")
SCHEMA = "clipper-full-book-episode-v1"
EXPECTED_BOOK = "Atomic Habits"
PROMPTS = (
    "Cinematic stylized 3D young adult host with curly dark hair, black rectangular "
    "glasses and charcoal hoodie, in a warmly lit lived-in study. He speaks the "
    "narration and glances from an overwhelming smartphone to an easy first step "
    "on his desk. Genuine hand, brow and lip animation, consistent set geography.",
    "Same exact host and charcoal hoodie; change to a clean close-up of a desk "
    "with notebook in clear view and a phone placed out of reach. Host points "
    "to a simple visible cue while speaking. Believable hand contact, gentle "
    "motivated movement, cinematic amber light and realistic depth of field.",
    "Same exact animated host in the warm bookshelf study, wearing the charcoal "
    "hoodie and rectangular glasses. He lifts a neutral unbranded book, looks "
    "pleased at completing a small task, and speaks naturally. Genuine physical "
    "book interaction, readable face, no invented printed words.",
    "Same exact character, camera now framed medium-wide, returns to his desk "
    "to open his notebook and actually write the first line. He smiles with "
    "relief as he speaks the conclusion. End on an intentional finished gesture, "
    "not a random cut. Consistent warm studio, physically plausible hands.",
)


class EpisodeError(ValueError):
    """Source or full-video delivery violates contract."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def command_run(command: list[str], timeout: int = 180) -> str:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=True)
    return result.stdout


def inspect_audio(path: Path) -> tuple[int, bytes]:
    with wave.open(str(path), "rb") as wav:
        if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
            raise EpisodeError("Kokoro source must decode to PCM16 mono 24kHz")
        return wav.getnframes(), wav.readframes(wav.getnframes())


def silence_spans(wav_bytes: bytes, seconds: float, max_rms: float = 0.014) -> list[tuple[float, float]]:
    """Find long low-energy pauses; never assume proportional word cuts."""
    samples = array.array("h")
    samples.frombytes(wav_bytes)
    chunk = RATE // 50  # 20 ms windows
    quiet: list[bool] = []
    for start in range(0, len(samples), chunk):
        part = samples[start : start + chunk]
        if not part:
            break
        energy = math.sqrt(sum(v * v for v in part) / len(part)) / 32768
        quiet.append(energy < max_rms)
    spans: list[tuple[float, float]] = []
    index = 0
    while index < len(quiet):
        if not quiet[index]:
            index += 1
            continue
        stop = index + 1
        while stop < len(quiet) and quiet[stop]:
            stop += 1
        first, last = index / 50, stop / 50
        if last - first >= 0.12 and first > 2 and last < seconds - 2:
            spans.append((first, last))
        index = stop
    return spans


def select_cuts(audio_bytes: bytes, narrations: list[str], duration: float) -> list[float]:
    """Use acoustic pause candidates near semantic scene text-length targets."""
    total_words = sum(len(x.split()) for x in narrations)
    if total_words < 45:
        raise EpisodeError("insufficient original narration for full episode")
    spans = silence_spans(audio_bytes, duration)
    if len(spans) < len(narrations) - 1:
        raise EpisodeError("no adequate source narration pauses; do not cut mid-speech")
    expected: list[float] = []
    running = 0
    for sentence in narrations[:-1]:
        running += len(sentence.split())
        expected.append(duration * running / total_words)
    cuts: list[float] = []
    last = 0.0
    for i, goal in enumerate(expected):
        remaining = len(expected) - i
        options = [
            (abs((a + b) / 2 - goal), (a + b) / 2)
            for a, b in spans
            if last + 3 <= (a + b) / 2 <= duration - (remaining * 3)
        ]
        if not options:
            raise EpisodeError(f"scene {i} has no safe acoustic boundary")
        score, cut = min(options)
        if score > 3.7:
            raise EpisodeError(f"scene {i} boundary too far from semantic break: {score:.2f}s")
        cuts.append(cut)
        last = cut
    segments = list(zip([0.0, *cuts], [*cuts, duration], strict=True))
    if any(not 2.5 <= stop - start < 20 for start, stop in segments):
        raise EpisodeError("scene segment unsupported for Wan S2V (<20 seconds)")
    return cuts


def make_prepare(source_video: Path, manifest_file: Path, output: Path) -> None:
    manifest = json.loads(manifest_file.read_text())
    if manifest.get("book", {}).get("title") != EXPECTED_BOOK:
        raise EpisodeError("source must be original Atomic Habits Qwen3/Kokoro episode")
    if manifest.get("tts_model") != "hexgrad/Kokoro-82M":
        raise EpisodeError("source narration has wrong TTS model")
    scenes = manifest.get("episode", {}).get("scenes")
    if not isinstance(scenes, list) or len(scenes) != 4:
        raise EpisodeError("expected four original authored source scenes")
    narrations = [s["narration"] for s in scenes]
    output.mkdir(parents=True, exist_ok=True)
    full_audio = output / "full-kokoro.wav"
    command_run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_video),
        "-map", "0:a:0", "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le",
        str(full_audio),
    ])
    samples, data = inspect_audio(full_audio)
    duration = samples / RATE
    if not 30 <= duration <= 60:
        raise EpisodeError(f"cached audio duration {duration:.3f}s, not 30-60s")
    cuts = select_cuts(data, narrations, duration)
    borders = [0, *[round(t * RATE) for t in cuts], samples]
    shots = []
    for index, scene in enumerate(scenes):
        clip = output / f"scene-{index:02d}.wav"
        first, last = borders[index : index + 2]
        with wave.open(str(clip), "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(RATE)
            wav.writeframes(data[first * 2 : last * 2])
        name = RENDERED_SHOT_NAMES[index]
        shots.append({
            "id": f"scene-{index:02d}",
            "keyframe_name": name,
            "narration": scene["narration"],
            "headline": scene["headline"],
            "audio": clip.name,
            "audio_sha256": digest(clip.read_bytes()),
            "duration_seconds": round((last - first) / RATE, 6),
            "start_seconds": round(first / RATE, 6),
            "end_seconds": round(last / RATE, 6),
            "prompt": PROMPTS[index],
            "seed": 735 + index,
            "renderer": "wan2.2-s2v-14b",
            "creative_review": "PENDING",
            "publish_eligible": False,
        })
    payload = {
        "schema": SCHEMA,
        "source_manifest_sha256": digest(manifest_file.read_bytes()),
        "source_video_sha256": digest(source_video.read_bytes()),
        "full_audio_sha256": digest(full_audio.read_bytes()),
        "book": manifest["book"],
        "title": manifest["episode"]["title"],
        "source_script_fingerprint": manifest["script_fingerprint"],
        "seconds": round(duration, 6),
        "shots": shots,
        "status": "SCRIPT_AUDIO_LOCKED_AWAITING_FOUR_WAN_SHOTS",
        "publish_eligible": False,
    }
    payload["fingerprint"] = digest(json.dumps(payload, sort_keys=True).encode())
    (output / "plan.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"EPISODE_AUDIO_LOCKED duration={duration:.3f}s cuts={cuts} shots={len(shots)}")


def prepare_concat(plan: dict, paths: list[Path], output: Path) -> None:
    """Produce professional vertical 24fps picture, exact original Kokoro narration.

    Audio is not regenerated or retimed, each shot is required and hash-checked.
    This remains an editorial-review candidate, NOT automatically publishable.
    """
    from wan_shared_shot_cache import verified_wan_output

    output.mkdir(parents=True, exist_ok=True)
    if plan.get("schema") != SCHEMA or not 30 <= plan["seconds"] <= 60:
        raise EpisodeError("not a valid 30-60 second source-backed episode")
    clips = []
    for shot, folder in zip(plan["shots"], paths, strict=True):
        name = shot["id"]
        video = folder / "wan-s2v-candidate.mp4"
        manifest = folder / "model-evidence.json"
        ref = Path("artifacts/full-book/frames") / (shot["keyframe_name"] + ".png")
        voice = Path("artifacts/full-book/source") / shot["audio"]
        if not verified_wan_output(
            video, manifest, digest(ref.read_bytes()), digest(voice.read_bytes()),
            shot["prompt"], shot["seed"],
        ):
            raise EpisodeError(f"invalid or missing Wan shot {name}")
        probe = json.loads(command_run([
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "json", str(video),
        ]))
        actual = float(probe["format"]["duration"])
        wanted = shot["duration_seconds"]
        if abs(actual - wanted) > 1.0:
            raise EpisodeError(f"{name} video duration diverges from narration by >1s")
        clips.append(video)
    inputs = []
    filters = []
    for i, (video, shot) in enumerate(zip(clips, plan["shots"], strict=True)):
        inputs += ["-i", str(video)]
        frames = round(shot["duration_seconds"] * FPS)
        filters.append(
            f"[{i}:v]fps=24,scale=1080:1920:flags=lanczos,setsar=1,"
            f"trim=end_frame={frames},setpts=PTS-STARTPTS[v{i}]"
        )
    audio = Path("artifacts/full-book/source/full-kokoro.wav")
    inputs += ["-i", str(audio)]
    labels = "".join(f"[v{i}]" for i in range(len(clips)))
    filters.append(f"{labels}concat=n={len(clips)}:v=1:a=0[vout]")
    target = output / "atomic-habits-full-candidate.mp4"
    command_run([
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs,
        "-filter_complex", ";".join(filters), "-map", "[vout]",
        "-map", f"{len(clips)}:a:0", "-c:v", "libx264",
        "-preset", "medium", "-crf", "18", "-r", str(FPS),
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
        "-shortest", str(target),
    ], timeout=1200)
    decoded = json.loads(command_run([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,width,height,r_frame_rate",
        "-of", "json", str(target),
    ]))
    duration = float(decoded["format"]["duration"])
    visual = [item for item in decoded["streams"] if item["codec_type"] == "video"]
    sound = [item for item in decoded["streams"] if item["codec_type"] == "audio"]
    if not 30 <= duration <= 60 or len(visual) != 1 or len(sound) != 1:
        raise EpisodeError("assembled video lacks production-length audio/video")
    if (visual[0]["width"], visual[0]["height"], visual[0]["r_frame_rate"]) != (
        1080, 1920, "24/1"
    ):
        raise EpisodeError("final video geometry is not 1080x1920 24fps")
    command_run(["ffmpeg", "-v", "error", "-i", str(target), "-f", "null", "-"], timeout=200)
    qa = {
        "schema": "clipper-full-book-delivery-evidence-v1",
        "source_plan_sha256": plan["fingerprint"],
        "video_sha256": digest(target.read_bytes()),
        "duration_seconds": duration,
        "width": 1080,
        "height": 1920,
        "fps": FPS,
        "source_audio_sha256": digest(audio.read_bytes()),
        "shots": [{"id": s["id"], "seed": s["seed"]} for s in plan["shots"]],
        "status": "COMPLETE_MP4_TECHNICAL_CANDIDATE_NEEDS_CREATIVE_REVIEW",
        "word_level_captions": "NOT_YET_IMPLEMENTED",
        "creative_review": "PENDING",
        "publish_eligible": False,
    }
    (output / "full-book-evidence.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(f"FULL_BOOK_MP4_TECHNICALLY_VALID duration={duration:.3f}s sha256={qa['video_sha256']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="operation", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--video", type=Path, required=True)
    prep.add_argument("--manifest", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    assembly = sub.add_parser("assemble")
    assembly.add_argument("--plan", type=Path, required=True)
    assembly.add_argument("--shots", type=Path, nargs=4, required=True)
    assembly.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "prepare":
        make_prepare(args.video, args.manifest, args.output)
    else:
        prepare_concat(json.loads(args.plan.read_text()), args.shots, args.output)


if __name__ == "__main__":
    main()
