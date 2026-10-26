"""Source-backed Atomic Habits episode-two CPU preparation (no Modal inference).

Contract -> cached Kokoro baritone -> Whisper timestamps -> immutable dynamic EDL.
GPU render is a separate workflow job gated by these artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import wave
from pathlib import Path
from typing import Any

import book_ai_director as director
import book_audio_timeline as timeline
import book_episode_selection as chooser

MODEL = "hexgrad/Kokoro-82M"
REVISION = "v1.0"
VOICE = "am_adam"
SPEED = 0.98
PAUSE = 0.14
FPS = 24


class EpisodeTwoError(ValueError):
    """Refuse unsupported narration, mismatched evidence or guessed timestamps."""


def write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise EpisodeTwoError(f"expected a JSON object: {path}")
    return result


def build_contract(\n    brief: dict[str, Any], catalog: dict[str, Any]\n) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if (\n        brief.get("schema") != "clipper-episode-brief-v1"\n        or brief.get("publish_eligible") is not False\n    ):
        raise EpisodeTwoError("unsupported editorial brief or publication gate")
    selection = chooser.choose_lesson(
        catalog,
        book_id=brief["book_id"],
        lesson_id=brief["lesson_id"],
        angle_id=brief["angle_id"],
    )
    if selection["episode_id"] != brief.get("episode_id"):
        raise EpisodeTwoError("lesson identity is not catalog-verified")
    if selection["lesson"]["claim_source_url"] != brief["source_url"]:
        raise EpisodeTwoError("source reference is not authoritative")
    script = brief["script"]
    request = director.build_request(selection, script)
    specs = brief.get("shot_briefs")
    if not isinstance(specs, list):
        raise EpisodeTwoError("no AI director shot briefs")
    words = [w for section in request["narrative_sections"] for w in section["narration"].split()]
    shots: list[dict[str, Any]] = []
    cursor = 0
    for index, spec in enumerate(specs):
        section_index = spec["scene_index"]
        if (\n            type(section_index) is not int\n            or section_index not in range(len(request["narrative_sections"]))\n        ):
            raise EpisodeTwoError("unknown narrative section")
        section = request["narrative_sections"][section_index]
        if cursor < section["start_word"] or cursor >= section["end_word"]:
            raise EpisodeTwoError("shot briefs skip or duplicate narrative sections")
        split = spec.get("cut_after_words")
        if split is not None and (type(split) is not int or split < 3):
            raise EpisodeTwoError("split marker must be at least three words")
        end = cursor + split if split is not None else section["end_word"]
        if end > section["end_word"] or end <= cursor:
            raise EpisodeTwoError("director split is beyond narration section")
        role = spec["role"]
        frame = spec["reference_name"]
        if not isinstance(frame, str) or not re.fullmatch(r"[a-z_]+", frame):
            raise EpisodeTwoError("invalid verified character-reference name")
        shots.append(
            {
                "id": f"shot-{index:02d}",
                "role": role,
                "scene_index": section_index,
                "start_word": cursor,
                "end_word": end,
                "narration_text": " ".join(words[cursor:end]),
                "renderer": "s2v",
                "character_action": spec["character_action"],
                "camera": spec["camera"],
                "setting": spec["setting"],
                "emotion": spec["emotion"],
                "required_objects": spec["required_objects"],
                "source_evidence": request["approved_claims"] if section_index == 1 else [],
                "qa_criteria": spec["qa_criteria"],
            }
        )
        cursor = end
    candidate = {
        "schema": director.PLAN_SCHEMA,
        "request_fingerprint": request["fingerprint"],
        "shots": shots,
    }
    plan = director.validate_plan(request, candidate)
    return selection, request, plan


def contract(brief: Path, catalog: Path, target: Path) -> None:
    root = load(brief)
    selection, request, plan = build_contract(root, load(catalog))
    write(target / "selection.json", selection)
    write(target / "director-request.json", request)
    write(target / "director-plan.json", plan)
    write(target / "script.json", root["script"])
    write(target / "brief.json", root)
    print(json.dumps({
        "episode_id": selection["episode_id"],
        "shot_count": plan["shot_count"],
        "words": request["total_words"],
        "script_sha256": request["script_sha256"],
        "no_gpu": True,
    }))


def synthesize(target: Path, cache: Path) -> None:
    import numpy as np
    import soundfile as sf

    script = load(target / "script.json")
    request = load(target / "director-request.json")
    scenes = script["scenes"]
    if timeline.digest(script) != request["script_sha256"]:
        raise EpisodeTwoError("script was changed after director review")
    from wan_full_book_voice import make_wav, pcm

    cache.mkdir(parents=True, exist_ok=True)
    pipeline = None
    frames = bytearray()
    for index, scene in enumerate(scenes):
        text = scene["narration"]
        identity = {
            "text": text,
            "model": MODEL,
            "revision": REVISION,
            "voice_id": VOICE,
            "style": "original_warm_low_cinematic_baritone",
            "speed": SPEED,
        }
        key = timeline.digest(identity)
        cached = cache / f"{key}.wav"
        if cached.is_file():
            samples = pcm(cached.read_bytes())
            print(f"KOKORO_CACHE_VERIFIED scene={index}")
        else:
            if pipeline is None:
                from kokoro import KPipeline

                pipeline = KPipeline(lang_code="a", repo_id=MODEL)
            segments = [
                np.asarray(audio, dtype=np.float32)
                for _, _, audio in pipeline(text, voice=VOICE, speed=SPEED)
            ]
            if not segments:
                raise EpisodeTwoError("Kokoro produced no narration")
            arr = np.concatenate(segments)
            if arr.ndim != 1 or not np.isfinite(arr).all():
                raise EpisodeTwoError("Kokoro produced invalid samples")
            import io

            buffer = io.BytesIO()
            sf.write(buffer, np.clip(arr, -1, 1), timeline.RATE, format="WAV", subtype="PCM_16")
            samples = pcm(buffer.getvalue())
            cached.write_bytes(make_wav(samples))
            print(f"KOKORO_GENERATED scene={index}")
        frames.extend(samples)
        if index < len(scenes) - 1:
            frames.extend(b"\0\0" * round(PAUSE * timeline.RATE))
    audio_path = target / "full-kokoro.wav"
    audio_path.write_bytes(make_wav(bytes(frames)))
    duration_frames, wav_sha = timeline.inspect_audio(audio_path)
    words = [
        word for section in request["narrative_sections"] for word in section["narration"].split()
    ]
    voice = {
        "model": MODEL,
        "revision": REVISION,
        "voice_id": VOICE,
        "style": "original_warm_low_cinematic_baritone",
        "speed": SPEED,
        "text_sha256": timeline.digest(" ".join(words)),
        "wav_sha256": wav_sha,
    }
    write(target / "voice.json", voice)
    print(\n        json.dumps(\n            {"audio_locked": True, "seconds": duration_frames / timeline.RATE, "sha256": wav_sha}\n        )\n    )


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def align(target: Path) -> None:
    """Use real Whisper timestamps, never text-length or silence heuristics."""
    from faster_whisper import WhisperModel

    request = load(target / "director-request.json")
    audio_path = target / "full-kokoro.wav"
    frames, wav_sha = timeline.inspect_audio(audio_path)
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    script_words = [
        word for section in request["narrative_sections"] for word in section["narration"].split()
    ]
    segments, info = model.transcribe(
        str(audio_path),
        language="en",
        beam_size=4,
        word_timestamps=True,
        vad_filter=False,
        initial_prompt=" ".join(script_words),
        condition_on_previous_text=False,
    )
    recognized = [
        word for segment in segments for word in segment.words or ()
        if word.start is not None and word.end is not None and normalize(word.word)
    ]
    if len(recognized) != len(script_words):
        raise EpisodeTwoError(
            f"word alignment is not one-to-one: original={len(script_words)}, "
            f"transcribed={len(recognized)}, "\n            f"probability={getattr(info, 'language_probability', None)}"
        )
    stamps = []
    previous = 0
    for index, (expected, found) in enumerate(zip(script_words, recognized, strict=True)):
        if normalize(expected) != normalize(found.word):
            raise EpisodeTwoError(
                f"Whisper word mismatch at index {index}: script={expected!r}, "
                f"recognized={found.word!r}"
            )
        start = round(float(found.start) * timeline.RATE)
        end = round(float(found.end) * timeline.RATE)
        if previous > start or not 0 <= start < end <= frames:
            raise EpisodeTwoError(f"non-monotonic acoustic alignment at word {index}")
        stamps.append({"text": expected, "start_sample": start, "end_sample": end})
        previous = end
    output = {
        "audio_sha256": wav_sha,
        "script_sha256": request["script_sha256"],
        "aligner_model": "faster-whisper-small.en",
        "aligner_revision": "faster-whisper-1.2.1",
        "words": stamps,
    }
    write(target / "alignment.json", output)
    print(json.dumps({"whisper_word_matches": len(stamps), "strict": True}))


def seal(target: Path) -> None:
    request = load(target / "director-request.json")
    plan = load(target / "director-plan.json")
    audio = target / "full-kokoro.wav"
    edl = timeline.build_timeline(
        request,
        plan,
        audio,
        load(target / "voice.json"),
        load(target / "alignment.json"),
    )
    write(target / "audio-locked-edl.json", edl)
    with wave.open(str(audio), "rb") as recording:
        pcm = recording.readframes(recording.getnframes())
    briefs = load(target / "brief.json")["shot_briefs"]
    render_jobs: list[dict[str, Any]] = []
    for index, (shot, spec) in enumerate(zip(edl["shots"], briefs, strict=True)):
        start, end = shot["start_sample"], shot["end_sample"]
        shot_audio = target / f"shot-{index:02d}.wav"
        with wave.open(str(shot_audio), "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(timeline.RATE)
            recording.writeframes(pcm[start * 2 : end * 2])
        frame = spec["reference_name"] + ".png"
        prompt = (
            "Same exact stylized dark curly-haired host wearing black rectangular "
            "glasses and charcoal hoodie as reference image. Cinematic natural "
            "physical actions and synchronized narration; never change identity, "
            "outfit, or the warm studio layout. "
            + spec["character_action"]
            + ". Camera: " + spec["camera"]
            + ". Set: " + spec["setting"]
            + ". Emotion: " + spec["emotion"]
            + ". No added captions, floating text, overlays or new characters."
        )
        render_jobs.append({
            "id": shot["id"],
            "index": index,
            "frame": frame,
            "audio": shot_audio.name,
            "prompt": prompt,
            "seed": 1842 + index,
            "audio_sha256": timeline.file_sha(shot_audio),
            "duration_seconds": round((end - start) / timeline.RATE, 6),
        })
    write(target / "render-jobs.json", {"jobs": render_jobs, "edl_fingerprint": edl["fingerprint"]})
    print(json.dumps({
        "episode_id": edl["episode_id"],
        "shots": len(render_jobs),
        "source_audio_sha256": edl["audio"]["sha256"],
        "ready_for_cached_or_paid_visual_render": True,
    }))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["contract", "voice", "align", "seal"])
    parser.add_argument("--brief", type=Path, default=Path("niches/atomic-habits-episode-02.json"))
    parser.add_argument("--catalog", type=Path, default=chooser.CATALOG_PATH)
    parser.add_argument("--output", type=Path, default=Path("artifacts/episode-two/source"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/episode-two/voice-cache"))
    args = parser.parse_args()
    if args.mode == "contract":
        contract(args.brief, args.catalog, args.output)
    elif args.mode == "voice":
        synthesize(args.output, args.cache)
    elif args.mode == "align":
        align(args.output)
    else:
        seal(args.output)


if __name__ == "__main__":
    main()
