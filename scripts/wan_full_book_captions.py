"""Original-source, word-timed ASS captions for a validated 30-60s book episode.

Uses open-weight Whisper transcription on GitHub CPU. Text must reconcile with
the original verified Qwen narration; no synthetic timing from word counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from difflib import SequenceMatcher
from math import isfinite
from pathlib import Path
from typing import Any

MODEL = "small.en"
MIN_SIMILARITY = 0.77
SCHEMA = "clipper-kokoro-whisper-caption-v1"


def normalized(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", value.lower())


def ass_time(seconds: float) -> str:
    centis = round(max(0, seconds) * 100)
    hours, rem = divmod(centis, 360_000)
    minutes, rem = divmod(rem, 6000)
    secs, rem = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{rem:02d}"


def escape_ass(value: str) -> str:
    return value.replace("\\", r"\\").replace("{", "(").replace("}", ")").replace("\n", " ")


def make_ass(words: list[dict[str, Any]], duration: float) -> str:
    if not 30 <= duration <= 60 or len(words) < 45:
        raise ValueError("caption timeline lacks a full valid episode")
    for i, word in enumerate(words):
        if not 0 <= word["start"] < word["end"] <= duration + 0.1:
            raise ValueError("invalid word timing")
        if i and word["start"] < words[i - 1]["start"]:
            raise ValueError("word timestamps are out of order")
    header = (
        "[Script Info]\n"
        "Title: Clipper original book lesson\n"
        "ScriptType: v4.00+\n"
        "PlayResX: 1080\n"
        "PlayResY: 1920\n"
        "WrapStyle: 2\n\n"
        "[V4+ Styles]\n"
        "Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,"
        "BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,"
        "BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding\n"
        "Style: Words,DejaVu Sans,65,&H00FFFFFF,&H000000FF,&H65000000,"
        "&HB0000000,1,0,0,0,100,100,0,0,3,2,0,2,85,85,230,1\n"
        "Style: Book,DejaVu Sans,42,&H00FFFFFF,&H000000FF,&H80000000,"
        "&H70000000,1,0,0,0,100,100,0,0,3,1,0,8,100,100,125,1\n\n"
        "[Events]\n"
        "Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text\n"
    )
    lines = ["Dialogue: 0,0:00:00.00,0:00:03.00,Book,,0,0,0,,ATOMIC HABITS  /  JAMES CLEAR"]
    for start in range(0, len(words), 5):
        phrase = words[start : start + 5]
        for j, word in enumerate(phrase):
            index = start + j
            nxt = words[index + 1]["start"] if index + 1 < len(words) else duration
            end = min(duration, max(word["end"], nxt))
            if end <= word["start"]:
                end = min(duration, word["start"] + 0.08)
            text = []
            for k, member in enumerate(phrase):
                fragment = escape_ass(member["word"].strip())
                if k == j:
                    text.append(r"{\c&H009EFF&\b1}" + fragment + r"{\rWords}")
                else:
                    text.append(fragment)
            lines.append(
                f"Dialogue: 0,{ass_time(word['start'])},{ass_time(end)},"
                "Words,,0,0,0,," + " ".join(text)
            )
    return header + "\n".join(lines) + "\n"


def verified_whisper_word_timestamps(
    segments: Any, duration: float
) -> tuple[list[dict[str, Any]], int]:
    """Repair only subsecond alignment jitter, not arbitrary missing speech."""
    words: list[dict[str, Any]] = []
    adjustments = 0
    for segment in segments:
        for item in segment.words or ():
            if item.start is None or item.end is None or not normalized(item.word):
                continue
            raw_start = float(item.start)
            raw_end = float(item.end)
            if (
                not isfinite(raw_start)
                or not isfinite(raw_end)
                or raw_start < -0.25
                or raw_end < -0.25
                or raw_start > duration + 0.25
                or raw_end > duration + 0.25
                or raw_end < raw_start - 0.25
            ):
                raise ValueError("Whisper word timing differs substantially from source narration")
            start = min(max(0.0, raw_start), duration - 0.06)
            if words and start < words[-1]["start"]:
                if words[-1]["start"] - start > 0.25:
                    raise ValueError("Whisper timestamps materially out of order")
                start = words[-1]["start"]
            end = min(duration, max(start + 0.06, raw_end))
            if end <= start:
                raise ValueError("Whisper timestamp cannot fit source duration")
            if abs(start - raw_start) > 0.01 or abs(end - raw_end) > 0.01:
                adjustments += 1
            words.append(
                {
                    "word": item.word.strip(),
                    "start": round(start, 3),
                    "end": round(end, 3),
                }
            )
    if words and adjustments > max(4, len(words) // 6):
        raise ValueError("too many Whisper timing corrections; manual audio review required")
    return words, adjustments


def generate_captions(audio: Path, plan: Path, output: Path) -> None:
    from faster_whisper import WhisperModel

    episode = json.loads(plan.read_text())
    expected = " ".join(shot["narration"] for shot in episode["shots"])
    duration = float(episode["seconds"])
    if not 30 <= duration <= 60:
        raise ValueError("bad production narration length")
    actual_hash = hashlib.sha256(audio.read_bytes()).hexdigest()
    if actual_hash != episode["full_audio_sha256"]:
        raise ValueError("cached Kokoro source hash changed")
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(
        str(audio),
        language="en",
        beam_size=4,
        word_timestamps=True,
        vad_filter=False,
        initial_prompt=expected,
        condition_on_previous_text=False,
    )
    words, timing_adjustments = verified_whisper_word_timestamps(segments, duration)
    recognized = " ".join(w["word"] for w in words)
    match = SequenceMatcher(None, normalized(expected), normalized(recognized)).ratio()
    if match < MIN_SIMILARITY:
        raise ValueError(f"recognized speech disagrees with original Qwen script: {match:.3f}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(make_ass(words, duration))
    qa = {
        "schema": SCHEMA,
        "model": MODEL,
        "audio_sha256": actual_hash,
        "source_episode_fingerprint": episode["fingerprint"],
        "word_count": len(words),
        "timing_adjustments": timing_adjustments,
        "word_agreement": round(match, 4),
        "subtitle_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "creative_review": "PENDING",
        "publish_eligible": False,
    }
    (output.parent / "caption-evidence.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(f"WHISPER_CAPTIONS_ALIGNED words={len(words)} agreement={match:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    generate_captions(args.audio, args.plan, args.output)


if __name__ == "__main__":
    main()
