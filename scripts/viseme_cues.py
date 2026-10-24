"""Validated Rhubarb Lip Sync mouth cues for real Kokoro voice recordings.

Maps acoustic recognizer mouth shapes to jaw-opening amplitudes. A jaw bone
alone is not a complete lip rig; label this as timed jaw animation, not perfect
phoneme/viseme lips. No audio generated and no AI inference in this module.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

JAW_OPENING_RADIANS = {
    "X": 0.0, "A": 0.0, "B": 0.055, "C": 0.16, "D": 0.28,
    "E": 0.095, "F": 0.055, "G": 0.045, "H": 0.07,
}


class MouthCueError(ValueError):
    """Audio-to-animation timestamps violated the acceptance contract."""


@dataclass(frozen=True)
class MouthCue:
    start: float
    end: float
    shape: str

    @property
    def angle(self) -> float:
        return JAW_OPENING_RADIANS[self.shape]


def read_cues(path: Path, *, max_duration: float = 60.0) -> tuple[MouthCue, ...]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise MouthCueError("Rhubarb output must be a JSON object")
    raw = data.get("mouthCues")
    if not isinstance(raw, list) or len(raw) < 3:
        raise MouthCueError("recognizer supplied too few mouth cues")
    parsed: list[MouthCue] = []
    last_end = 0.0
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise MouthCueError("mouth cue entry is not an object")
        try:
            cue = MouthCue(float(item["start"]), float(item["end"]), str(item["value"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise MouthCueError("invalid mouth cue fields") from exc
        if cue.shape not in JAW_OPENING_RADIANS:
            raise MouthCueError(f"unrecognized mouth cue: {cue.shape}")
        if cue.start < 0 or cue.end <= cue.start or cue.end > max_duration + 0.1:
            raise MouthCueError(f"invalid cue interval at {index}")
        if abs(cue.start - last_end) > 0.035:
            raise MouthCueError("recognizer timestamps contain a gap or overlap")
        parsed.append(cue)
        last_end = cue.end
    if len({cue.shape for cue in parsed}) < 3:
        raise MouthCueError("insufficient distinct mouth shapes to justify lip motion")
    if max(cue.angle for cue in parsed) < 0.09:
        raise MouthCueError("speech contains no clearly visible vowel opening")
    return tuple(parsed)


def angle_at(cues: tuple[MouthCue, ...], time_s: float) -> float:
    if time_s < 0:
        return 0.0
    for cue in cues:
        if cue.start <= time_s < cue.end:
            return cue.angle
    return 0.0


def frame_angles(
    cues: tuple[MouthCue, ...], *, fps: int, frames: int
) -> tuple[float, ...]:
    if fps < 1 or frames < 2:
        raise MouthCueError("invalid animation frame rate or frame count")
    return tuple(angle_at(cues, n / fps) for n in range(frames))
