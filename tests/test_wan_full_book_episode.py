"""30-60s original-source contract and semantic audio-boundary tests."""

from __future__ import annotations

import array
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from wan_full_book_episode import EpisodeError, select_cuts, silence_spans  # noqa: E402


def make_audio(scene_lengths: list[float], pauses: bool = True) -> bytes:
    pcm = array.array("h")
    for seconds in scene_lengths:
        count = round(seconds * 24000)
        for i in range(count):
            value = 0 if pauses and i < 6000 else round(9000 * math.sin(i * (math.pi / 40)))
            pcm.append(value)
    return pcm.tobytes()


def test_38_second_source_splits_only_at_speech_pauses() -> None:
    script = [" ".join([f"word{x}" for x in range(n)]) for n in (12, 13, 12, 13)]
    lengths = [9, 10, 9, 10]
    pcm = make_audio(lengths)
    spans = silence_spans(pcm, float(sum(lengths)))
    assert len(spans) >= 3
    cuts = select_cuts(pcm, script, float(sum(lengths)))
    assert len(cuts) == 3
    assert all(abs(cut - sum(lengths[:idx])) < 0.4 for idx, cut in enumerate(cuts, 1))


def test_must_refuse_mid_speech_scene_cut() -> None:
    script = ["Narration words repeated in a spoken paragraph about simple daily routine actions and better cues"] * 4
    pcm = make_audio([9, 10, 9, 10], pauses=False)
    with pytest.raises(EpisodeError, match="no adequate source narration pauses"):
        select_cuts(pcm, script, 38.0)


def test_silence_spans_skip_long_trailing_pause() -> None:
    pcm = make_audio([5, 5, 5, 5])
    spans = silence_spans(pcm, 20.0)
    assert all(first > 2.0 and last < 18.0 for first, last in spans)
