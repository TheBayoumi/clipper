import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from clipper.transcript import load_vtt, parse_vtt, transcribe_with_faster_whisper


VTT = """WEBVTT

00:00:01.000 --> 00:00:03.500
<v Speaker>Hello &amp; welcome</v>

2
00:00:03,500 --> 00:00:05,000
This is a test.

00:00:05.000 --> 00:00:06.000
This is a test.

bad --> timestamp
ignored
"""


def test_parse_vtt_and_merge_duplicate_caption() -> None:
    segments = parse_vtt(VTT)
    assert len(segments) == 2
    assert segments[0].text == "Hello & welcome"
    assert segments[1].start == 3.5
    assert segments[1].end == 6.0


def test_load_vtt(tmp_path: Path) -> None:
    path = tmp_path / "x.vtt"
    path.write_text(VTT, encoding="utf-8")
    assert load_vtt(path)[0].start == 1.0


def test_parse_vtt_skips_empty_and_invalid_ranges() -> None:
    text = """WEBVTT

00:00:01.000 --> 00:00:01.000
bad

00:00:02.000 --> 00:00:03.000
<font></font>
"""
    assert parse_vtt(text) == []


def test_faster_whisper_adapter_filters_invalid_segments(tmp_path: Path) -> None:
    class FakeModel:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def transcribe(self, *_args, **_kwargs):
            return (
                [
                    SimpleNamespace(start=0, end=2, text=" hello "),
                    SimpleNamespace(start=2, end=2, text="bad"),
                    SimpleNamespace(start=3, end=4, text=" "),
                ],
                object(),
            )

    fake_module = SimpleNamespace(WhisperModel=FakeModel)
    with patch.dict(sys.modules, {"faster_whisper": fake_module}):
        segments = transcribe_with_faster_whisper(
            tmp_path / "audio.mp4",
            model_name="tiny",
            device="cpu",
            compute_type="int8",
            language="en",
        )
    assert [(s.start, s.end, s.text) for s in segments] == [(0.0, 2.0, "hello")]
