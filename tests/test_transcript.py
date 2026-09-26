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


def test_parse_youtube_rolling_vtt_drops_snapshots_and_repeated_leading_line() -> None:
    text = """WEBVTT
Kind: captions
Language: en

00:00:00.160 --> 00:00:02.310 align:start position:0%

Like<00:00:00.400><c> you</c><00:00:00.480><c> ever</c>

00:00:02.310 --> 00:00:02.320 align:start position:0%
Like you ever

00:00:02.320 --> 00:00:04.150 align:start position:0%
Like you ever
alive<00:00:02.720><c> surge</c><00:00:03.040><c> ticking</c>

00:00:04.150 --> 00:00:04.160 align:start position:0%
alive surge ticking

00:00:04.160 --> 00:00:05.750 align:start position:0%
alive surge ticking
damage<00:00:04.480><c> or</c><00:00:04.720><c> you're</c><00:00:04.960><c> gone.</c>
"""
    segments = parse_vtt(text)
    assert [segment.text for segment in segments] == [
        "Like you ever",
        "alive surge ticking",
        "damage or you're gone.",
    ]
    assert all(segment.duration > 0.05 for segment in segments)


def test_word_aligned_whisper_keeps_real_word_timestamps_and_fallback(tmp_path: Path) -> None:
    calls: list[object] = []
    words = [
        SimpleNamespace(start=i * 0.4, end=i * 0.4 + 0.3, word=token)
        for i, token in enumerate(
            [
                "Wait",
                " what",
                " happened?",
                " I",
                " just",
                " watched",
                " that",
                " candle",
                " reverse",
            ]
        )
    ]
    words.extend(
        [
            SimpleNamespace(start=None, end=2.0, word=" skipped"),
            SimpleNamespace(start=3.0, end=None, word=" skipped"),
            SimpleNamespace(start=4.0, end=3.0, word=" skipped"),
            SimpleNamespace(start=4.0, end=5.0, word=" "),
        ]
    )

    class FakeModel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def transcribe(self, *_args: object, **kwargs: object) -> tuple[object, object]:
            calls.append(kwargs)
            return (
                [
                    SimpleNamespace(start=0, end=4, text="Wait what happened?", words=words),
                    SimpleNamespace(start=4, end=5, text=" Fallback original line ", words=[]),
                    SimpleNamespace(start=6, end=6, text="Zero-duration", words=[]),
                    SimpleNamespace(start=7, end=8, text=" ", words=None),
                ],
                object(),
            )

    with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=FakeModel)}):
        actual = transcribe_with_faster_whisper(
            tmp_path / "audio.mp4",
            model_name="small.en",
            device="cpu",
            language="en",
            word_timestamps=True,
        )
    assert len(actual) >= 3
    assert actual[0].text == "Wait what happened?"
    assert actual[0].start == 0.0
    assert actual[0].end == 1.1
    assert actual[-1].text == "Fallback original line"
    assert actual[-1].start == 4.0
    assert all(item.duration > 0 for item in actual)
    assert calls and isinstance(calls[0], dict) and calls[0]["word_timestamps"] is True


def test_word_aligned_whisper_splits_long_sentences_on_actual_word_times() -> None:
    words = [
        SimpleNamespace(start=float(i), end=float(i) + 0.3, word=" hey" if i else "Wait")
        for i in range(8)
    ]

    class FakeModel:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def transcribe(self, *_args: object, **_kwargs: object) -> tuple[object, object]:
            return ([SimpleNamespace(start=0, end=8, text="Wait hey ...", words=words)], object())

    with patch.dict(sys.modules, {"faster_whisper": SimpleNamespace(WhisperModel=FakeModel)}):
        actual = transcribe_with_faster_whisper("sample.mp4", word_timestamps=True)
    assert len(actual) > 1
    assert actual[0].start == 0.0
    assert actual[0].end < actual[-1].end
    assert actual[-1].end == 7.3
