"""Real FFmpeg acceptance smoke: two Clipper beats produce 10-12s encoded media."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from clipper_engine.gameplay.semantics import EditSegment, SemanticPlan
from clipper_engine.rendering.source_fidelity import _build_source_native_filter


def _run(command: list[str]) -> None:
    proc = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr[-1500:]


@pytest.mark.skipif(
    not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
    reason="FFmpeg media-contract smoke requires installed decoder and encoder",
)
def test_edited_source_native_two_beat_lossless_duration_and_audio(tmp_path: Path) -> None:
    source = tmp_path / "source.mov"
    out = tmp_path / "stitched.nut"
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=96x64:rate=60000/1001:duration=25",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=950:sample_rate=48000:duration=25",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "15",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(source),
        ]
    )
    segments = (
        EditSegment(1.5, 5.75, 1.0, "verified_combat_setup"),
        EditSegment(12.0, 18.0, 1.0, "verified_combat_payoff"),
    )
    target = 10.25
    plan = SemanticPlan(
        1.5,
        18.0,
        target,
        target,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.9,
        0.8,
        0.05,
        0.2,
        "combat_montage",
        "source_native_combat_hard_cuts",
        segments,
        (),
        (),
        None,
        ("source-native encoded duration proof",),
    )
    graph, planned = _build_source_native_filter(
        plan, {"_audio_sample_rate": 48000, "sample_aspect_ratio": "1:1"}
    )
    assert "concat=n=2:v=1:a=1" in graph
    assert "trim=start=1.500000:end=5.750000" in graph
    assert "trim=start=12.000000:end=18.000000" in graph
    assert planned == target
    _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-i",
            str(source),
            "-filter_complex",
            graph,
            "-map",
            "[outv]",
            "-map",
            "[aout]",
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "pcm_s16le",
            "-r",
            "60000/1001",
            "-t",
            str(target),
            "-f",
            "nut",
            str(out),
        ]
    )
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_name,r_frame_rate",
            "-of",
            "json",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    result = json.loads(proc.stdout)
    duration = float(result["format"]["duration"])
    assert 10.0 <= duration <= 12.0
    assert abs(duration - target) < 0.05
    assert len(result["streams"]) == 2
    video, audio = result["streams"]
    assert video["codec_name"] == "ffv1"
    assert video["r_frame_rate"] == "60000/1001"
    assert audio["codec_name"] == "pcm_s16le"
