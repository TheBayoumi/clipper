"""Clean source-only renderer tests; no YouTube, ASR, model or production footage."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from scripts.tjr_clean_render import (
    clean_ffmpeg_command,
    render,
    validate_inputs,
)


def _fixture(path: Path) -> tuple[dict, dict, Path]:
    source = path / "source.mp4"
    source.write_bytes(b"source integrity fixture")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    selection = {
        "schema": "clipper-clean-source-ranges-v1",
        "selection_run_id": 38469818017,
        "verified_original_run_id": 38358793137,
        "video_id": "3S-fL9rNPp0",
        "channel_id": "UCf1q6dhccWr6eQEcFFnJSbA",
        "source_sha256": sha,
        "clips": [
            {"id": "clean-01", "start_seconds": 0.1, "end_seconds": 1.1}
        ],
    }
    staging = {
        "status": "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED",
        "video_id": selection["video_id"],
        "channel_id": selection["channel_id"],
        "source_sha256": sha,
    }
    return selection, staging, source


def test_clean_ffmpeg_never_uses_captions_asr_text_or_overlay_filters() -> None:
    cmd = clean_ffmpeg_command(Path("raw.mp4"), Path("clean.mp4"), start=1, end=2.5)
    joined = " ".join(cmd).lower()
    assert cmd[cmd.index("-vf") + 1].startswith("crop=")
    assert "scale=1080:1920" in joined
    assert "drawtext" not in joined
    assert "ass=" not in joined
    assert "subtitles=" not in joined
    assert "overlay=" not in joined
    assert "whisper" not in joined
    assert "llama" not in joined
    assert "-sn" in cmd and "-dn" in cmd
    assert "-map_metadata" in cmd


def test_clean_rejects_unverified_original(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import tjr_clean_render

    selection, staging, source = _fixture(tmp_path)
    monkeypatch.setattr(
        tjr_clean_render,
        "_probe",
        lambda _path: {
            "format": {"duration": "2.0"},
            "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
        },
    )
    rows, duration = validate_inputs(selection, staging, source)
    assert duration == 2.0 and len(rows) == 1
    source.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="CLEAN_ORIGINAL_BYTES_SHA256_MISMATCH"):
        validate_inputs(selection, staging, source)
    source.write_bytes(b"source integrity fixture")
    staging["video_id"] = "INVALID_ID_"
    with pytest.raises(ValueError, match="CLEAN_SOURCE_PROVENANCE_MISMATCH"):
        validate_inputs(selection, staging, source)


def test_clean_rejects_duplicate_or_outside_source_ranges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from scripts import tjr_clean_render

    selection, staging, source = _fixture(tmp_path)
    monkeypatch.setattr(
        tjr_clean_render,
        "_probe",
        lambda _path: {
            "format": {"duration": "2.0"},
            "streams": [{"codec_type": "video"}, {"codec_type": "audio"}],
        },
    )
    selection["clips"].append(
        {"id": "clean-01", "start_seconds": 0.2, "end_seconds": 1.2}
    )
    with pytest.raises(ValueError, match="CLEAN_INVALID_SOURCE_RANGE"):
        validate_inputs(selection, staging, source)
    selection["clips"] = [{"id": "clean-01", "start_seconds": 1.5, "end_seconds": 2.6}]
    with pytest.raises(ValueError, match="CLEAN_INVALID_SOURCE_RANGE"):
        validate_inputs(selection, staging, source)


def test_clean_renders_real_portrait_mp4_without_text_sidecars(tmp_path: Path) -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("FFmpeg is required for media integration test")
    selection, staging, source = _fixture(tmp_path)
    subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "color=c=blue:s=320x180:r=24:d=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264",
            "-c:a", "aac", "-shortest", str(source),
        ],
        check=True, capture_output=True,
    )
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    selection["source_sha256"] = sha
    staging["source_sha256"] = sha
    selection_path = tmp_path / "selection.json"
    staging_path = tmp_path / "staged-original.json"
    selection_path.write_text(json.dumps(selection))
    staging_path.write_text(json.dumps(staging))
    output = tmp_path / "rendered"
    report = json.loads(render(selection_path, staging_path, source, output).read_text())
    assert report["verified_clean_clip_count"] == 1
    assert report["policy"]["asr"] is False
    assert report["policy"]["text_overlays"] is False
    assert report["policy"]["automatic_publication"] is False
    assert (output / report["clips"][0]["file"]).is_file()
    assert not list(output.rglob("*.ass"))
    assert not list(output.rglob("*.srt"))
    assert report["clips"][0]["full_decode_passed"] is True
