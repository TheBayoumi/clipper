"""Review-only animatic invariants: never treat a still montage as animation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cinematic_animatic.py"
SPEC = importlib.util.spec_from_file_location("cinematic_animatic", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
media = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = media
SPEC.loader.exec_module(media)


def test_preview_artwork_is_explicitly_portrait_and_not_upscaled_from_tiny_source(
    tmp_path: Path,
) -> None:
    artwork = tmp_path / "approved.png"
    Image.new("RGB", (500, 900), "#243955").save(artwork)
    target = tmp_path / "prepared.jpg"
    media._frame(artwork, target)
    with Image.open(target) as rendered:
        assert rendered.size == (1080, 1920)
    assert media.CREATIVE_STATUS == "MOTION_ANIMATIC_REFERENCE_ONLY"


def test_missing_or_incorrect_artwork_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(media.AnimaticError, match="missing"):
        media._frame(tmp_path / "missing.png", tmp_path / "frame.jpg")
    small = tmp_path / "small.png"
    Image.new("RGB", (50, 50), "blue").save(small)
    with pytest.raises(media.AnimaticError, match="below"):
        media._frame(small, tmp_path / "frame.jpg")


def test_no_asset_and_no_provenance_does_not_render(tmp_path: Path) -> None:
    inputs = (tmp_path / "1.png", tmp_path / "2.png", tmp_path / "3.png")
    with pytest.raises(media.AnimaticError, match="missing"):
        media.render(inputs, tmp_path / "missing.wav", tmp_path / "out.mp4", "3")
    wav = tmp_path / "narration.wav"
    wav.write_bytes(b"not a waveform")
    with pytest.raises(media.AnimaticError, match="numeric"):
        media.render(inputs, wav, tmp_path / "out.mp4", "unknown")


def test_sample_timeline_is_under_fifteen_seconds() -> None:
    assert 13.8 < media.DURATION_SECONDS < 14.2
    assert len(media.SHOT_DURATIONS) == 3
