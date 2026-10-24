"""Tests for the real Wan video-backend boundary, without GPU/model downloads."""

from __future__ import annotations

import importlib.util
import math
import struct
import sys
import wave
from pathlib import Path

import pytest
from PIL import Image

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "wan_shot_contract.py"
SPEC = importlib.util.spec_from_file_location("wan_shot_contract", MODULE)
assert SPEC and SPEC.loader
WAN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = WAN
SPEC.loader.exec_module(WAN)


def make_audio(path: Path, seconds: float = 3.0) -> None:
    rate = 24000
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(
            b"".join(
                struct.pack("<h", int(6000 * math.sin(2 * math.pi * 220 * i / rate)))
                for i in range(round(rate * seconds))
            )
        )


def make_reference(path: Path, width: int = 1024, height: int = 1536) -> None:
    Image.new("RGB", (width, height), (20, 30, 50)).save(path, "PNG")


def test_qualified_s2v_request_is_cached_by_provenance(tmp_path: Path) -> None:
    reference, audio = tmp_path / "identity.png", tmp_path / "kokoro.wav"
    make_reference(reference)
    make_audio(audio)
    prompt = "An expressive cinematic character turns toward the camera and speaks."
    first = WAN.prepare(reference, audio, prompt)
    again = WAN.prepare(reference, audio, prompt)
    assert first["fingerprint"] == again["fingerprint"]
    assert first["model_id"] == "Wan-AI/Wan2.2-S2V-14B"
    assert first["stage"] == "INPUTS_QUALIFIED_NOT_RENDERED"
    assert first["publish_eligible"] is False
    assert first["creative_review"] == "PENDING"
    changed = WAN.prepare(reference, audio, prompt + " Slowly.")
    assert changed["fingerprint"] != first["fingerprint"]


def test_degraded_image_rejected_before_any_gpu_request(tmp_path: Path) -> None:
    reference, audio = tmp_path / "low.png", tmp_path / "voice.wav"
    make_reference(reference, 512, 768)
    make_audio(audio)
    with pytest.raises(WAN.ShotContractError, match="too small"):
        WAN.prepare(reference, audio, "Detailed stylized cinematic adult presenter speaking.")


def test_non_kokoro_audio_and_bad_duration_fail_closed(tmp_path: Path) -> None:
    reference, audio = tmp_path / "face.png", tmp_path / "voice.wav"
    make_reference(reference)
    make_audio(audio, 8.0)
    with pytest.raises(WAN.ShotContractError, match="1 to 6.5"):
        WAN.prepare(reference, audio, "Cinematic medium close-up of the narrator speaking.")


def test_i2v_cannot_silently_claim_to_be_implemented(tmp_path: Path) -> None:
    reference, audio = tmp_path / "face.png", tmp_path / "voice.wav"
    make_reference(reference)
    make_audio(audio)
    with pytest.raises(WAN.ShotContractError, match="not yet qualified"):
        WAN.prepare(
            reference,
            audio,
            "A character is lifting a book in a warmly lit studio.",
            backend="i2v",
        )


def test_video_gate_rejects_missing_output_without_creative_approval(tmp_path: Path) -> None:
    reference, audio = tmp_path / "face.png", tmp_path / "voice.wav"
    make_reference(reference)
    make_audio(audio)
    request = WAN.prepare(
        reference, audio, "The stylized presenter addresses the audience naturally."
    )
    with pytest.raises(WAN.ShotContractError, match="missing"):
        WAN.audit_mp4(tmp_path / "never-rendered.mp4", request)
