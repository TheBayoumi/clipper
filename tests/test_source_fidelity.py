"""Source-aware decisions are grounded in probed media and measured fidelity."""

from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import Mock, patch

import pytest

from clipper.source_fidelity import (
    FidelityError,
    compare_encoded_to_composition,
    crf_attempts,
    parse_source_profile,
    probe_source_profile,
)


def source_info(
    *,
    fps: str = "30000/1001",
    bitrate: str | None = "4500000",
    transfer: str = "bt709",
) -> dict[str, object]:
    return {
        "streams": [
            {
                "codec_type": "video",
                "codec_name": "vp9",
                "pix_fmt": "yuv420p",
                "width": 1920,
                "height": 1080,
                "avg_frame_rate": fps,
                "bit_rate": bitrate,
                "color_transfer": transfer,
                "color_space": "bt709",
            },
            {"codec_type": "audio", "bit_rate": "192000"},
        ],
        "format": {"bit_rate": "4692000"},
    }


def test_preserves_exact_source_profile_and_uses_measured_bpp() -> None:
    profile = parse_source_profile(source_info(fps="60000/1001"))
    assert profile.fps == "60000/1001"
    assert profile.codec == "vp9"
    assert profile.color_space == "bt709"
    assert profile.video_bitrate == 4500000
    assert profile.bits_per_pixel_frame is not None
    assert crf_attempts(profile) == (18, 16, 14, 12)
    assert profile.as_dict()["width"] == 1920


def test_high_bitrate_source_starts_with_stronger_encode() -> None:
    info = source_info(bitrate="20000000")
    source = parse_source_profile(info)
    assert crf_attempts(source) == (16, 14, 12)


def test_container_bitrate_is_only_an_estimate() -> None:
    info = source_info(bitrate=None)
    profile = parse_source_profile(info)
    assert profile.video_bitrate == 4500000
    info["format"] = {}
    assert parse_source_profile(info).bits_per_pixel_frame is None


@pytest.mark.parametrize("fps", ["0/0", "120/1", "7/1"])
def test_unsupported_source_cadence_fails_closed(fps: str) -> None:
    with pytest.raises(FidelityError, match="unverifiable"):
        parse_source_profile(source_info(fps=fps))


def test_hdr_needs_real_tonemap_not_misleading_sdr_tag() -> None:
    with pytest.raises(FidelityError, match="HDR source"):
        parse_source_profile(source_info(transfer="smpte2084"))


def test_source_probe_reads_actual_ffprobe_data(tmp_path: Path) -> None:
    import json

    result = Mock(stdout=json.dumps(source_info()))
    with patch("clipper.source_fidelity.subprocess.run", return_value=result):
        assert probe_source_profile(tmp_path / "real.mp4").width == 1920
    with (
        patch(
            "clipper.source_fidelity.subprocess.run",
            side_effect=CalledProcessError(1, "ffprobe"),
        ),
        pytest.raises(FidelityError, match="probe"),
    ):
        probe_source_profile(tmp_path / "bad.mp4")


def test_ssim_compares_same_layout_and_rejects_missing_frames(tmp_path: Path) -> None:
    original, rendered = tmp_path / "original.mp4", tmp_path / "output.mp4"
    filter_cmd = ["ffmpeg", "-filter_complex", "[0:v]format=yuv420p,setsar=1[v]"]
    stats = tmp_path / "frames.ssim"

    def fake_ffmpeg(*_args: object, **_kwargs: object) -> Mock:
        stats.write_text("".join(f"n:{i} All:0.995\n" for i in range(1, 61)))
        return Mock()

    with patch("clipper.source_fidelity.subprocess.run", side_effect=fake_ffmpeg):
        score, count = compare_encoded_to_composition(
            filter_cmd,
            source=original,
            output=rendered,
            clip_start=8,
            duration=2,
            fps="30/1",
            stats_path=stats,
        )
    assert score == 0.995 and count == 60
    with (
        patch("clipper.source_fidelity.subprocess.run", return_value=Mock()),
        pytest.raises(FidelityError, match="did not produce"),
    ):
        stats.unlink()
        compare_encoded_to_composition(
            filter_cmd,
            source=original,
            output=rendered,
            clip_start=8,
            duration=2,
            fps="30/1",
            stats_path=stats,
        )
    assert not stats.exists()


def test_ssim_rejects_wrong_graph_and_comparison_failure(tmp_path: Path) -> None:
    with pytest.raises(FidelityError, match="unrecorded"):
        compare_encoded_to_composition(
            [],
            source=tmp_path / "a",
            output=tmp_path / "b",
            clip_start=0,
            duration=2,
            fps="30/1",
            stats_path=tmp_path / "q",
        )
    with (
        patch(
            "clipper.source_fidelity.subprocess.run",
            side_effect=CalledProcessError(1, "ffmpeg", stderr="different frames"),
        ),
        pytest.raises(FidelityError, match="different frames"),
    ):
        compare_encoded_to_composition(
            ["-filter_complex", "format=yuv420p,setsar=1[v]"],
            source=tmp_path / "a",
            output=tmp_path / "b",
            clip_start=0,
            duration=2,
            fps="30/1",
            stats_path=tmp_path / "q",
        )
