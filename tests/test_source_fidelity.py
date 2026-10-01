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


def test_ssim_preserves_watermark_input_and_compares_encoded_third_input(tmp_path: Path) -> None:
    source, watermark, output = (
        tmp_path / name for name in ("source.mp4", "logo.png", "output.mp4")
    )
    graph = (
        "[0:v]null[base];[1:v]scale=180:-1[wm];[base][wm]overlay=10:10,format=yuv420p,setsar=1[v]"
    )
    original_inputs = ["ffmpeg", "-ss", "8.000", "-i", str(source), "-i", str(watermark)]
    command = [*original_inputs, "-filter_complex", graph]
    stats = tmp_path / "stats.txt"

    def compare(actual: list[str], **kwargs: object) -> Mock:
        assert actual[: len(original_inputs)] == original_inputs
        assert actual[len(original_inputs) : len(original_inputs) + 2] == ["-i", str(output)]
        reference = actual[actual.index("-filter_complex") + 1]
        assert "[1:v]scale=180:-1" in reference
        assert "[2:v]trim=start=0,settb=AVTB,setpts=N/(30/1)/TB[encoded]" in reference
        assert "[reference][encoded]ssim=" in reference
        stats.write_text("".join(f"n:{i} All:0.999\n" for i in range(60)))
        return Mock()

    with patch("clipper.source_fidelity.subprocess.run", side_effect=compare):
        assert compare_encoded_to_composition(
            command,
            source=source,
            output=output,
            clip_start=8,
            duration=2,
            fps="30/1",
            stats_path=stats,
        ) == (0.999, 60)


def test_real_watermarked_video_keeps_corresponding_frames_aligned(tmp_path: Path) -> None:
    import shutil
    import subprocess

    from PIL import Image

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for the real composition regression")
    source, output, logo = (tmp_path / name for name in ("source.mkv", "output.mp4", "logo.png"))
    Image.new("RGBA", (50, 30), (30, 190, 20, 255)).save(logo)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30",
            "-t",
            "2",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    graph = (
        "[0:v]null[base];[1:v]scale=50:-1[wm];"
        "[base][wm]overlay=W-w-10:10:format=auto,format=yuv420p,setsar=1[v]"
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        "0.000",
        "-i",
        str(source),
        "-i",
        str(logo),
        "-filter_complex",
        graph,
        "-map",
        "[v]",
        "-t",
        "2",
        "-c:v",
        "libx264",
        "-crf",
        "12",
        str(output),
    ]
    subprocess.run(command, check=True, capture_output=True)
    score, frames = compare_encoded_to_composition(
        command,
        source=source,
        output=output,
        clip_start=0,
        duration=2,
        fps="30/1",
        stats_path=tmp_path / "ssim.txt",
    )
    assert score >= 0.99
    assert frames == 60


def test_real_animated_render_drops_negative_seek_frames_consistently(tmp_path: Path) -> None:
    import shutil
    import subprocess

    from PIL import Image

    from clipper.models import ClipCandidate, TranscriptSegment
    from clipper.render import build_ffmpeg_command
    from clipper.source_fidelity import SourceProfile
    from clipper.tiktok import create_tiktok_ass

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for the negative-timestamp regression")
    source, output, logo = (tmp_path / name for name in ("source.mkv", "output.mp4", "logo.png"))
    Image.new("RGBA", (50, 30), (30, 190, 20, 255)).save(logo)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=30000/1001",
            "-t",
            "3",
            "-c:v",
            "ffv1",
            str(source),
        ],
        check=True,
        capture_output=True,
    )
    clip = ClipCandidate("video", 0.3, 2.3, "Why did this happen?", 1)
    ass = create_tiktok_ass(
        clip,
        [TranscriptSegment(0.3, 2.3, "Why did this happen?")],
        tmp_path / "captions.ass",
        hook_text=clip.text,
    )
    profile = SourceProfile(320, 240, "30000/1001", "ffv1", "yuv420p", None, None)
    command = build_ffmpeg_command(
        source,
        output,
        clip,
        ass,
        watermark_path=logo,
        source_fps=profile.fps,
        source_profile=profile,
        crf_override=12,
    )
    graph_index = command.index("-filter_complex") + 1
    assert "fps=30000/1001:start_time=0" in command[graph_index]
    command[graph_index] = command[graph_index].replace(
        "[0:v]split=2", "[0:v]setpts=PTS-0.1/TB,split=2", 1
    )
    subprocess.run(command, check=True, capture_output=True)
    score, frames = compare_encoded_to_composition(
        command,
        source=source,
        output=output,
        clip_start=0.3,
        duration=2,
        fps=profile.fps,
        stats_path=tmp_path / "ssim.txt",
    )
    assert score >= 0.99
    assert frames >= 59
