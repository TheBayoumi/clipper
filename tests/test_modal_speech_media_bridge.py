from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from clipper.providers.modal_speech import ModalMediaBridge


def _fake_modal_volume() -> tuple[object, Mock, Mock]:
    volume = Mock()
    volume.listdir.return_value = []
    upload = Mock()
    manager = Mock()
    manager.__enter__ = Mock(return_value=upload)
    manager.__exit__ = Mock(return_value=False)
    volume.batch_upload.return_value = manager
    modal_module = SimpleNamespace(Volume=SimpleNamespace(from_name=Mock(return_value=volume)))
    return modal_module, volume, upload


def test_modal_mounted_source_skips_derivative_and_volume_upload() -> None:
    source = Path("/media/inputs/abc123.mkv")
    bridge = ModalMediaBridge("media")

    with (
        patch("clipper.providers.modal_speech.subprocess.run") as ffmpeg,
        patch("clipper.providers.modal_speech.importlib.import_module") as import_module,
    ):
        remote = bridge.ensure_uploaded(source, "abc123")

    assert remote == "/media/inputs/abc123.mkv"
    ffmpeg.assert_not_called()
    import_module.assert_not_called()


def test_video_source_is_reduced_to_cached_speech_wav_before_modal_upload(tmp_path: Path) -> None:
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"video" * 200_000)
    modal_module, _volume, upload = _fake_modal_volume()

    def fake_ffmpeg(command: list[str], **_kwargs: object) -> None:
        output = Path(command[-1])
        output.write_bytes(b"R" * 96_000)

    bridge = ModalMediaBridge("media")
    with (
        patch("clipper.providers.modal_speech.subprocess.run", side_effect=fake_ffmpeg) as ffmpeg,
        patch("clipper.providers.modal_speech.importlib.import_module", return_value=modal_module),
    ):
        remote = bridge.ensure_uploaded(source, "abc123")

    assert remote == "/media/inputs/abc123.speech.wav"
    command = ffmpeg.call_args.args[0]
    assert "-vn" in command
    assert "16000" in command
    assert "pcm_s16le" in command
    uploaded_source, uploaded_target = upload.put_file.call_args.args
    derivative = Path(uploaded_source)
    assert derivative.name == "abc123.speech.wav"
    assert derivative.is_file()
    assert uploaded_target == "/inputs/abc123.speech.wav"


def test_cached_speech_derivative_skips_ffmpeg_on_next_bridge(tmp_path: Path) -> None:
    source = tmp_path / "episode.mkv"
    source.write_bytes(b"video" * 200_000)
    derivative_dir = tmp_path / ".speech-cache"
    derivative_dir.mkdir()
    derivative = derivative_dir / "abc123.speech.wav"
    derivative.write_bytes(b"R" * 96_000)
    modal_module, _volume, upload = _fake_modal_volume()

    bridge = ModalMediaBridge("media")
    with (
        patch("clipper.providers.modal_speech.subprocess.run") as ffmpeg,
        patch("clipper.providers.modal_speech.importlib.import_module", return_value=modal_module),
    ):
        remote = bridge.ensure_uploaded(source, "abc123")

    assert remote == "/media/inputs/abc123.speech.wav"
    ffmpeg.assert_not_called()
    upload.put_file.assert_called_once_with(str(derivative), "/inputs/abc123.speech.wav")


def test_existing_audio_source_keeps_direct_upload_contract(tmp_path: Path) -> None:
    source = tmp_path / "episode.wav"
    source.write_bytes(b"R" * 96_000)
    modal_module, _volume, upload = _fake_modal_volume()

    bridge = ModalMediaBridge("media")
    with (
        patch("clipper.providers.modal_speech.subprocess.run") as ffmpeg,
        patch("clipper.providers.modal_speech.importlib.import_module", return_value=modal_module),
    ):
        remote = bridge.ensure_uploaded(source, "abc123")

    assert remote == "/media/inputs/abc123.wav"
    ffmpeg.assert_not_called()
    upload.put_file.assert_called_once_with(str(source), "/inputs/abc123.wav")
