from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError, URLError

import pytest

from clipper.canonical import CanonicalTimeline, CanonicalWord
from clipper.providers import factory
from clipper.providers.base import ModelIdentity
from clipper.providers.local import (
    LocalEditorialProvider,
    LocalVisionProvider,
    ProviderUnavailable,
)
from clipper.providers.local import _usage as local_usage
from clipper.providers.modal_endpoint import ModalEndpointEditorialProvider
from clipper.providers.modal_speech import (
    ModalAlignmentProvider,
    ModalDiarizationProvider,
    ModalMediaBridge,
    ModalTranscriptionProvider,
    _ModalSpeechBase,
)
from clipper.providers.speech import (
    FasterWhisperTranscriptionProvider,
    PassthroughDiarizationProvider,
    PyannoteDiarizationProvider,
    WhisperXAlignmentProvider,
    _alignment_segments,
    _drop_nonmonotonic_alignment_updates,
    _float_value,
    _normalize_token,
    _replace_words,
    apply_speaker_turns,
    apply_whisperx_alignment,
)


class _Tensor:
    def __init__(self, count: int) -> None:
        self.count = count
        self.shape = (1, count)

    def __getitem__(self, _key: object) -> _Tensor:
        return self

    def numel(self) -> int:
        return self.count


class _Batch(dict[str, _Tensor]):
    def to(self, _device: object) -> _Batch:
        return self


class _Tokenizer:
    def __init__(self, text: str = '{"ok": true}') -> None:
        self.text = text
        self.messages: object | None = None

    def apply_chat_template(self, messages: object, **_kwargs: object) -> str:
        self.messages = messages
        return "rendered"

    def __call__(self, _rendered: str, **_kwargs: object) -> _Batch:
        return _Batch(input_ids=_Tensor(3))

    def decode(self, _generated: object, **_kwargs: object) -> str:
        return self.text


class _Processor(_Tokenizer):
    def apply_chat_template(self, messages: object, **_kwargs: object) -> _Batch:
        self.messages = messages
        return _Batch(input_ids=_Tensor(4))


class _Model:
    device = "cpu"

    def generate(self, **_kwargs: object) -> list[_Tensor]:
        return [_Tensor(2)]


def _identity(model_id: str = "model") -> ModelIdentity:
    return ModelIdentity(model_id, "rev", "none", "test", "editor", "schema")


def _timeline(count: int = 4) -> CanonicalTimeline:
    return CanonicalTimeline(
        "video",
        "source",
        tuple(
            CanonicalWord(
                f"video:w{index:07d}:x",
                text,
                float(index),
                float(index) + 0.8,
                None,
                0.9,
                "word_exact",
                "test",
            )
            for index, text in enumerate(("Hello", "world", "again", "today")[:count])
        ),
    )


def test_local_usage_and_editorial_provider_contract() -> None:
    usage = local_usage("now", 0.0, provider="local", input_units=3, output_units=2)
    assert usage.provider == "local"
    assert usage.input_units == 3
    provider = LocalEditorialProvider(model_id="editor", revision="rev")
    provider._tokenizer = _Tokenizer()
    provider._model = _Model()
    result = provider.complete_json(task="semantic_cores:0", payload={"words": []})
    assert result.value == {"ok": True}
    assert result.usage.input_units == 3
    assert result.usage.output_units == 2
    assert provider.identity.model_id == "editor"


@pytest.mark.parametrize("text,match", [("not-json", "valid JSON"), ("[]", "must be an object")])
def test_local_editorial_provider_rejects_bad_json(text: str, match: str) -> None:
    provider = LocalEditorialProvider()
    provider._tokenizer = _Tokenizer(text)
    provider._model = _Model()
    with pytest.raises(ValueError, match=match):
        provider.complete_json(task="semantic_cores:0", payload={})


def test_local_provider_load_paths_without_model_downloads() -> None:
    editorial = LocalEditorialProvider()
    with (
        patch("clipper.providers.local.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="editorial"),
    ):
        editorial._load()

    tokenizer = object()
    model = object()
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    with patch("clipper.providers.local.importlib.import_module", return_value=transformers):
        assert editorial._load() == (tokenizer, model)
        assert editorial._load() == (tokenizer, model)
    assert transformers.AutoTokenizer.from_pretrained.call_count == 1

    vision = LocalVisionProvider()
    with (
        patch("clipper.providers.local.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="vision"),
    ):
        vision._load()
    processor = object()
    vmodel = object()
    transformers = SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)),
        AutoModelForMultimodalLM=SimpleNamespace(from_pretrained=Mock(return_value=vmodel)),
    )
    with patch("clipper.providers.local.importlib.import_module", return_value=transformers):
        assert vision._load() == (processor, vmodel)


def test_local_vision_provider_contract(tmp_path: Path) -> None:
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")
    provider = LocalVisionProvider()
    provider._processor = _Processor()
    provider._model = _Model()
    result = provider.inspect(task="visual_timeline_scout", frames=[frame], context={"x": 1})
    assert result.value == {"ok": True}
    assert result.usage.input_units == 4
    with pytest.raises(ValueError, match="at least one frame"):
        provider.inspect(task="visual_timeline_scout", frames=[], context={})

    provider._processor = _Processor("not-json")
    with pytest.raises(ValueError, match="valid JSON"):
        provider.inspect(task="visual_timeline_scout", frames=[frame], context={})
    provider._processor = _Processor("[]")
    with pytest.raises(ValueError, match="must be an object"):
        provider.inspect(task="visual_timeline_scout", frames=[frame], context={})


def _endpoint(**overrides: object) -> ModalEndpointEditorialProvider:
    values: dict[str, object] = {
        "endpoint_url": "https://example.modal.run",
        "proxy_token_id": "id",
        "proxy_token_secret": "secret",
        "identity": _identity("endpoint-model"),
        "timeout_seconds": 1.0,
    }
    values.update(overrides)
    return ModalEndpointEditorialProvider(**values)  # type: ignore[arg-type]


def test_modal_endpoint_configuration_and_json_parser() -> None:
    with pytest.raises(ProviderUnavailable, match="URL is required"):
        _endpoint(endpoint_url="")
    with pytest.raises(ProviderUnavailable, match="must use https"):
        _endpoint(endpoint_url="http://bad")
    with pytest.raises(ProviderUnavailable, match="proxy token"):
        _endpoint(proxy_token_id="")
    provider = _endpoint(endpoint_url="https://example.modal.run/")
    assert provider.endpoint_url == "https://example.modal.run"
    assert provider._json_object('```json\n{"ok": true}\n```') == {"ok": True}
    with pytest.raises(ValueError, match="JSON object"):
        provider._json_object("[]")


class _HTTPResponse:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __enter__(self) -> _HTTPResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _http_error(code: int, detail: bytes = b"detail") -> HTTPError:
    return HTTPError("https://example", code, "error", {}, io.BytesIO(detail))


def test_modal_endpoint_success_retry_and_usage() -> None:
    provider = _endpoint()
    payload = {
        "choices": [{"message": {"content": '{"cores": []}'}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7},
    }
    with (
        patch(
            "clipper.providers.modal_endpoint.urlopen",
            side_effect=[_http_error(503), _HTTPResponse(payload)],
        ) as opened,
        patch("clipper.providers.modal_endpoint.time.sleep") as sleep,
    ):
        result = provider.complete_json(task="semantic_cores:0", payload={"words": []})
    assert result.value == {"cores": []}
    assert result.usage.input_units == 11
    assert result.usage.output_units == 7
    assert opened.call_count == 2
    sleep.assert_called_once_with(1.0)


def test_modal_endpoint_network_and_response_failures() -> None:
    provider = _endpoint()
    with (
        patch("clipper.providers.modal_endpoint.urlopen", return_value=_HTTPResponse([])),
        pytest.raises(ValueError, match="invalid response"),
    ):
        provider.complete_json(task="semantic_cores:0", payload={})
    for payload, match in (
        ({}, "no choices"),
        ({"choices": [{}]}, "no message content"),
    ):
        with (
            patch("clipper.providers.modal_endpoint.urlopen", return_value=_HTTPResponse(payload)),
            pytest.raises(ValueError, match=match),
        ):
            provider.complete_json(task="semantic_cores:0", payload={})

    with (
        patch("clipper.providers.modal_endpoint.urlopen", side_effect=_http_error(400, b"bad")),
        pytest.raises(RuntimeError, match="HTTP 400: bad"),
    ):
        provider.complete_json(task="semantic_cores:0", payload={})
    with (
        patch("clipper.providers.modal_endpoint.urlopen", side_effect=URLError("offline")),
        patch("clipper.providers.modal_endpoint.time.sleep"),
        pytest.raises(RuntimeError, match="request failed: offline"),
    ):
        provider.complete_json(task="semantic_cores:0", payload={})


def test_speech_helpers_preserve_canonical_identity_and_source_order() -> None:
    timeline = _timeline()
    assert _normalize_token(" Hello,! ") == "hello"
    assert _float_value("1.5", 0.0) == 1.5
    assert _float_value(None, 2.0) == 2.0
    replaced = _replace_words(
        timeline,
        {timeline.words[0].word_id: {"source_start": 0.1, "speaker_id": "S1"}},
    )
    assert replaced.words[0].word_id == timeline.words[0].word_id
    assert replaced.words[0].source_start == 0.1
    assert replaced.words[0].speaker_id == "S1"
    assert _alignment_segments(CanonicalTimeline("v", "s", ())) == []
    assert len(_alignment_segments(timeline, max_seconds=1.0)) >= 2


def test_whisperx_alignment_maps_tokens_and_drops_nonmonotonic_updates() -> None:
    timeline = _timeline()
    aligned = apply_whisperx_alignment(
        timeline,
        [
            {
                "words": [
                    {"word": "Hello", "start": 0.05, "end": 0.7, "score": 0.9},
                    {"word": "world", "start": 0.8, "end": 1.6, "score": 0.8},
                ]
            }
        ],
    )
    assert aligned.words[0].timing_mode == "aligned"
    assert "whisperx" in aligned.words[0].transcript_source
    with pytest.raises(ValueError, match="no canonical word matches"):
        apply_whisperx_alignment(timeline, [{"words": [{"word": "missing", "start": 0, "end": 1}]}])

    replacements = {
        timeline.words[0].word_id: {"source_start": 2.0, "confidence": 0.1},
        timeline.words[1].word_id: {"source_start": 1.0, "confidence": 0.9},
    }
    stable = _drop_nonmonotonic_alignment_updates(timeline, replacements)
    assert timeline.words[0].word_id not in stable
    assert timeline.words[1].word_id in stable


def test_speaker_turn_assignment_uses_maximum_overlap() -> None:
    timeline = _timeline()
    assigned = apply_speaker_turns(timeline, [(0.0, 1.0, "A"), (1.0, 4.0, "B")])
    assert assigned.words[0].speaker_id == "A"
    assert assigned.words[2].speaker_id == "B"


def test_faster_whisper_provider_load_and_transcribe_paths(tmp_path: Path) -> None:
    provider = FasterWhisperTranscriptionProvider(
        model_id="tiny", device="cpu", compute_type="int8"
    )
    with (
        patch("clipper.providers.speech.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="asr"),
    ):
        provider._load()

    whisper_model = Mock()
    module = SimpleNamespace(WhisperModel=Mock(return_value=whisper_model))
    with patch("clipper.providers.speech.importlib.import_module", return_value=module):
        assert provider._load() is whisper_model
    raw = SimpleNamespace(start=0.0, end=0.5, word=" hello ", probability=0.8)
    whisper_model.transcribe.return_value = ([SimpleNamespace(words=[raw])], object())
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    result = provider.transcribe(source, video_id="video", source_hash="hash")
    assert result.value.words[0].text == "hello"
    assert result.value.words[0].confidence == 0.8

    whisper_model.transcribe.return_value = ([SimpleNamespace(words=[])], object())
    with pytest.raises(ValueError, match="no timestamped words"):
        provider.transcribe(source, video_id="video", source_hash="hash")


def test_whisperx_alignment_provider_external_boundary(tmp_path: Path) -> None:
    provider = WhisperXAlignmentProvider(device="cpu")
    with pytest.raises(ValueError, match="empty"):
        provider.align(tmp_path / "x.wav", CanonicalTimeline("v", "s", ()))
    with (
        patch("clipper.providers.speech.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="alignment"),
    ):
        provider.align(tmp_path / "x.wav", _timeline())

    module = SimpleNamespace(
        load_audio=Mock(return_value="audio"),
        load_align_model=Mock(return_value=("model", {"meta": True})),
        align=Mock(
            return_value={
                "segments": [
                    {
                        "words": [
                            {"word": "Hello", "start": 0.0, "end": 0.7, "score": 0.9},
                            {"word": "world", "start": 1.0, "end": 1.7, "score": 0.9},
                        ]
                    }
                ]
            }
        ),
    )
    with patch("clipper.providers.speech.importlib.import_module", return_value=module):
        result = provider.align(tmp_path / "x.wav", _timeline())
    assert result.value.words[0].timing_mode == "aligned"
    module.align.return_value = {}
    with (
        patch("clipper.providers.speech.importlib.import_module", return_value=module),
        pytest.raises(ValueError, match="no aligned segments"),
    ):
        provider.align(tmp_path / "x.wav", _timeline())


def test_pyannote_provider_load_turns_and_diarize(tmp_path: Path) -> None:
    with pytest.raises(ProviderUnavailable, match="HF_TOKEN"):
        PyannoteDiarizationProvider(token="")._load()
    provider = PyannoteDiarizationProvider(token=pytest.__name__, device="cpu")
    with (
        patch("clipper.providers.speech.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="diarization"),
    ):
        provider._load()

    pipeline = Mock()
    module = SimpleNamespace(Pipeline=SimpleNamespace(from_pretrained=Mock(return_value=pipeline)))
    torch = SimpleNamespace(device=Mock(return_value="cpu-device"))
    with patch(
        "clipper.providers.speech.importlib.import_module",
        side_effect=lambda name: module if name == "pyannote.audio" else torch,
    ):
        assert provider._load() is pipeline
    pipeline.to.assert_called_once_with("cpu-device")

    segment = SimpleNamespace(start=0.0, end=2.0)
    diarization = SimpleNamespace(itertracks=lambda **_kwargs: [(segment, "track", "S1")])
    assert provider._turns(diarization) == [(0.0, 2.0, "S1")]
    with pytest.raises(ValueError, match="no speaker diarization tracks"):
        provider._turns(object())
    empty = SimpleNamespace(itertracks=lambda **_kwargs: [])
    with pytest.raises(ValueError, match="no speaker turns"):
        provider._turns(empty)

    pipeline.return_value = diarization
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    result = provider.diarize(source, _timeline())
    assert result.value.words[0].speaker_id == "S1"
    passthrough = PassthroughDiarizationProvider().diarize(source, _timeline())
    assert passthrough.degraded is True
    assert passthrough.usage.provider == "degraded"


def test_modal_media_bridge_paths_upload_cache_and_speech_derivative(tmp_path: Path) -> None:
    bridge = ModalMediaBridge("volume")
    with (
        patch("clipper.providers.modal_speech.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="modal"),
    ):
        bridge._volume()
    volume = Mock()
    modal = SimpleNamespace(Volume=SimpleNamespace(from_name=Mock(return_value=volume)))
    with patch("clipper.providers.modal_speech.importlib.import_module", return_value=modal):
        assert bridge._volume() is volume

    assert bridge._mounted_modal_path(Path("/media/inputs/hash.mkv")) == "/media/inputs/hash.mkv"
    assert bridge._mounted_modal_path(tmp_path / "x.mkv") is None
    audio = tmp_path / "audio.wav"
    audio.write_bytes(b"audio")
    assert bridge._speech_source(audio, "hash") == audio

    video = tmp_path / "video.mp4"
    video.write_bytes(b"v" * 100)

    def fake_ffmpeg(command: list[str], **_kwargs: object) -> None:
        Path(command[-1]).write_bytes(b"w" * 64)

    with patch("clipper.providers.modal_speech.subprocess.run", side_effect=fake_ffmpeg) as run:
        derivative = bridge._speech_source(video, "hash")
        assert bridge._speech_source(video, "hash") == derivative
    assert derivative.stat().st_size == 64
    assert run.call_count == 1

    mounted = Path("/media/inputs/already.wav")
    assert bridge.ensure_uploaded(mounted, "mounted") == "/media/inputs/already.wav"
    assert bridge.ensure_uploaded(mounted, "mounted") == "/media/inputs/already.wav"


def test_modal_media_bridge_volume_upload_and_existing_reuse(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")

    class Upload:
        def __init__(self) -> None:
            self.puts: list[tuple[str, str]] = []

        def __enter__(self) -> Upload:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def put_file(self, local: str, remote: str) -> None:
            self.puts.append((local, remote))

    upload = Upload()
    volume = SimpleNamespace(listdir=Mock(return_value=[]), batch_upload=Mock(return_value=upload))
    bridge = ModalMediaBridge("volume")
    with patch.object(bridge, "_volume", return_value=volume):
        remote = bridge.ensure_uploaded(source, "hash")
        assert bridge.ensure_uploaded(source, "hash") == remote
    assert remote == "/media/inputs/hash.wav"
    assert upload.puts == [(str(source), "/inputs/hash.wav")]

    reused = ModalMediaBridge("volume")
    existing = SimpleNamespace(listdir=Mock(return_value=["present"]), batch_upload=Mock())
    with patch.object(reused, "_volume", return_value=existing):
        assert reused.ensure_uploaded(source, "other") == "/media/inputs/other.wav"
    existing.batch_upload.assert_not_called()


def _modal_provider(cls: type[_ModalSpeechBase], bridge: ModalMediaBridge, name: str):
    return cls(app_name="app", function_name=name, identity=_identity(name), media_bridge=bridge)


def test_modal_speech_base_function_identity_and_usage() -> None:
    bridge = ModalMediaBridge()
    base = _ModalSpeechBase(
        app_name="app", function_name="fn", identity=_identity("base"), media_bridge=bridge
    )
    with (
        patch("clipper.providers.modal_speech.importlib.import_module", side_effect=ImportError),
        pytest.raises(ProviderUnavailable, match="modal"),
    ):
        base._function()
    handle = object()
    modal = SimpleNamespace(Function=SimpleNamespace(from_name=Mock(return_value=handle)))
    with patch("clipper.providers.modal_speech.importlib.import_module", return_value=modal):
        assert base._function() is handle
    assert base._resolved_identity({}).model_id == "base"
    resolved = base._resolved_identity({"model": {"model_id": "actual", "revision": "r2"}})
    assert (resolved.model_id, resolved.revision) == ("actual", "r2")
    usage = base._usage(
        {
            "usage": {
                "started_at": "now",
                "duration_seconds": 1,
                "gpu_type": "L4",
                "gpu_seconds": 2,
                "peak_vram_mb": 3,
                "input_units": 4,
                "output_units": 5,
                "estimated_cost_usd": 0.01,
            }
        }
    )
    assert usage.gpu_type == "L4"
    assert usage.peak_vram_mb == 3


def test_modal_transcription_alignment_and_diarization_contracts(tmp_path: Path) -> None:
    source = tmp_path / "audio.wav"
    source.write_bytes(b"audio")
    bridge = Mock(spec=ModalMediaBridge)
    bridge.ensure_uploaded.return_value = "/media/input.wav"
    timeline = _timeline()

    transcription = _modal_provider(ModalTranscriptionProvider, bridge, "transcribe")
    function = Mock()
    function.remote.return_value = {
        "words": [
            {
                "text": "hello",
                "source_start": 0.0,
                "source_end": 0.5,
                "confidence": 0.9,
            }
        ]
    }
    with patch.object(transcription, "_function", return_value=function):
        result = transcription.transcribe(source, video_id="video", source_hash="source")
    assert result.value.words[0].text == "hello"
    function.remote.return_value = {"words": "bad"}
    with (
        patch.object(transcription, "_function", return_value=function),
        pytest.raises(ValueError, match="transcription provider"),
    ):
        transcription.transcribe(source, video_id="video", source_hash="source")

    alignment = _modal_provider(ModalAlignmentProvider, bridge, "align")
    function.remote.return_value = {
        "segments": [
            {
                "words": [
                    {"word": "Hello", "start": 0.0, "end": 0.7, "score": 0.9},
                    {"word": "world", "start": 1.0, "end": 1.7, "score": 0.9},
                ]
            }
        ]
    }
    with patch.object(alignment, "_function", return_value=function):
        assert alignment.align(source, timeline).value.words[0].timing_mode == "aligned"
    function.remote.return_value = {}
    with (
        patch.object(alignment, "_function", return_value=function),
        pytest.raises(ValueError, match="alignment provider"),
    ):
        alignment.align(source, timeline)

    diarization = _modal_provider(ModalDiarizationProvider, bridge, "diarize")
    function.remote.return_value = {"turns": [[0, 2, "S1"]]}
    with patch.object(diarization, "_function", return_value=function):
        assert diarization.diarize(source, timeline).value.words[0].speaker_id == "S1"
    function.remote.return_value = {"error": {"type": "Boom", "message": "bad"}}
    with (
        patch.object(diarization, "_function", return_value=function),
        pytest.raises(RuntimeError, match="Boom: bad"),
    ):
        diarization.diarize(source, timeline)
    function.remote.return_value = {"turns": [[0, 1]]}
    with (
        patch.object(diarization, "_function", return_value=function),
        pytest.raises(ValueError, match="turn is invalid"),
    ):
        diarization.diarize(source, timeline)


def test_provider_factory_selects_local_modal_managed_and_degraded(monkeypatch) -> None:
    assert isinstance(factory.editorial_provider("local-lite"), LocalEditorialProvider)
    assert isinstance(factory.vision_provider("local-lite"), LocalVisionProvider)
    with pytest.raises(ValueError, match="large VLM"):
        factory.vision_provider("local-lite", large=True)

    monkeypatch.setenv("CLIPPER_MODAL_EDITORIAL_BACKEND", "managed")
    monkeypatch.setenv("CLIPPER_MODAL_EDITORIAL_ENDPOINT_URL", "https://example.modal.run")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_ID", "id")
    monkeypatch.setenv("MODAL_PROXY_TOKEN_SECRET", "secret")
    managed = factory.editorial_provider("balanced")
    assert isinstance(managed, ModalEndpointEditorialProvider)

    monkeypatch.setenv("CLIPPER_MODAL_EDITORIAL_BACKEND", "function")
    modal_editor = factory.editorial_provider("balanced")
    assert modal_editor.identity.inference_engine == "modal-transformers"
    modal_vision = factory.vision_provider("quality", large=True)
    assert modal_vision.identity.model_id == "Qwen/Qwen3-VL-30B-A3B-Instruct"

    monkeypatch.setenv("CLIPPER_DIARIZATION_MODE", "passthrough")
    local_speech = factory.speech_providers("local-lite")
    assert isinstance(local_speech[2], PassthroughDiarizationProvider)
    modal_speech = factory.speech_providers("balanced")
    assert isinstance(modal_speech[0], ModalTranscriptionProvider)
    assert isinstance(modal_speech[1], ModalAlignmentProvider)
    assert isinstance(modal_speech[2], PassthroughDiarizationProvider)

    monkeypatch.setenv("CLIPPER_DIARIZATION_MODE", "invalid")
    with pytest.raises(ValueError, match="unsupported diarization mode"):
        factory.speech_providers("balanced")
    monkeypatch.setenv("CLIPPER_MODAL_EDITORIAL_BACKEND", "invalid")
    with pytest.raises(ValueError, match="unsupported Modal editorial backend"):
        factory.editorial_provider("balanced")
