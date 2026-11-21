"""CPU-only dispatch regressions for Qwen keyframe Volume cache reuse."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest

WORKER = Path(__file__).resolve().parents[1] / "scripts" / "modal_qwen_keyframes.py"
REFERENCE = b"approved-reference-fixture"


class _FakeImage:
    @classmethod
    def from_registry(cls, *_args: object, **_kwargs: object) -> _FakeImage:
        return cls()

    @classmethod
    def debian_slim(cls, *_args: object, **_kwargs: object) -> _FakeImage:
        return cls()

    def __getattr__(self, _name: str) -> Any:
        return lambda *_args, **_kwargs: self


class _FakeVolume:
    @classmethod
    def from_name(cls, *_args: object, **_kwargs: object) -> _FakeVolume:
        return cls()

    def reload(self) -> None:
        pass

    def commit(self) -> None:
        pass


class _FakeSecret:
    @staticmethod
    def from_name(*_args: object, **_kwargs: object) -> object:
        return object()


class _FakeFunction:
    def __init__(self, function: Any) -> None:
        self.function = function
        self.remote = Mock()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)


class _FakeApp:
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        pass

    def function(self, **_kwargs: object) -> Any:
        return _FakeFunction

    def local_entrypoint(self) -> Any:
        return lambda function: function


def _worker(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(Image=_FakeImage, Volume=_FakeVolume, Secret=_FakeSecret, App=_FakeApp),
    )
    spec = importlib.util.spec_from_file_location("testable_modal_qwen", WORKER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(module: ModuleType) -> dict[str, Any]:
    return {
        "schema": module.SCHEMA,
        "model_id": module.MODEL_ID,
        "model_revision": module.MODEL_REVISION,
        "shot_id": "hook",
        "stage": "KEYFRAME_REQUEST_NOT_RENDERED",
        "publish_eligible": False,
        "source_image_sha256": hashlib.sha256(REFERENCE).hexdigest(),
        "fingerprint": "a" * 64,
        "prompt": "A precise approved storyboard reference. " * 12,
        "render_width": 832,
        "render_height": 1472,
        "steps": 40,
        "seed": 724,
    }


def _request_main(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> dict[str, Any]:
    request = _request(module)
    monkeypatch.setitem(
        sys.modules,
        "wan_keyframe_contract",
        SimpleNamespace(
            prepare=lambda *_args: request,
            verified_local_keyframe=lambda *_args: False,
        ),
    )
    return request


def test_cpu_volume_hit_uses_same_fingerprint_and_size_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _worker(monkeypatch)
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path)
    request = _request(module)
    content = b"x" * 100_000
    (tmp_path / f"{request['fingerprint']}.png").write_bytes(content)
    result = module.reuse_cached_keyframe(REFERENCE, request)
    assert result is not None and result["png"] == content
    assert result["image_sha256"] == hashlib.sha256(content).hexdigest()
    assert result["cache"] == "REUSED" and result["gpu"] is None
    (tmp_path / f"{request['fingerprint']}.png").write_bytes(b"too short")
    with pytest.raises(RuntimeError, match="cached keyframe size invalid"):
        module.reuse_cached_keyframe(REFERENCE, request)
    with pytest.raises(ValueError, match="source bytes changed"):
        module.reuse_cached_keyframe(b"other", request)


@pytest.mark.parametrize("cached", [True, False])
def test_cpu_first_dispatch_only_reaches_gpu_for_misses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cached: bool
) -> None:
    module = _worker(monkeypatch)
    request = _request_main(monkeypatch, module)
    reference = tmp_path / "reference.png"
    reference.write_bytes(REFERENCE)
    png = b"p" * 100_000
    response = {
        "png": png,
        "image_sha256": hashlib.sha256(png).hexdigest(),
        "model_id": module.MODEL_ID,
        "model_revision": module.MODEL_REVISION,
        "cache": "REUSED" if cached else "GENERATED",
        "gpu": None if cached else "mock-gpu",
    }
    module.reuse_cached_keyframe.remote.return_value = response.copy() if cached else None
    module.generate_keyframe.remote.return_value = response.copy()
    module.main(reference=str(reference), storyboard="unused", output=str(tmp_path / "out"))
    module.reuse_cached_keyframe.remote.assert_called_once_with(REFERENCE, request)
    assert module.generate_keyframe.remote.call_count == (0 if cached else 1)
    if not cached:
        module.generate_keyframe.remote.assert_called_once_with(REFERENCE, request)
    evidence = json.loads((tmp_path / "out/hook.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "FRAME_GENERATED_TECHNICAL_CANDIDATE"
    assert evidence["publish_eligible"] is False
    assert evidence["image_sha256"] == hashlib.sha256(png).hexdigest()


def test_cpu_cache_validation_failure_records_error_and_never_dispatches_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _worker(monkeypatch)
    _request_main(monkeypatch, module)
    reference = tmp_path / "reference.png"
    reference.write_bytes(REFERENCE)
    module.reuse_cached_keyframe.remote.side_effect = RuntimeError("cached keyframe size invalid")
    with pytest.raises(RuntimeError, match="CPU cache preflight failed") as caught:
        module.main(reference=str(reference), storyboard="unused", output=str(tmp_path / "out"))
    assert isinstance(caught.value.__cause__, RuntimeError)
    assert "cached keyframe size invalid" in str(caught.value.__cause__)
    module.generate_keyframe.remote.assert_not_called()
    evidence = json.loads((tmp_path / "out/hook.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "FRAME_GENERATION_FAILED"
    assert evidence["error_type"] == "RuntimeError"


def test_undeployed_cpu_cache_function_fails_closed_before_any_gpu_dispatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _worker(monkeypatch)
    _request_main(monkeypatch, module)
    reference = tmp_path / "reference.png"
    reference.write_bytes(REFERENCE)
    module.reuse_cached_keyframe.remote.side_effect = LookupError(
        "synthetic Modal function unavailable"
    )

    with pytest.raises(RuntimeError, match="GPU dispatch blocked") as caught:
        module.main(reference=str(reference), storyboard="unused", output=str(tmp_path / "out"))
    assert isinstance(caught.value.__cause__, LookupError)
    module.generate_keyframe.remote.assert_not_called()

    evidence = json.loads((tmp_path / "out/hook.json").read_text(encoding="utf-8"))
    assert evidence["status"] == "FRAME_GENERATION_FAILED"
    assert evidence["error_type"] == "RuntimeError"
    assert "reuse_cached_keyframe" in evidence["error_reason"]
    assert evidence["publish_eligible"] is False
