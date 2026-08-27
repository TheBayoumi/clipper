from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from clipper.providers.base import ModelIdentity
from clipper.providers.modal import ModalEditorialProvider, ModalVisionProvider


def _identity(model_id: str) -> ModelIdentity:
    return ModelIdentity(
        model_id,
        "requested",
        "none",
        "modal-transformers",
        "prompt",
        "schema",
    )


def test_modal_vision_provider_reuses_class_handle_and_surfaces_runtime(
    tmp_path: Path,
) -> None:
    ready = SimpleNamespace(
        remote=Mock(
            return_value={
                "value": {"ready": True},
                "model": {"model_id": "vision", "revision": "actual"},
                "runtime": {
                    "worker_lifecycle_id": "worker-a",
                    "model_load_count": 1,
                },
            }
        )
    )
    inspect = SimpleNamespace(
        remote=Mock(
            return_value={
                "value": {"observations": []},
                "model": {"model_id": "vision", "revision": "actual"},
                "usage": {
                    "duration_seconds": 1.0,
                    "peak_vram_mb_by_device": {"0": 10.0, "1": 11.0},
                },
                "runtime": {
                    "worker_lifecycle_id": "worker-a",
                    "model_load_count": 1,
                },
            }
        )
    )
    instance = SimpleNamespace(ready=ready, inspect=inspect)
    class_handle = Mock(return_value=instance)
    modal = SimpleNamespace(Cls=SimpleNamespace(from_name=Mock(return_value=class_handle)))
    provider = ModalVisionProvider(
        app_name="app",
        class_name="VisionModel",
        method_name="inspect",
        identity=_identity("vision"),
    )
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"frame")

    with patch(
        "clipper.providers.modal.importlib.import_module",
        return_value=modal,
    ):
        runtime = provider.warm()
        result = provider.inspect(
            task="source_policy_visual_scout",
            frames=[frame],
            context={},
        )
        provider.inspect(
            task="source_policy_visual_scout",
            frames=[frame],
            context={},
        )

    assert runtime["worker_lifecycle_id"] == "worker-a"
    assert modal.Cls.from_name.call_count == 1
    class_handle.assert_called_once_with()
    assert inspect.remote.call_count == 2
    assert result.usage.runtime["worker_lifecycle_id"] == "worker-a"
    assert result.usage.runtime["peak_vram_mb_by_device"] == {
        "0": 10.0,
        "1": 11.0,
    }


def test_modal_editorial_provider_can_use_persistent_class_method() -> None:
    complete = SimpleNamespace(
        remote=Mock(
            return_value={
                "value": {"ok": True},
                "usage": {},
                "runtime": {
                    "worker_lifecycle_id": "editor-a",
                    "model_load_count": 1,
                },
            }
        )
    )
    instance = SimpleNamespace(complete=complete)
    class_handle = Mock(return_value=instance)
    modal = SimpleNamespace(Cls=SimpleNamespace(from_name=Mock(return_value=class_handle)))
    provider = ModalEditorialProvider(
        app_name="app",
        class_name="EditorialModel",
        method_name="complete",
        identity=_identity("editor"),
    )
    with patch(
        "clipper.providers.modal.importlib.import_module",
        return_value=modal,
    ):
        assert provider.complete_json(task="task", payload={}).value == {"ok": True}
        assert provider.complete_json(task="task", payload={}).value == {"ok": True}
    assert modal.Cls.from_name.call_count == 1
    assert complete.remote.call_count == 2


def test_modal_worker_source_uses_enter_loaded_classes_and_dynamic_capacity() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    visual_source = Path("src/clipper/visual_ai.py").read_text(encoding="utf-8")
    assert "class EditorialModel:" in source
    assert "class VisionModel:" in source
    assert "class VisionModelLarge:" in source
    assert "modal.parameter" not in source
    assert source.count("@modal.enter()") >= 3
    assert "def vision(" not in source
    assert "def vision_large(" not in source
    assert '"20GiB"' not in source
    assert '"22GiB"' not in source
    assert "SOURCE_POLICY_BATCH_SIZE" not in visual_source
    assert "_is_vision_capacity_error" in visual_source
    assert "checkpoint_commit" in visual_source
