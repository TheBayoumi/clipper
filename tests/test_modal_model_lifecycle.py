from __future__ import annotations

import ast
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
                "model": {"model_id": "vision", "revision": "requested", "quantization": "none"},
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
                "model": {"model_id": "vision", "revision": "requested", "quantization": "none"},
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
                "model": _identity("editor").to_dict(),
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
    assert "from clipper.providers.speech_contract import (" in source
    assert "ASR_MODEL_ID," in source
    assert "ASR_MODEL_REVISION," in source
    assert "ASR_COMPUTE_TYPE," in source
    assert "revision=ASR_MODEL_REVISION" in source
    assert "compute_type=ASR_COMPUTE_TYPE" in source
    assert '"quantization": ASR_COMPUTE_TYPE' in source
    assert "ALIGNMENT_MODEL_ID," in source
    assert "ALIGNMENT_MODEL_REVISION," in source
    assert "ALIGNMENT_QUANTIZATION," in source
    assert "snapshot_download(" in source
    assert "repo_id=ALIGNMENT_MODEL_ID" in source
    assert "revision=ALIGNMENT_MODEL_REVISION" in source
    assert "model_name=alignment_snapshot" in source
    assert "model_cache_only=True" in source
    assert '"quantization": ALIGNMENT_QUANTIZATION' in source
    assert '"20GiB"' not in source
    assert '"22GiB"' not in source
    assert "SOURCE_POLICY_BATCH_SIZE" not in visual_source
    assert "_is_vision_capacity_error" in visual_source
    assert "checkpoint_commit" in visual_source


def test_modal_worker_vision_task_and_model_contracts_are_distinct() -> None:
    source = Path("scripts/modal_open_models.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = [
        node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    duplicates = sorted({name for name in names if names.count(name) > 1})

    assert duplicates == []
    assert "def _vision_task_contract(task: str) -> str:" in source
    assert source.count("def _vision_contract(") == 1
    prompt_start = source.index("def _vision_prompt(")
    prompt_end = source.index("\n\nclass VisionOutputCapacityError", prompt_start)
    prompt_source = source[prompt_start:prompt_end]
    assert "_vision_task_contract(task)" in prompt_source
    assert "_vision_contract(task)" not in prompt_source
