from __future__ import annotations

import ast
import importlib.util
from pathlib import Path


def _open_models_tree() -> ast.Module:
    return ast.parse(Path("scripts/modal_open_models.py").read_text(encoding="utf-8"))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(
        item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name
    )


def _event_mapping(function: ast.FunctionDef, event_name: str) -> dict[str, ast.expr]:
    matches = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Dict):
            continue
        mapping = {
            key.value: value
            for key, value in zip(node.keys, node.values, strict=True)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        event = mapping.get("event")
        if isinstance(event, ast.Constant) and event.value == event_name:
            matches.append(mapping)
    assert len(matches) == 1
    return matches[0]


def test_vision_terminal_telemetry_preserves_payload_task_identity() -> None:
    tree = _open_models_tree()
    complete = _event_mapping(_function(tree, "_vision_infer"), "vision_generation_complete")
    error = _event_mapping(_function(tree, "_vision_worker_inspect"), "vision_inference_error")
    assert ast.unparse(complete["task"]) == "str(payload.get('task') or '')"
    assert ast.unparse(error["task"]) == "str(payload.get('task') or '')"


def _spy_module():
    path = Path("scripts/modal_execution_spy.py")
    spec = importlib.util.spec_from_file_location("vision_task_contract_spy", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load modal execution spy")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_spy_closes_matching_vision_task_identity(tmp_path: Path) -> None:
    module = _spy_module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "vision-match.ndjson",
        execution_id="exec-123",
    )
    spy._record(
        "clipper-open-editor",
        "2026-09-11T13:00:00Z fc-VISION1 "
        '{"event":"vision_generation_start","execution_id":"exec-123",'
        '"worker_lifecycle_id":"vision-1","task":"source_policy_visual_scout",'
        '"attempt":1,"frames":32}',
    )
    spy._record(
        "clipper-open-editor",
        "2026-09-11T13:00:01Z fc-VISION1 "
        '{"event":"vision_generation_complete","execution_id":"exec-123",'
        '"worker_lifecycle_id":"vision-1","task":"source_policy_visual_scout",'
        '"attempt":1,"frames":32,"generated_tokens":2222}',
    )
    assert spy.abort_reason is None
    assert spy.summary()["active_vision_generations"] == []


def test_spy_still_rejects_real_vision_task_mismatch(tmp_path: Path) -> None:
    module = _spy_module()
    spy = module.ModalExecutionSpy(
        ("clipper-open-editor", "clipper-production-pipeline"),
        tmp_path / "vision-mismatch.ndjson",
        execution_id="exec-123",
    )
    spy._record(
        "clipper-open-editor",
        "2026-09-11T13:00:00Z fc-VISION1 "
        '{"event":"vision_generation_start","execution_id":"exec-123",'
        '"worker_lifecycle_id":"vision-1","task":"source_policy_visual_scout",'
        '"attempt":1,"frames":32}',
    )
    spy._record(
        "clipper-open-editor",
        "2026-09-11T13:00:01Z fc-VISION1 "
        '{"event":"vision_generation_complete","execution_id":"exec-123",'
        '"worker_lifecycle_id":"vision-1","task":"rendered_clip_review",'
        '"attempt":1,"frames":32,"generated_tokens":2222}',
    )
    assert spy.abort_reason is not None
    assert "terminal task does not match start" in spy.abort_reason
    assert len(spy.summary()["active_vision_generations"]) == 1
