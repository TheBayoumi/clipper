"""Pinned editorial endpoint identity must be checked before endpoint reuse."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import Mock, patch

import pytest

from clipper.providers.editorial_model_contract import (
    EDITORIAL_MODEL_ID,
    EDITORIAL_MODEL_REVISION,
)

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "modal_endpoint_bootstrap.py"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("verified_endpoint_bootstrap", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(*, model: str = EDITORIAL_MODEL_ID, revision: str | None) -> str:
    return json.dumps(
        [
            {
                "name": "clipper-editorial",
                "url": "https://example.invalid/model",
                "status": "ready",
                "model_id": model,
                "model_revision": revision,
            }
        ]
    )


def test_default_cli_model_is_imported_from_pinned_contract(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    module = _module()
    monkeypatch.setattr(sys, "argv", ["bootstrap"])
    with (
        patch.object(module, "create_proxy_token", return_value=("wk-fake", "ws-fake")),
        patch.object(
            module, "ensure_endpoint", return_value="https://example.invalid/model"
        ) as run,
    ):
        module.main()
    run.assert_called_once_with("clipper-editorial", EDITORIAL_MODEL_ID)
    assert json.loads(capsys.readouterr().out)["model_id"] == EDITORIAL_MODEL_ID


def test_cli_rejects_model_override_before_creating_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    monkeypatch.setattr(sys, "argv", ["bootstrap", "--model", "unapproved-model"])
    with (
        patch.object(module, "create_proxy_token") as credentials,
        pytest.raises(SystemExit, match="2"),
    ):
        module.main()
    credentials.assert_not_called()


@pytest.mark.parametrize(
    "model,revision",
    [
        (EDITORIAL_MODEL_ID, None),
        ("Qwen/unapproved", EDITORIAL_MODEL_REVISION),
        (EDITORIAL_MODEL_ID, "unapproved-revision"),
    ],
)
def test_unverified_existing_endpoint_refused_without_mutation(
    model: str, revision: str | None
) -> None:
    module = _module()
    runner = Mock(return_value=_metadata(model=model, revision=revision))
    with (
        patch.object(module, "_run", runner),
        pytest.raises(RuntimeError, match=r"metadata missing|identity differs"),
    ):
        module.ensure_endpoint("clipper-editorial", EDITORIAL_MODEL_ID)
    runner.assert_called_once_with("modal", "endpoint", "list", "--json")


def test_matching_metadata_allows_reuse_without_creating_endpoint() -> None:
    module = _module()
    runner = Mock(return_value=_metadata(revision=EDITORIAL_MODEL_REVISION))
    with patch.object(module, "_run", runner):
        result = module.ensure_endpoint("clipper-editorial", EDITORIAL_MODEL_ID)
    assert result == "https://example.invalid/model"
    assert all(
        call.args == ("modal", "endpoint", "list", "--json") for call in runner.call_args_list
    )


def test_synthetic_flat_missing_entry_has_distinct_absence_error() -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="absent from the endpoint listing"):
        module._assert_endpoint_identity([], "clipper-editorial")


def test_synthetic_nested_model_metadata_with_revision_passes_exact_identity() -> None:
    module = _module()
    payload = [
        {
            "endpoint_name": "clipper-editorial",
            "endpoint_url": "https://example.invalid/model",
            "state": "ready",
            "model": {"id": EDITORIAL_MODEL_ID, "revision": EDITORIAL_MODEL_REVISION},
        }
    ]
    runner = Mock(return_value=json.dumps(payload))
    with patch.object(module, "_run", runner):
        assert module.ensure_endpoint("clipper-editorial", EDITORIAL_MODEL_ID) == (
            "https://example.invalid/model"
        )
    assert all(
        call.args == ("modal", "endpoint", "list", "--json") for call in runner.call_args_list
    )


def test_synthetic_missing_revision_exposes_key_names_only_not_values() -> None:
    module = _module()
    sample_marker = "private-value-must-never-appear"
    payload = [
        {
            "name": "clipper-editorial",
            "url": "https://example.invalid/model",
            "model": {"id": EDITORIAL_MODEL_ID, "unknown_version": sample_marker},
            "private_token": sample_marker,
        }
    ]
    with pytest.raises(RuntimeError, match="metadata missing or unrecognized") as exc:
        module._assert_endpoint_identity(payload, "clipper-editorial")
    error = str(exc.value)
    assert "listing entry keys" in error and "nested model keys" in error
    assert "private_token" in error and "unknown_version" in error
    assert sample_marker not in error and "example.invalid" not in error


@pytest.mark.parametrize(
    "model,revision",
    [
        ("synthetic-wrong-model", EDITORIAL_MODEL_REVISION),
        (EDITORIAL_MODEL_ID, "synthetic-wrong-revision"),
    ],
)
def test_synthetic_mismatching_identity_never_mutates_endpoint(model: str, revision: str) -> None:
    module = _module()
    runner = Mock(return_value=_metadata(model=model, revision=revision))
    with (
        patch.object(module, "_run", runner),
        pytest.raises(RuntimeError, match="identity differs"),
    ):
        module.ensure_endpoint("clipper-editorial", EDITORIAL_MODEL_ID)
    runner.assert_called_once_with("modal", "endpoint", "list", "--json")


def test_synthetic_created_endpoint_with_missing_revision_is_held_no_deletion() -> None:
    module = _module()
    runner = Mock(side_effect=["[]", "", _metadata(revision=None)])
    with (
        patch.object(module, "_run", runner),
        pytest.raises(RuntimeError, match="just created but is unverified") as exc,
    ):
        module.ensure_endpoint("clipper-editorial", EDITORIAL_MODEL_ID)
    assert "must not be used" in str(exc.value)
    assert "owner decision required" in str(exc.value)
    assert "metadata missing" in str(exc.value)
    assert [call.args[:3] for call in runner.call_args_list] == [
        ("modal", "endpoint", "list"),
        ("modal", "endpoint", "create"),
        ("modal", "endpoint", "list"),
    ]
    assert not any("delete" in call.args or "stop" in call.args for call in runner.call_args_list)


def test_synthetic_created_endpoint_absent_on_poll_requires_owner_decision() -> None:
    module = _module()
    runner = Mock(side_effect=["[]", "", "[]"])
    clock = iter([0.0, 0.0, 1.0])
    with (
        patch.object(module, "_run", runner),
        patch.object(module.time, "monotonic", side_effect=lambda: next(clock)),
        patch.object(module.time, "sleep") as sleep,
        pytest.raises(RuntimeError, match="absent from the endpoint listing") as exc,
    ):
        module.ensure_endpoint("clipper-editorial", EDITORIAL_MODEL_ID, timeout=0.1)
    sleep.assert_called_once_with(5)
    assert "was just created" in str(exc.value)
    assert "must not be used" in str(exc.value)
    assert not any("delete" in call.args or "stop" in call.args for call in runner.call_args_list)
