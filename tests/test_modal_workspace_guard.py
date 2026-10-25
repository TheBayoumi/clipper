"""Protect deployment and Function.from_name from stale workspace routing."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / "scripts" / "modal_workspace_guard.py"


def _check(
    monkeypatch: pytest.MonkeyPatch,
    *,
    active_workspace: str,
    expected_workspace: str,
    environment: str = "main",
    token_id: str = "ak-synthetic",
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "modal",
        SimpleNamespace(
            Workspace=SimpleNamespace(
                from_context=lambda: SimpleNamespace(name=lambda: active_workspace)
            )
        ),
    )
    monkeypatch.setenv("MODAL_TOKEN_ID", token_id)
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "as-synthetic")
    monkeypatch.setenv("MODAL_ENVIRONMENT", environment)
    monkeypatch.setattr(sys, "argv", [str(GUARD), "--expected", expected_workspace])
    runpy.run_path(str(GUARD), run_name="__main__")


def test_accepts_live_expected_backup(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    _check(
        monkeypatch,
        active_workspace="mahmoud-bayoumi-bb",
        expected_workspace="mahmoud-bayoumi-bb",
    )
    assert "MODAL_WORKSPACE_CONFIRMED name=mahmoud-bayoumi-bb environment=main" in (
        capsys.readouterr().out
    )


@pytest.mark.parametrize(
    ("actual", "expected", "environment"),
    (
        ("mahmoud-bayoumi-mb", "mahmoud-bayoumi-bb", "main"),
        ("mahmoud-bayoumi-bb", "mahmoud-bayoumi-bb", "dev"),
        ("mahmoud-bayoumi-bb", "thebayoumi", "main"),
    ),
)
def test_fail_closed_for_wrong_account_or_environment(
    monkeypatch: pytest.MonkeyPatch,
    actual: str,
    expected: str,
    environment: str,
) -> None:
    with pytest.raises(SystemExit):
        _check(
            monkeypatch,
            active_workspace=actual,
            expected_workspace=expected,
            environment=environment,
        )


def test_cinematic_function_lookup_uses_names_not_persisted_ids() -> None:
    for filename in ("modal_deployed_client.py", "modal_deployment_probe.py"):
        source = (ROOT / "scripts" / filename).read_text()
        assert "Function.from_name(" in source
        assert 'environment_name=os.environ.get("MODAL_ENVIRONMENT", "main")' in source
        assert "Function.from_id(" not in source

    router = (ROOT / "scripts" / "modal_versioned_dispatch.sh").read_text()
    assert "modal_workspace_guard.py" in router
    assert "export MODAL_ENVIRONMENT=main" in router
    assert "modal deploy --env main" in router
    assert "MODAL_TOKEN_ID_2" not in router or "MODAL_TOKEN_ID_${slot}" in router
