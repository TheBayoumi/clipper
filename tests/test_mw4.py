import argparse
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from clipper import mw4


def test_mw4_bridge_lazy_loads_installed_orchestrator() -> None:
    args = argparse.Namespace(mw4_command="self-test")
    runner = Mock(return_value=7)
    module = SimpleNamespace(run=runner)

    with patch("clipper.mw4.importlib.import_module", return_value=module) as import_module:
        assert mw4.run(args) == 7

    import_module.assert_called_once_with("clipper_mw4.orchestrator")
    runner.assert_called_once_with(args)


def test_mw4_bridge_rejects_invalid_runtime() -> None:
    module = SimpleNamespace(run=None)
    with (
        patch("clipper.mw4.importlib.import_module", return_value=module),
        pytest.raises(RuntimeError, match="callable orchestrator"),
    ):
        mw4.run(argparse.Namespace(mw4_command="self-test"))
