from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import pytest

from clipper.books import model_profile, private_phase, profile_probe

SHA = "a" * 40


class FakeModal:
    """Records the image the factory builds; allocates nothing."""

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.options: dict[str, Any] = {}

        modal = self

        class Image:
            def __getattr__(self, name: str) -> Any:
                def step(*args: Any, **kwargs: Any) -> Image:
                    modal.calls.append((name, args))
                    return self

                return step

        class App:
            def __init__(self, **kwargs: Any) -> None:
                pass

            def function(self, **options: Any) -> Any:
                modal.options = options
                return lambda fn: ("function", fn)

        self.Image = types.SimpleNamespace(debian_slim=lambda **kw: Image())
        self.App = App
        self.Volume = None


def test_plan_only_without_execute_allocates_nothing() -> None:
    result = profile_probe.run(SHA, "qwen3.5-4b", 900, execute=False)
    assert result["status"] == "PLAN_ONLY" and result["accelerator"] == "L4"
    assert "made-up public sentence" in result["inputs"]


def test_build_uses_the_probe_entry_on_an_l4_with_a_hard_timeout() -> None:
    modal = FakeModal()
    _app, function = profile_probe.build(modal, SHA, "qwen3.5-4b", 900)
    assert function[1] is profile_probe.probe_in_container
    assert modal.options["gpu"] == "L4" and modal.options["timeout"] == 900
    assert modal.options["block_network"] is True and modal.options["retries"] == 0
    pins = [a for name, a in modal.calls if name == "pip_install"]
    assert any("transformers==5.24.1" in p for args in pins for p in args)


@pytest.mark.parametrize("timeout", [0, 59, 1801])
def test_timeout_out_of_range_is_refused(timeout: int) -> None:
    with pytest.raises(private_phase.PrivatePhaseError):
        profile_probe.build(FakeModal(), SHA, "qwen3.5-4b", timeout)


def test_bad_commit_and_unknown_profile_are_refused() -> None:
    with pytest.raises(private_phase.PrivatePhaseError):
        profile_probe.build(FakeModal(), "main", "qwen3.5-4b", 900)
    with pytest.raises(model_profile.ModelProfileError):
        profile_probe.build(FakeModal(), SHA, "nope", 900)


def test_cli_prints_plan_and_failure_without_leaking(capsys: pytest.CaptureFixture[str]) -> None:
    assert profile_probe.main(["--commit", SHA, "--profile", "qwen3.5-4b"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "PLAN_ONLY"
    assert profile_probe.main(["--commit", "bad", "--profile", "qwen3.5-4b", "--execute"]) == 2
    assert json.loads(capsys.readouterr().out)["status"] == "PROBE_FAILED"


def test_probe_in_container_reports_facts_and_hashes_the_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    from clipper.books import scripting

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            get_device_name=lambda i: "L4",
            max_memory_allocated=lambda: 2**30,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv(
        "CLIPPER_QWEN_DEVICE", "cpu"
    )  # the probe overwrites it; monkeypatch restores
    model = types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=248044))

    def tokenizer(text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
        return {"input_ids": list(range(len(text.split())))}

    class Tok:
        eos_token_id = 248046

        def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
            return tokenizer(text)

    monkeypatch.setattr(scripting, "_script_pipeline", lambda: (model, Tok()))
    monkeypatch.setattr(scripting, "_run_pinned_qwen", lambda request, **kw: "blue light scatters")
    monkeypatch.setattr(profile_probe, "version", lambda name: "1.0")
    report = profile_probe.probe_in_container("qwen3.5-4b")
    assert report["status"] == "PROBE_OK" and report["cuda"] == "L4"
    assert report["reencoded_text_tokens"] == 3 and report["peak_gpu_gib"] == 1.0
    assert report["tokenizer_eos_token_id"] == 248046
    assert report["generation_config_eos_token_id"] == 248044
    assert "blue light" not in json.dumps(report)


def test_execute_needs_a_journal_root_and_the_reviewed_workspace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(private_phase.PrivatePhaseError, match="journal root"):
        profile_probe.run(SHA, "qwen3.5-4b", 900, execute=True)
    started: list[Any] = []

    def wrong_workspace() -> Any:
        raise private_phase.PrivatePhaseError("private SDK identity unavailable")

    monkeypatch.setattr(profile_probe.private_modal, "identify", wrong_workspace)
    monkeypatch.setattr(profile_probe, "build", lambda *a: started.append(a))
    with pytest.raises(private_phase.PrivatePhaseError):
        profile_probe.run(SHA, "qwen3.5-4b", 900, execute=True, journal_root=tmp_path)
    assert started == []


def test_the_same_probe_plan_cannot_start_twice(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs: list[str] = []

    class App:
        def run(self, **kwargs: Any) -> Any:
            import contextlib

            runs.append("run")
            return contextlib.nullcontext()

    class Function:
        def remote(self, name: str) -> dict[str, Any]:
            return {"status": "PROBE_OK"}

    closed: list[Any] = []
    monkeypatch.setattr(profile_probe.private_modal, "identify", lambda: ("modal", "client"))
    monkeypatch.setattr(profile_probe.private_modal, "close_client", lambda c: closed.append(c))
    monkeypatch.setattr(profile_probe, "build", lambda *a: (App(), Function()))
    first = profile_probe.run(SHA, "qwen3.5-4b", 900, execute=True, journal_root=tmp_path)
    assert first["status"] == "PROBE_OK"
    with pytest.raises(private_phase.PrivatePhaseError, match="already started"):
        profile_probe.run(SHA, "qwen3.5-4b", 900, execute=True, journal_root=tmp_path)
    assert runs == ["run"] and closed == ["client", "client"]


def test_an_empty_generation_is_not_a_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys

    from clipper.books import scripting

    fake_torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: False,
            get_device_name=lambda i: "x",
            max_memory_allocated=lambda: 0,
        )
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setenv("CLIPPER_QWEN_DEVICE", "cpu")

    class Tok:
        eos_token_id = 1

        def __call__(self, text: str, add_special_tokens: bool = True) -> dict[str, list[int]]:
            return {"input_ids": []}

    model = types.SimpleNamespace(generation_config=types.SimpleNamespace(eos_token_id=1))
    monkeypatch.setattr(scripting, "_script_pipeline", lambda: (model, Tok()))
    monkeypatch.setattr(scripting, "_run_pinned_qwen", lambda request, **kw: "")
    monkeypatch.setattr(profile_probe, "version", lambda name: "1.0")
    report = profile_probe.probe_in_container("qwen3.5-4b")
    assert report["status"] == "PROBE_EMPTY_GENERATION"
    assert "chapter-generator" in str(report["scope"])
