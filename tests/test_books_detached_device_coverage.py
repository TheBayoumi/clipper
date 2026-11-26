"""Edge paths of the detached launcher CLI, Volume cleanup and CUDA device selection."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from test_books_private_modal_detached import _setup

from clipper.books import constrained, private_modal, private_phase, scripting
from clipper.books import private_modal_detached as det


def test_delete_volume_failure_is_journaled_and_reported(tmp_path, monkeypatch):
    ctx = _setup(tmp_path, monkeypatch)
    ctx["modal"].objects.delete = Mock(side_effect=RuntimeError("boom"))
    journal = Mock()
    assert det.delete_volume(ctx["modal"], ctx["client"], "vol", journal) is False
    assert journal.record.call_args.args[0] == "VOLUME_DELETE_FAILED"


def test_start_uploads_prior_checkpoint_to_the_volume(tmp_path, monkeypatch):
    ctx = _setup(tmp_path, monkeypatch)
    real = private_phase.preflight
    monkeypatch.setattr(private_phase, "preflight", lambda m, p, _c: real(m, p, None))
    result = det.start(
        ctx["manifest"],
        ctx["packet"],
        b"prior-checkpoint",
        approved_manifest_sha256=ctx["identity"],
        execute=True,
        checkout=tmp_path,
        journal_root=tmp_path,
    )
    assert result["status"] != "PRIVATE_PHASE_BLOCKED_NO_RETRY"
    assert ctx["modal"].store[private_modal.volume_name(result_run(ctx))]["prior.bin"] == (
        b"prior-checkpoint"
    )


def result_run(ctx):
    run, _, _ = private_phase.preflight(ctx["manifest"], ctx["packet"], None)
    return run


def _cli_files(tmp_path: Path, ctx) -> list[str]:
    import json

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(ctx["manifest"]))
    packet = tmp_path / "packet.bin"
    packet.write_bytes(ctx["packet"])
    return [
        "--manifest",
        str(manifest),
        "--journal-root",
        str(tmp_path),
        "--approved-manifest-sha256",
        ctx["identity"],
    ]


def test_main_start_and_poll_commands(tmp_path, monkeypatch, capsys):
    ctx = _setup(tmp_path, monkeypatch)
    common = _cli_files(tmp_path, ctx)
    start = Mock(return_value={"status": "DETACHED_STARTED"})
    poll = Mock(return_value={"status": "BRIEF_READY_UNREVIEWED"})
    monkeypatch.setattr(det, "start", start)
    monkeypatch.setattr(det, "poll", poll)
    assert (
        det.main(
            [
                "start",
                *common,
                "--packet",
                str(tmp_path / "packet.bin"),
                "--checkout",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert start.call_args.kwargs["execute"] is False
    assert start.call_args.args[2] is None
    prior = tmp_path / "prior.bin"
    prior.write_bytes(b"prior")
    assert (
        det.main(
            [
                "start",
                *common,
                "--packet",
                str(tmp_path / "packet.bin"),
                "--checkout",
                str(tmp_path),
                "--checkpoint",
                str(prior),
                "--execute",
            ]
        )
        == 0
    )
    assert start.call_args.args[2] == b"prior"
    assert start.call_args.kwargs["execute"] is True
    assert det.main(["poll", *common, "--output", str(tmp_path / "out.bin")]) == 0
    assert poll.call_args.kwargs["output"] == tmp_path / "out.bin"
    capsys.readouterr()
    poll.return_value = {"status": "PRIVATE_PHASE_BLOCKED_NO_RETRY"}
    assert det.main(["poll", *common, "--output", str(tmp_path / "out.bin")]) == 2
    poll.side_effect = RuntimeError("synthetic")
    assert det.main(["poll", *common, "--output", str(tmp_path / "out.bin")]) == 2
    assert "PRIVATE_PHASE_BLOCKED_NO_RETRY" in capsys.readouterr().out


def test_module_entry_point_blocks_without_a_readable_manifest(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "x",
            "poll",
            "--manifest",
            str(tmp_path / "missing.json"),
            "--journal-root",
            str(tmp_path),
            "--approved-manifest-sha256",
            "0" * 64,
            "--output",
            str(tmp_path / "o"),
        ],
    )
    monkeypatch.delitem(sys.modules, "clipper.books.private_modal_detached", raising=False)
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("clipper.books.private_modal_detached", run_name="__main__")
    assert raised.value.code == 2
    assert "PRIVATE_PHASE_BLOCKED_NO_RETRY" in capsys.readouterr().out


def test_remote_phase_without_checkpoint_returns_report_only(monkeypatch):
    report = {"status": "BLOCKED"}
    monkeypatch.setattr(private_modal, "_execute_in_container", lambda *_: (report, None))
    assert private_modal._remote_phase({}, b"", None) == (report, None)


def _fake_runtime(tmp_path: Path, *, cuda_available: bool):
    model = SimpleNamespace()
    model.eval = Mock(return_value=model)
    model.to = Mock(return_value="moved-model")
    tokenizer = SimpleNamespace()
    torch = SimpleNamespace(
        set_num_threads=Mock(), cuda=SimpleNamespace(is_available=lambda: cuda_available)
    )
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    snapshot = (
        tmp_path / "hub" / "models--Qwen--Qwen3-1.7B" / "snapshots" / scripting.SCRIPT_REVISION
    )
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors"):
        (snapshot / name).write_bytes(b"synthetic")
    hub = SimpleNamespace(snapshot_download=Mock(return_value=str(snapshot)))
    modules = {"torch": torch, "transformers": transformers, "huggingface_hub": hub}
    return model, modules


@pytest.mark.parametrize("available", [True, False])
def test_script_pipeline_moves_model_to_cuda_or_fails_closed(tmp_path, monkeypatch, available):
    model, modules = _fake_runtime(tmp_path, cuda_available=available)
    monkeypatch.setenv("CLIPPER_QWEN_DEVICE", "cuda")
    monkeypatch.setattr(scripting, "importlib", SimpleNamespace(import_module=modules.__getitem__))
    scripting._script_pipeline.cache_clear()
    try:
        if available:
            loaded, _tokenizer = scripting._script_pipeline()
            assert loaded == "moved-model"
            model.to.assert_called_once_with("cuda")
        else:
            with pytest.raises(scripting.NicheVideoError, match="CUDA"):
                scripting._script_pipeline()
            model.to.assert_not_called()
    finally:
        scripting._script_pipeline.cache_clear()


def test_local_book_generate_rejects_model_on_the_wrong_device(monkeypatch):
    for name in ("decode_schema", "_decoder", "check_prompt_context"):
        monkeypatch.setattr(constrained, name, Mock())
    monkeypatch.setattr(scripting, "_public_book_context", lambda: (Mock(), 40960))
    monkeypatch.setattr(
        scripting, "_script_pipeline", lambda: (SimpleNamespace(device="cpu"), None)
    )
    monkeypatch.setenv("CLIPPER_QWEN_DEVICE", "cuda")
    with pytest.raises(scripting.NicheVideoError, match="configured device"):
        scripting._local_book_generate({"messages": []}, capture_receipt=False)


def _result_ready_with_unreadable_volume(tmp_path, monkeypatch):
    from test_books_private_modal_detached import _raw, _report, _start

    ctx = _setup(tmp_path, monkeypatch)
    _start(tmp_path, ctx, now=lambda: 1000.0)
    raw = _raw()
    name = private_modal.volume_name(ctx["manifest"])
    ctx["modal"].store[name]["checkpoint.bin"] = private_modal.compress_checkpoint(raw)
    get = Mock(return_value=(_report(ctx, raw), None))
    real_from_name = ctx["modal"].Volume.from_name.side_effect
    ctx["modal"].Volume.from_name.side_effect = RuntimeError("transient read failure")
    return ctx, raw, name, get, real_from_name


def test_poll_keeps_volume_and_app_when_checkpoint_read_fails_before_deadline(
    tmp_path, monkeypatch
):
    from test_books_private_modal_detached import _poll

    ctx, raw, name, get, real_from_name = _result_ready_with_unreadable_volume(
        tmp_path, monkeypatch
    )
    result, out = _poll(tmp_path, ctx, monkeypatch, get)
    assert result["status"] == "DETACHED_PENDING"
    assert name in ctx["modal"].store and not out.exists()
    ctx["cleanup"].assert_not_called()
    ctx["modal"].Volume.from_name.side_effect = real_from_name
    again, out = _poll(tmp_path, ctx, monkeypatch, get)
    assert again["status"] == "BRIEF_READY_UNREVIEWED" and out.read_bytes() == raw
    assert ctx["modal"].store == {}


def test_poll_gives_up_and_cleans_when_checkpoint_read_fails_after_deadline(tmp_path, monkeypatch):
    from test_books_private_modal_detached import _poll

    ctx, _raw_bytes, _name, get, _real = _result_ready_with_unreadable_volume(tmp_path, monkeypatch)
    result, out = _poll(tmp_path, ctx, monkeypatch, get, now=10_000.0)
    assert result["status"] == "PRIVATE_PHASE_BLOCKED_NO_RETRY"
    assert not out.exists()
    ctx["cleanup"].assert_called_once()
    assert ctx["modal"].store == {}
