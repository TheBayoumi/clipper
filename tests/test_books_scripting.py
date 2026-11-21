"""Canonical package Qwen3 script contract checks without loading weights."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from clipper.books import scripting


def _config(tmp_path: Path) -> scripting.NicheConfig:
    path = tmp_path / "niche.yml"
    path.write_text(
        "niche: Books\naudience: Adults\nbrand: PAGE TO LIFE\ntopic: Tiny habits\n"
        "format: book_lesson\npillar: self_improvement\nbook_title: Atomic Habits\n"
        "book_author: James Clear\nsource_url: https://jamesclear.com/three-steps-habit-change\n"
        "facts:\n  - A smaller beginning can be easier to repeat.\n",
        encoding="utf-8",
    )
    return scripting.NicheConfig.load(path)


def _episode() -> dict[str, object]:
    return {
        "title": "Small starts",
        "scenes": [
            {
                "headline": f"Step {n}",
                "narration": (
                    "Make the next beginning small enough to try today. "
                    "Then notice which simple action you can repeat tomorrow."
                ),
            }
            for n in range(4)
        ],
    }


def test_qwen_contract_and_content_cache_are_stable(tmp_path: Path) -> None:
    config = _config(tmp_path)
    request = scripting._script_request(config)
    assert request["model"] == scripting.SCRIPT_MODEL
    assert request["revision"] == scripting.SCRIPT_REVISION
    fingerprint = scripting._script_fingerprint(config)
    assert len(fingerprint) == 64

    fake = Mock(return_value=json.dumps(_episode()))
    first, first_key = scripting.generate_episode(config, tmp_path, completion=fake)
    second, second_key = scripting.generate_episode(config, tmp_path, completion=fake)
    assert first == second
    assert first_key == second_key == fingerprint
    fake.assert_called_once()
    assert json.loads((tmp_path / "scripts" / f"{fingerprint}.json").read_text()) == first.to_dict()


def test_qwen_repair_audit_is_bounded_and_preserves_context(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rejected = json.dumps({"title": "bad", "scenes": []})
    fake = Mock(side_effect=[rejected, json.dumps(_episode())])
    episode, key = scripting.generate_episode(config, tmp_path, completion=fake)
    assert episode.title == "Small starts"
    assert fake.call_count == 2
    assert "Tiny habits" in str(fake.call_args_list[-1].args[0])
    audit = json.loads((tmp_path / "script_diagnostics" / f"{key}.json").read_text())
    assert [record["accepted"] for record in audit] == [False, True]


def test_qwen_invalid_cache_and_recovery_reject_source_mismatch(tmp_path: Path) -> None:
    config = _config(tmp_path)
    cache = tmp_path / "cache"
    with pytest.raises(scripting.NicheVideoError, match="recovery requires"):
        scripting._recover_book_script(config, cache, tmp_path / "missing.json", "wrong")
    fingerprint = scripting._script_fingerprint(config)
    path = cache / "scripts" / f"{fingerprint}.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(scripting.NicheVideoError, match="episode schema mismatch"):
        scripting.generate_episode(config, cache, completion=Mock())


def test_fake_optional_qwen_loader_and_completion_preserve_pinned_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the real adapter with synthetic module objects, never weights."""
    tokenized = SimpleNamespace(shape=(1, 2))
    tokenizer = SimpleNamespace(
        apply_chat_template=Mock(return_value={"input_ids": tokenized}),
        eos_token_id=0,
        decode=Mock(return_value="```json\n" + json.dumps(_episode()) + "\n```"),
    )
    model = SimpleNamespace(generate=Mock(return_value=[[10, 20, 30, 40]]))
    model.eval = Mock(return_value=model)
    torch = SimpleNamespace(
        set_num_threads=Mock(),
        manual_seed=Mock(),
        inference_mode=nullcontext,
    )
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    loaded: list[str] = []

    def fake_import(name: str) -> object:
        loaded.append(name)
        return {"torch": torch, "transformers": transformers}[name]

    monkeypatch.setattr(scripting, "importlib", SimpleNamespace(import_module=fake_import))
    scripting._script_pipeline.cache_clear()
    try:
        config = _config(tmp_path)
        result = scripting._local_script_completion(scripting._script_request(config))
        assert json.loads(result) == _episode()
        script_model, script_tokenizer = scripting._script_pipeline()
        assert script_model is model and script_tokenizer is tokenizer
        assert transformers.AutoModelForCausalLM.from_pretrained.call_count == 1
        assert transformers.AutoTokenizer.from_pretrained.call_args.kwargs["revision"] == (
            scripting.SCRIPT_REVISION
        )
        assert loaded.count("transformers") == 1
        torch.manual_seed.assert_called_once_with(42)
        assert model.generate.call_args.kwargs["max_new_tokens"] == 960
    finally:
        scripting._script_pipeline.cache_clear()


def test_missing_optional_qwen_runtime_fails_without_loading_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(_name: str) -> object:
        raise ImportError("synthetic missing optional dependency")

    monkeypatch.setattr(scripting, "importlib", SimpleNamespace(import_module=missing))
    scripting._script_pipeline.cache_clear()
    try:
        with pytest.raises(scripting.NicheVideoError, match="niche-models"):
            scripting._script_pipeline()
    finally:
        scripting._script_pipeline.cache_clear()
