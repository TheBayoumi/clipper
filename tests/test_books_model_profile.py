"""The model profile is the single source of model facts for the book pipeline."""

from __future__ import annotations

import dataclasses
import importlib
import json
import types
from pathlib import Path
from typing import Any

import pytest

from clipper.books import constrained, model_profile, pipeline_plan, scripting, whole_book
from clipper.books.model_profile import ModelProfile, ModelProfileError
from clipper.stage_contracts import content_fingerprint

FAKE = ModelProfile(
    name="fake-test-model",
    model="example-org/Fake-Test-Model",
    revision="a" * 40,
    tokenizer_json_sha256="b" * 64,
    context_window=8192,
    tokenizer_max_length=16384,
    safe_context_tokens=4096,
    chat_template_kwargs={"enable_thinking": True, "extra_flag": "x"},
    generation_settings={"do_sample": False, "temperature": 0.1},
    max_output_tokens=2048,
    output_margin_tokens=256,
)


def test_default_profile_reproduces_pinned_qwen_values() -> None:
    profile = model_profile.active()
    assert profile.name == model_profile.DEFAULT_PROFILE == "qwen3-1.7b"
    assert profile.model == "Qwen/Qwen3-1.7B"
    assert profile.revision == "8d4744f9e13072f4920c326350fa81eedb74eae9"
    assert profile.tokenizer_json_sha256 == (
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
    )
    assert profile.context_window == 40_960
    assert profile.tokenizer_max_length == 131_072
    assert profile.effective_context == 40_960
    assert profile.safe_context_tokens == 32_768
    assert profile.template_kwargs == {"enable_thinking": False}
    assert profile.sampling == {"do_sample": True, "temperature": 0.7, "top_p": 0.8}


def test_historical_names_follow_the_active_profile() -> None:
    assert scripting.SCRIPT_MODEL == "Qwen/Qwen3-1.7B"
    assert scripting.MAX_CHAPTER_CONTEXT == 40_960
    assert whole_book.LOCAL_GENERATION["max_new_tokens"] == 960
    with model_profile.use(FAKE):
        assert FAKE.model == scripting.SCRIPT_MODEL
        assert FAKE.revision == scripting.SCRIPT_REVISION
        assert scripting.MAX_CHAPTER_CONTEXT == 8192
        assert scripting.QWEN_GENERATION_SETTINGS == {"do_sample": False, "temperature": 0.1}
        assert constrained.PINNED_TOKENIZER_SHA256 == "b" * 64
        generation = whole_book.LOCAL_GENERATION
        assert generation["model"] == FAKE.model
        assert generation["enable_thinking"] is True
        assert generation["extra_flag"] == "x"
        assert generation["temperature"] == 0.1
    assert scripting.SCRIPT_MODEL == "Qwen/Qwen3-1.7B"
    assert model_profile.active().name == "qwen3-1.7b"


def test_stage_registry_takes_model_from_the_profile() -> None:
    default = pipeline_plan.registry()
    assert "Qwen/Qwen3-1.7B" in {spec.model for spec in default.values()}
    with model_profile.use(FAKE):
        swapped = pipeline_plan.registry()
    models = {spec.model for spec in swapped.values() if spec.model}
    revisions = {spec.revision for spec in swapped.values() if spec.revision}
    assert FAKE.model in models
    assert FAKE.revision in revisions
    assert "Qwen/Qwen3-1.7B" not in models
    assert "8d4744f9e13072f4920c326350fa81eedb74eae9" not in revisions


def test_profile_restores_after_error_and_clears_caches() -> None:
    with pytest.raises(RuntimeError), model_profile.use(FAKE):
        raise RuntimeError("boom")
    assert model_profile.active().name == "qwen3-1.7b"


def test_context_checks_use_the_profile_window() -> None:
    def request(profile: ModelProfile) -> dict[str, object]:
        return {
            "model": profile.model,
            "revision": profile.revision,
            "seed": 42,
            "messages": [{"role": "user", "content": "x"}],
            "max_new_tokens": 2048,
        }

    class Tokenizer:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def apply_chat_template(self, messages: object, **kwargs: object) -> list[int]:
            self.kwargs = kwargs
            return [1] * 5000

    tokenizer = Tokenizer()
    with model_profile.use(FAKE):
        with pytest.raises(scripting.NicheVideoError):
            scripting._chapter_context_tokens(
                request(FAKE), tokenizer_context=(tokenizer, FAKE.effective_context - 6000)
            )
        assert tokenizer.kwargs["enable_thinking"] is True
        assert tokenizer.kwargs["extra_flag"] == "x"
        assert (
            scripting._chapter_context_tokens(
                request(FAKE), tokenizer_context=(tokenizer, FAKE.effective_context)
            )
            == 5000
        )
    with pytest.raises(scripting.NicheVideoError):
        scripting._chapter_context_tokens(request(FAKE), tokenizer_context=(tokenizer, 40_960))


@pytest.mark.parametrize(
    "change",
    [
        {"name": "Bad Name"},
        {"model": "no-slash"},
        {"revision": "main"},
        {"tokenizer_json_sha256": "short"},
        {"context_window": 10},
        {"tokenizer_max_length": 0},
        {"safe_context_tokens": 100},
        {"safe_context_tokens": 9000},
    ],
)
def test_invalid_profiles_are_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ModelProfileError):
        dataclasses.replace(FAKE, **change)


def test_clear_model_caches_is_callable_and_idempotent() -> None:
    scripting.clear_model_caches()
    scripting.clear_model_caches()


def test_registry_lookup_and_conflicting_registration() -> None:
    assert model_profile.get("qwen3-1.7b") is model_profile.QWEN3_1_7B
    with pytest.raises(ModelProfileError):
        model_profile.get("missing")
    with pytest.raises(ModelProfileError):
        model_profile.register(
            dataclasses.replace(model_profile.QWEN3_1_7B, safe_context_tokens=16384)
        )
    model_profile.register(model_profile.QWEN3_1_7B)
    assert "qwen3-1.7b" in model_profile.profiles()


def test_profile_is_immutable() -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        model_profile.QWEN3_1_7B.model = "other/model"  # type: ignore[misc]
    with pytest.raises(TypeError):
        model_profile.QWEN3_1_7B.chat_template_kwargs["x"] = 1  # type: ignore[index]


@pytest.fixture
def fake_registered(monkeypatch: pytest.MonkeyPatch) -> ModelProfile:
    monkeypatch.setitem(model_profile._PROFILES, FAKE.name, FAKE)
    return FAKE


def test_config_selects_profile_and_default_is_unchanged(fake_registered: ModelProfile) -> None:
    digest = "c" * 64
    default = pipeline_plan.template(digest)
    assert "model_profile" not in default
    assert pipeline_plan.config_profile_name(default) == "qwen3-1.7b"
    chosen = pipeline_plan.template(digest, "script", FAKE.name)
    assert chosen["model_profile"] == FAKE.name
    with pipeline_plan.profile_scope(chosen):
        assert model_profile.active() is FAKE
        validated = pipeline_plan.validate_config(chosen, pipeline_plan.registry())
        assert validated["model_profile"] == FAKE.name
        models = {s.model for s in pipeline_plan.registry().values() if s.model}
        assert FAKE.model in models
        assert "Qwen/Qwen3-1.7B" not in models
    assert model_profile.active().name == "qwen3-1.7b"
    assert content_fingerprint(chosen) != content_fingerprint(default)


def test_unknown_profile_in_config_is_rejected(fake_registered: ModelProfile) -> None:
    config = pipeline_plan.template("c" * 64)
    config["model_profile"] = "no-such-profile"
    with pytest.raises(pipeline_plan.PlanError):
        pipeline_plan.validate_config(config, pipeline_plan.registry())
    with pipeline_plan.profile_scope(config):
        assert model_profile.active().name == "qwen3-1.7b"
    with pytest.raises(pipeline_plan.PlanError):
        pipeline_plan.template("c" * 64, "script", "no-such-profile")
    config["model_profile"] = 7
    with pytest.raises(pipeline_plan.PlanError):
        pipeline_plan.config_profile_name(config)


def test_chapter_evaluation_and_private_phase_run_under_the_configured_profile(
    fake_registered: ModelProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    from clipper.books import chapter, private_phase

    seen: list[str] = []

    def fake_inner(*args: object, **kwargs: object) -> dict[str, object]:
        seen.append(model_profile.active().name)
        return {"ok": True}

    monkeypatch.setattr(chapter, "_evaluate_hybrid_chapter", fake_inner)
    config = pipeline_plan.template("c" * 64, "script", FAKE.name)
    assert chapter.evaluate_hybrid_chapter(pipeline_config=config) == {"ok": True}
    assert chapter.evaluate_hybrid_chapter() == {"ok": True}
    assert seen == [FAKE.name, "qwen3-1.7b"]

    packet = private_phase.pack({"config.json": json.dumps(config).encode()})
    with private_phase._packet_profile_scope(packet):
        assert model_profile.active() is FAKE
    with private_phase._packet_profile_scope(b"not a packet"):
        assert model_profile.active().name == "qwen3-1.7b"
    assert model_profile.active().name == "qwen3-1.7b"


def test_active_profile_is_isolated_per_thread_and_restores_when_nested() -> None:
    import threading

    seen: dict[str, str] = {}
    ready = threading.Event()
    release = threading.Event()

    def worker() -> None:
        with model_profile.use(FAKE):
            seen["worker"] = model_profile.active().name
            ready.set()
            release.wait(5)
            seen["worker_after"] = model_profile.active().name

    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(5)
    seen["main"] = model_profile.active().name
    release.set()
    thread.join(5)
    assert seen == {
        "worker": FAKE.name,
        "main": "qwen3-1.7b",
        "worker_after": FAKE.name,
    }
    other = dataclasses.replace(FAKE, name="fake-nested")
    with model_profile.use(FAKE), model_profile.use(other):
        assert model_profile.active() is other
        with model_profile.use(model_profile.QWEN3_1_7B):
            assert model_profile.active().name == "qwen3-1.7b"
        assert model_profile.active() is other
    assert model_profile.active().name == "qwen3-1.7b"


def test_fingerprint_changes_with_every_fact_and_keys_caches() -> None:
    base = model_profile.QWEN3_1_7B.fingerprint
    assert base == model_profile.QWEN3_1_7B.fingerprint
    for change in (
        {"name": "other"},
        {"revision": "c" * 40},
        {"context_window": 50_000},
        {"generation_settings": {"temperature": 0.2}},
        {"chat_template_kwargs": {"enable_thinking": True}},
        {"safe_context_tokens": 2048, "max_output_tokens": 1024},
        {"max_output_tokens": 8192},
        {"output_margin_tokens": 128},
        {"min_output_tokens": 512},
    ):
        assert dataclasses.replace(model_profile.QWEN3_1_7B, **change).fingerprint != base

    calls: list[str] = []

    def counting(fingerprint: str, local_files_only: bool) -> tuple[object, int]:
        calls.append(fingerprint)
        return object(), 1

    original = scripting._public_book_context_cached
    scripting._public_book_context_cached = scripting.lru_cache(maxsize=4)(counting)  # type: ignore[assignment]
    try:
        scripting._public_book_context()
        scripting._public_book_context()
        with model_profile.use(FAKE):
            scripting._public_book_context()
        assert calls == [base, FAKE.fingerprint]
    finally:
        scripting._public_book_context_cached = original


def test_modal_phase_follows_the_profile_the_packet_names(fake_registered: ModelProfile) -> None:
    from clipper.books import private_phase

    default = private_phase.pack(
        {"config.json": json.dumps(pipeline_plan.template("c" * 64)).encode()}
    )
    assert private_phase.packet_profile(default).name == "qwen3-1.7b"
    chosen = private_phase.pack(
        {"config.json": json.dumps(pipeline_plan.template("c" * 64, "script", FAKE.name)).encode()}
    )
    assert private_phase.packet_profile(chosen).name == FAKE.name
    unknown = private_phase.pack(
        {
            "config.json": json.dumps(pipeline_plan.template("c" * 64, "script", "qwen3.5-4b"))
            .encode()
            .replace(b"qwen3.5-4b", b"nope")
        }
    )
    with pytest.raises(private_phase.PrivatePhaseError, match="unknown model profile"):
        private_phase.packet_profile(unknown)
    with pytest.raises(private_phase.PrivatePhaseError, match="unreadable"):
        private_phase.packet_profile(private_phase.pack({"config.json": b"not json"}))
    assert model_profile.active().name == "qwen3-1.7b"


def test_auto_output_budget_uses_measured_prompt_and_ceiling() -> None:
    qwen = model_profile.QWEN3_1_7B
    assert (qwen.max_output_tokens, qwen.output_margin_tokens, qwen.min_output_tokens) == (
        16_384,
        512,
        1_024,
    )
    assert qwen.auto_output_tokens(7_700) == 16_384  # ceiling binds
    assert qwen.auto_output_tokens(20_000) == 32_768 - 20_000 - 512
    assert qwen.auto_output_tokens(32_768 - 512 - 1_024) == 1_024  # exactly the minimum
    for prompt in (32_768 - 512 - 1_023, 40_000):
        with pytest.raises(ModelProfileError):
            qwen.auto_output_tokens(prompt)
    for bad in (0, -1, True, 7.5):
        with pytest.raises(ModelProfileError):
            qwen.auto_output_tokens(bad)  # type: ignore[arg-type]
    assert FAKE.auto_output_tokens(1_000) == 2_048
    assert FAKE.auto_output_tokens(2_000) == 1_840
    with pytest.raises(ModelProfileError):
        FAKE.auto_output_tokens(3_000)


@pytest.mark.parametrize(
    "change",
    [
        {"max_output_tokens": 4096},  # not below the safe context
        {"max_output_tokens": 512},  # below the minimum
        {"min_output_tokens": 0},
        {"output_margin_tokens": -1},
        {"output_margin_tokens": 4096},
        {"max_output_tokens": "2048"},
    ],
)
def test_invalid_output_budget_is_rejected(change: dict[str, object]) -> None:
    with pytest.raises(ModelProfileError):
        dataclasses.replace(FAKE, **change)


def test_output_budget_is_part_of_the_fingerprint_and_bounds() -> None:
    assert dataclasses.replace(FAKE, output_margin_tokens=128).fingerprint != FAKE.fingerprint
    assert model_profile.output_bounds() == (1_024, 16_384)
    with model_profile.use(FAKE):
        assert model_profile.output_bounds() == (1_024, 2_048)


def test_config_accepts_auto_or_bounded_int_only() -> None:
    config = pipeline_plan.template("c" * 64)
    base = pipeline_plan.validate_config(config, pipeline_plan.registry())
    assert base["stages"]["argument"]["semantic"]["max_new_tokens"] == 4096
    for good in ("auto", 1024, 16_384):
        config["stages"]["argument"]["semantic"]["max_new_tokens"] = good
        pipeline_plan.validate_config(config, pipeline_plan.registry())
    for bad in ("AUTO", "", 1023, 16_385, True, None, 4096.0):
        config["stages"]["argument"]["semantic"]["max_new_tokens"] = bad
        with pytest.raises(pipeline_plan.PlanError):
            pipeline_plan.validate_config(config, pipeline_plan.registry())


def test_hybrid_request_resolves_auto_budget_before_hashing() -> None:
    from clipper.books import chapter_hybrid as hybrid

    payload = {"source": {"id": "x"}}
    identity = {"schema": "runtime-v1", "runtime": "CPU", "producer": "a" * 64}

    def build(prompt: int, *, auto: bool = True) -> tuple[dict[str, object], str, int]:
        return hybrid.request(
            "argument",
            payload,
            output_tokens=16_384,
            token_count=lambda _messages: prompt,
            producer_identity=identity,
            auto_output=auto,
        )

    short, short_key, _ = build(7_700)
    assert short["max_new_tokens"] == 16_384
    long_, long_key, _ = build(20_000)
    assert long_["max_new_tokens"] == 32_768 - 20_000 - 512
    assert short_key != long_key  # the resolved budget is inside the request identity
    assert build(20_000) == build(20_000)  # deterministic
    fixed, fixed_key, _ = hybrid.request(
        "argument",
        payload,
        output_tokens=4_096,
        token_count=lambda _messages: 20_000,
        producer_identity=identity,
    )
    assert fixed["max_new_tokens"] == 4_096
    assert fixed_key != long_key
    with pytest.raises(hybrid.HybridStop):
        build(32_768 - 512 - 1_023)


# Qwen3.5-4B profile (model-free) ------------------------------------------------------


def test_qwen35_profile_facts_and_registration() -> None:
    p = model_profile.get("qwen3.5-4b")
    assert p is model_profile.QWEN3_5_4B
    assert p.model == "Qwen/Qwen3.5-4B"
    assert p.revision == "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    assert p.tokenizer_json_sha256.startswith("5f9e4d49")
    assert (p.context_window, p.tokenizer_max_length) == (262_144, 262_144)
    assert p.safe_context_tokens == 32_768 and p.template_kwargs == {"enable_thinking": False}
    assert p.model_class == "AutoModelForImageTextToText"
    assert p.transformers_requirement == "5.24.1"
    assert "presence_penalty" not in p.sampling  # generate() does not implement it
    assert model_profile.DEFAULT_PROFILE == "qwen3-1.7b"


def test_default_profile_fingerprint_is_unchanged_by_the_new_runtime_fields() -> None:
    import hashlib

    p = model_profile.QWEN3_1_7B
    material = {
        "name": p.name,
        "model": p.model,
        "revision": p.revision,
        "tokenizer_json_sha256": p.tokenizer_json_sha256,
        "context_window": p.context_window,
        "tokenizer_max_length": p.tokenizer_max_length,
        "safe_context_tokens": p.safe_context_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
        "generation_settings": {"do_sample": True, "temperature": 0.7, "top_p": 0.8},
        "max_output_tokens": 16_384,
        "output_margin_tokens": 512,
        "min_output_tokens": 1_024,
    }
    expected = hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()
    assert p.fingerprint == expected
    # Constant computed from the parent commit 8a2b6e40, independent of current attributes.
    assert p.fingerprint == "0308400868be065aeca837ad8d6268f15253be21f136b3d9d40f406fcdad2b81"
    assert model_profile.QWEN3_5_4B.fingerprint != expected


def test_runtime_fields_are_validated_and_enter_the_fingerprint() -> None:
    import dataclasses

    base = model_profile.QWEN3_5_4B
    for bad in (
        {"transformers_requirement": "latest"},
        {"huggingface_hub_requirement": ">=1"},
        {"model_class": "AutoModel"},
        {"tokenizer_files": ()},
        {"tokenizer_files": ("config.json", "tokenizer.json")},
        {"tokenizer_files": ("../x", "config.json", "tokenizer.json", "tokenizer_config.json")},
    ):
        with pytest.raises(model_profile.ModelProfileError):
            dataclasses.replace(base, **bad)
    other = dataclasses.replace(base, transformers_requirement="5.24.0")
    assert other.fingerprint != base.fingerprint


def test_public_context_preflight_reads_nested_text_limits_and_profile_file_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib
    import importlib
    import types

    tok = b'{"fake": "tokenizer"}'
    profile = dataclasses.replace(
        model_profile.QWEN3_5_4B,
        name="qwen35-nested-test",
        tokenizer_json_sha256=hashlib.sha256(tok).hexdigest(),
    )
    snap = tmp_path / "models--Qwen--Qwen3.5-4B" / "snapshots" / profile.revision
    snap.mkdir(parents=True)
    for name in profile.tokenizer_files:
        (snap / name).write_bytes(tok if name == "tokenizer.json" else b"{}")
    (snap / "config.json").write_text(
        json.dumps({"text_config": {"max_position_embeddings": 262_144}})
    )
    (snap / "tokenizer_config.json").write_text(json.dumps({"model_max_length": 262_144}))
    seen: dict[str, object] = {}

    def snapshot_download(**kwargs: object) -> str:
        seen.update(kwargs)
        return str(snap)

    fake_hub = types.SimpleNamespace(snapshot_download=snapshot_download)
    fake_tf = types.SimpleNamespace(
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda path, local_files_only: types.SimpleNamespace(
                model_max_length=262_144
            )
        )
    )
    real = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *a: (
            {"huggingface_hub": fake_hub, "transformers": fake_tf}.get(name) or real(name, *a)
        ),
    )
    scripting._public_book_context_cached.cache_clear()
    with model_profile.use(profile):
        _tokenizer, limit = scripting._public_book_context_cached("t", True)
    assert limit == 262_144
    assert seen["allow_patterns"] == list(profile.tokenizer_files)
    assert seen["repo_id"] == profile.model == "Qwen/Qwen3.5-4B"
    assert seen["revision"] == profile.revision
    scripting._public_book_context_cached.cache_clear()


class _MappingTokenizer:
    """transformers 5.x: apply_chat_template(tokenize=True) returns a BatchEncoding mapping."""

    name_or_path = ""

    def __init__(self, ids: list[int], mapping: bool = True) -> None:
        self.ids = ids
        self.mapping = mapping

    def apply_chat_template(self, messages: object, **kwargs: object) -> object:
        if self.mapping:
            return {"input_ids": list(self.ids), "attention_mask": [1] * len(self.ids)}
        return list(self.ids)


def test_template_token_ids_accepts_the_4x_list_and_the_5x_mapping() -> None:
    assert model_profile.template_token_ids([1, 2, 3]) == [1, 2, 3]
    assert model_profile.template_token_ids({"input_ids": [1, 2], "attention_mask": [1, 1]}) == [
        1,
        2,
    ]
    assert model_profile.template_token_ids({"attention_mask": [1]}) is None
    assert model_profile.template_token_ids("text") == "text"


@pytest.mark.parametrize("mapping", [False, True])
def test_prompt_context_check_counts_tokens_not_mapping_keys(mapping: bool) -> None:
    request = {"messages": [{"role": "user", "content": "x"}], "max_new_tokens": 960}
    tok = _MappingTokenizer(list(range(100)), mapping)
    assert constrained.check_prompt_context(request, tok, 4096) == 100
    with pytest.raises(constrained.ConstraintError):
        constrained.check_prompt_context(request, tok, 1000)


def test_chapter_prompt_count_ignores_the_mapping_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        scripting, "_chapter_tokenizer_material", lambda tokenizer: {"tokenizer_config_sha256": "a"}
    )
    for mapping in (False, True):
        encoded = scripting._chapter_prompt_encoding(
            [{"role": "user", "content": "x"}], tokenizer=_MappingTokenizer([5, 6, 7], mapping)
        )
        assert encoded["token_ids"] == [5, 6, 7]


def test_legacy_tokenizer_alias_is_restored_only_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib
    import types

    class Base:
        pass

    legacy = types.SimpleNamespace()
    modern = types.SimpleNamespace(PreTrainedTokenizerBase=Base)
    real = importlib.import_module
    table = {
        "transformers.tokenization_utils": legacy,
        "transformers.tokenization_utils_base": modern,
    }
    monkeypatch.setattr(
        importlib, "import_module", lambda name, *a: table.get(name) or real(name, *a)
    )
    constrained._restore_legacy_tokenizer_alias()
    assert legacy.PreTrainedTokenizerBase is Base
    other = type("Other", (), {})
    legacy.PreTrainedTokenizerBase = other
    constrained._restore_legacy_tokenizer_alias()
    assert legacy.PreTrainedTokenizerBase is other


@pytest.mark.parametrize(
    "changes",
    [
        {"tokenizer_files": ("config.json", "tokenizer.json", "tokenizer_config.json", [])},
        {"model_class": []},
        {"model_class": {}},
    ],
)
def test_malformed_runtime_fields_raise_the_typed_error(changes: dict[str, Any]) -> None:
    import dataclasses

    with pytest.raises(model_profile.ModelProfileError):
        dataclasses.replace(model_profile.QWEN3_5_4B, **changes)


def test_pinned_snapshot_requires_every_profile_tokenizer_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = model_profile.QWEN3_5_4B
    snap = tmp_path / "models--Qwen--Qwen3.5-4B" / "snapshots" / profile.revision
    snap.mkdir(parents=True)
    for name in profile.tokenizer_files:
        (snap / name).write_text("x")
    (snap / "model.safetensors").write_text("w")
    fake_hub = types.SimpleNamespace(snapshot_download=lambda *a, **k: str(snap))
    real = importlib.import_module
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name, *a: fake_hub if name == "huggingface_hub" else real(name, *a),
    )
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with model_profile.use(profile):
        assert scripting._pinned_qwen_snapshot() == snap.resolve()
        (snap / "chat_template.jinja").unlink()
        with pytest.raises(scripting.PinnedQwenSnapshotError):
            scripting._pinned_qwen_snapshot()


def test_loading_refuses_a_transformers_release_other_than_the_profile_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scripting, "_installed_version", lambda name: "9.9.9")
    with pytest.raises(scripting.NicheVideoError, match="does not match the model profile"):
        scripting._require_profile_transformers()
    monkeypatch.setattr(
        scripting,
        "_installed_version",
        lambda name: model_profile.active().transformers_requirement,
    )
    scripting._require_profile_transformers()


def test_producer_identity_records_the_installed_hub_version() -> None:
    assert "huggingface_hub" in whole_book.local_producer_material()["runtime"]  # type: ignore[operator]
