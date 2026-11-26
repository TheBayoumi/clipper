"""The model profile is the single source of model facts for the book pipeline."""

from __future__ import annotations

import dataclasses
import json

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
        {"safe_context_tokens": 2048},
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


def test_modal_phase_rejects_non_default_profile(fake_registered: ModelProfile) -> None:
    from clipper.books import private_phase

    default = private_phase.pack(
        {"config.json": json.dumps(pipeline_plan.template("c" * 64)).encode()}
    )
    private_phase.require_default_profile(default)
    chosen = private_phase.pack(
        {"config.json": json.dumps(pipeline_plan.template("c" * 64, "script", FAKE.name)).encode()}
    )
    with pytest.raises(private_phase.PrivatePhaseError):
        private_phase.require_default_profile(chosen)
    assert model_profile.active().name == "qwen3-1.7b"
