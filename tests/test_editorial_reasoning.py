from types import SimpleNamespace

import pytest

from clipper.editorial_reasoning import (
    SAMPLING,
    configure_native_thinking,
    parse_final_answer,
    reasoning_completion,
)


def test_reasoning_profile_is_not_a_production_or_default_gpu_profile():
    from scripts.tjr_semantic_editor import (
        _gpu_review_profiles,
        _review_model_profile,
        _thinking_review_profile,
    )

    candidate = _thinking_review_profile()
    assert candidate not in _gpu_review_profiles()
    assert candidate["sha256"] != _review_model_profile()["sha256"]
    assert candidate["context_tokens"] > candidate["max_output_tokens"]
    assert candidate["sampling"] == SAMPLING
    assert len(candidate["transport_sha256"]) == 64


def test_reasoning_request_cache_records_real_sampling_and_distinct_implementation(tmp_path):
    from scripts.tjr_semantic_editor import (
        LocalContextualEditor,
        LocalReasoningReviewer,
        ReviewRequestCache,
    )

    default = ReviewRequestCache(tmp_path / "default.json", lambda: None, {})
    thinking = ReviewRequestCache(tmp_path / "thinking.json", lambda: None, {}, reasoning=True)
    assert default.identity["temperature"] == 0
    assert default.json_implementation is LocalContextualEditor._review_completion
    assert thinking.json_implementation is LocalReasoningReviewer._review_completion
    assert all(thinking.identity[key] == value for key, value in SAMPLING.items())


def test_workflow_reasoning_mode_routes_only_to_gpu_probe():
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[1]
    text = (root / ".github/workflows/tjr-weekly-hd.yml").read_text()
    workflow = yaml.safe_load(text)
    steps = next(
        job["steps"]
        for job in workflow["jobs"].values()
        if job.get("name") == "Real-model editorial regression (no media acquisition)"
    )
    cpu = next(step for step in steps if step.get("name") == "Install pinned CPU reviewer")
    assert "inputs.reviewer_model_probe != 'reasoning_gpu'" in cpu["if"]
    gpu = next(
        step for step in steps if step.get("name") == "Install private GPU orchestration client"
    )
    assert "inputs.reviewer_model_probe == 'reasoning_gpu'" in gpu["if"]
    assert "PROBE_ARGUMENTS+=(--reasoning-gpu-probe)" in text


def response(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


def test_only_final_answer_can_supply_a_verdict():
    raw = response('First consider {"supported": true}. It is wrong.</think>{"supported": false}')
    assert parse_final_answer(raw, {"supported": {"type": "boolean"}}) == {"supported": False}


@pytest.mark.parametrize(
    "content,finish",
    [
        ('reasoning</think>{"supported": true}', "length"),
        ('{"supported": true}', "stop"),
        ('</think>{"supported": true}', "stop"),
        ('reasoning</think></think>{"supported": true}', "stop"),
        ('reasoning</think><think>{"supported": true}', "stop"),
        ('reasoning</think>{"supported": true} trailing text', "stop"),
        ('reasoning</think>{"supported": true, "supported": false}', "stop"),
        ('reasoning</think>{"different_field": true}', "stop"),
        ("reasoning</think>[]", "stop"),
    ],
)
def test_invalid_or_unfinished_answers_cannot_be_salvaged(content, finish):
    with pytest.raises(ValueError):
        parse_final_answer(response(content, finish), {"supported": {"type": "boolean"}})


def test_reasoning_is_not_constrained_by_a_json_grammar():
    calls = []

    def complete(**kwargs):
        calls.append(kwargs)
        return response('Source assessment.</think>{"supported": false}')

    result = reasoning_completion(
        SimpleNamespace(create_chat_completion=complete),
        "Existing task.",
        {"source": "Exact source."},
        {"supported": {"type": "boolean"}},
    )
    assert result == {"supported": False}
    assert "response_format" not in calls[0]
    assert "grammar" not in calls[0]
    assert calls[0]["max_tokens"] == 32768
    assert all(calls[0][key] == value for key, value in SAMPLING.items())


@pytest.mark.parametrize("ending,valid", [("<think>\n", True), ("</think>\n", False)])
def test_native_template_must_leave_reasoning_open(ending, valid):
    class Formatter:
        def __init__(self, **kwargs):
            assert kwargs["template"] == "native model template"

        def __call__(self, **kwargs):
            return SimpleNamespace(prompt="assistant\n" + ending)

        def to_chat_handler(self):
            return "native handler"

    model = SimpleNamespace(
        metadata={"tokenizer.chat_template": "native model template"},
        token_eos=lambda: 1,
        token_bos=lambda: 2,
        detokenize=lambda *args, **kwargs: b"token",
    )
    if valid:
        configure_native_thinking(model, Formatter)
        assert model.chat_handler == "native handler"
    else:
        with pytest.raises(ValueError, match="does not open"):
            configure_native_thinking(model, Formatter)
