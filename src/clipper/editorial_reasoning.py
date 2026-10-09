"""Isolated reasoning-model transport for qualification, not production approval.

The native template must open thinking. JSON is parsed only from the final
answer; JSON-shaped intermediate reasoning can never become a verdict.
"""

from __future__ import annotations

import json
from typing import Any

MODEL_IDENTITY = {
    "repo": "bartowski/Qwen_Qwen3-30B-A3B-Thinking-2507-GGUF",
    "revision": "6a509d3e0aef5b70ffdcaa478e23de77482e6165",
    "file": "Qwen_Qwen3-30B-A3B-Thinking-2507-Q4_K_M.gguf",
    "sha256": "1359aa08e2f2dfe7ce4b5ff88c4c996e6494c9d916b1ebacd214bb74bbd5a9db",
}
SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "seed": 0}


def configure_native_thinking(model: Any, formatter_class: Any) -> None:
    """Use the pinned model's template without forcing a closed thinking block."""
    template = model.metadata.get("tokenizer.chat_template")
    if not isinstance(template, str) or not template.strip():
        raise ValueError("thinking reviewer requires its native chat template")
    formatter = formatter_class(
        template=template,
        eos_token=model.detokenize([model.token_eos()], special=True).decode(),
        bos_token=model.detokenize([model.token_bos()], special=True).decode(),
        add_generation_prompt=True,
        stop_token_ids=[model.token_eos()],
    )
    rendered = formatter(messages=[{"role": "user", "content": "Check the source."}])
    if not rendered.prompt.rstrip().endswith("<think>"):
        raise ValueError("thinking reviewer template does not open a reasoning block")
    model.chat_handler = formatter.to_chat_handler()


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("reasoning final answer contains duplicate JSON fields")
        result[key] = value
    return result


def parse_final_answer(response: dict[str, Any], properties: dict[str, Any]) -> dict[str, Any]:
    """Reject truncation, absent reasoning closure, and non-final JSON snippets."""
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("reasoning reviewer requires exactly one completed choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("reasoning reviewer did not finish; no verdict is available")
    message = choice.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or content.count("</think>") != 1:
        raise ValueError("reasoning reviewer must close exactly one thinking block")
    reasoning, final = content.split("</think>")
    if not reasoning.removeprefix("<think>").strip() or "<think>" in final:
        raise ValueError("reasoning reviewer returned an invalid thinking boundary")
    result = json.loads(final.strip(), object_pairs_hook=_unique_object)
    if not isinstance(result, dict) or set(result) != set(properties):
        raise ValueError("reasoning final answer differs from the requested fields")
    if "reason" in properties and (
        not isinstance(result.get("reason"), str) or not result["reason"].strip()
    ):
        raise ValueError("reasoning final answer omitted its evidence reason")
    return result


def reasoning_completion(
    model: Any,
    prompt: str,
    payload: dict[str, Any],
    properties: dict[str, Any],
    *,
    max_tokens: int = 32768,
) -> dict[str, Any]:
    """Keep the task/schema intact; do not impose JSON grammar on reasoning.

    Callers must retain the raw response and sampling/budget identity in their
    diagnostic records. Existing claim/span validators remain authoritative
    for structural validation; parsing this answer does not establish truth.
    """
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("reasoning token budget must be a positive integer")
    schema = {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }
    response = model.create_chat_completion(
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": json.dumps({**payload, "output_schema": schema}, ensure_ascii=False),
            },
        ],
        max_tokens=max_tokens,
        **SAMPLING,
    )
    return parse_final_answer(response, properties)
