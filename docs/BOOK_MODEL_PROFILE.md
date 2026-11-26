# Model profiles

Every fact about the pinned text model lives in one record, `ModelProfile`
(`src/clipper/books/model_profile.py`). Pipeline code reads the active profile
instead of repeating these values:

| Field | Used for |
| --- | --- |
| `model`, `revision` | Hugging Face repo id and the pinned 40-hex commit. Part of every request, stage and cache identity. |
| `tokenizer_json_sha256` | Pinned public `tokenizer.json`. The preflight and the constrained decoder refuse a different tokenizer. |
| `context_window` | Must equal `max_position_embeddings` in the pinned `config.json`. Prompt plus output is checked against it. |
| `tokenizer_max_length` | Must equal `model_max_length` in the pinned `tokenizer_config.json`. |
| `safe_context_tokens` | Operational combined prompt-plus-output limit used by the hybrid evaluator (32,768 for Qwen3-1.7B). Must not exceed the window or tokenizer length. |
| `chat_template_kwargs` | Extra keyword arguments for `apply_chat_template` (for the Qwen3 family, `enable_thinking: false`). |
| `generation_settings` | Sampling settings. |

The default profile is `qwen3-1.7b` and reproduces the previous hard-coded
values exactly (model `Qwen/Qwen3-1.7B`, revision `8d4744f9...`, window 40,960).

## Adding or swapping a model

1. Add a `ModelProfile` entry and register it. Take the revision, tokenizer hash,
   window and tokenizer length from the pinned snapshot, not from the model card.
   The Qwen3-1.7B card says 32,768 context while its `config.json` says 40,960;
   the profile stores the `config.json` value because the preflight checks it
   against the pinned snapshot.
2. Run the tokenizer-only checks for the new profile offline. Constrained decoding
   depends on the tokenizer, so a new model family is not assumed to work.
3. Nothing in the pipeline changes. Profile facts enter the request, stage and
   cache identities, so a different model never reuses another model's checkpoint
   or cache.

Template kwargs and sampling settings default to the family's values and can be
overridden per profile. Same-family models are not assumed to share a tokenizer
hash or window: verify each profile.

## Isolation and limits

The active profile is a `ContextVar`, so it is isolated per thread and task and restored on exit. Tokenizer and model caches are keyed by the profile fingerprint (a hash of every profile fact), so profiles never share a cached object.

The Modal private phase supports only the default profile. Its image build downloads the default snapshot before the network is cut, so a configuration naming another profile is rejected before launch (`private_phase.require_default_profile`) instead of failing inside the worker. Selecting the profile in the image build is the next step for a second model on Modal.

## Output budget

Each profile carries its output-budget facts: `max_output_tokens` (finite ceiling,
default 16384 for Qwen3-1.7B), `output_margin_tokens` (default 512) and
`min_output_tokens` (default 1024). They must fit inside `safe_context_tokens`, so a
profile with a small safe context must also set a smaller `max_output_tokens`.

A stage's `max_new_tokens` in the pipeline configuration is either:

- an integer between `min_output_tokens` and `max_output_tokens` (fixed, as before), or
- the string `"auto"`.

For `"auto"`, the budget is resolved once per request, from the prompt measured with the
pinned tokenizer, before the request is hashed:

    budget = min(max_output_tokens, safe_context_tokens - prompt_tokens - output_margin_tokens)

If `budget` is below `min_output_tokens`, the stage stops before any model call with a clear
error. The resolved integer is what the request, stage key, receipts and replay check use,
so the same prompt always gives the same budget, and a different prompt gives a different
request identity. The limit is the profile's safe context (32768 for Qwen3-1.7B), not the
40960 window.
