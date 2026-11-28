"""Model profiles: every fact about the pinned text model lives here, nowhere else.

A profile names the model and its pinned revision, the tokenizer and context
facts the preflight must verify, the chat-template keyword arguments, and the
sampling settings. Pipeline code reads the active profile instead of repeating
these values, so adding or swapping a model is a new profile entry plus its own
offline tokenizer check, not a change to the pipeline.

Constrained decoding is tokenizer-bound: a new profile is usable only after its
tokenizer hash is pinned and the tokenizer-only tests pass for it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, cast

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


def template_token_ids(rendered: Any) -> Any:
    """Token IDs from an apply_chat_template(tokenize=True) result.

    transformers 4.x returns a plain list; 5.x returns a BatchEncoding mapping with
    input_ids. Anything else is returned unchanged so the caller's own check refuses it.
    """
    if isinstance(rendered, Mapping):
        return rendered.get("input_ids")
    return rendered


DEFAULT_TOKENIZER_FILES = (
    "added_tokens.json",
    "config.json",
    "generation_config.json",
    "merges.txt",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)
MODEL_CLASSES = frozenset({"AutoModelForCausalLM", "AutoModelForImageTextToText"})


class ModelProfileError(ValueError):
    """Raised for an invalid or unknown model profile."""


def _freeze(value: Any) -> Any:
    """Deep-freeze nested containers: mappings to read-only proxies, lists to tuples."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _thaw(value: Any) -> Any:
    """Fresh mutable copy of a frozen value, safe to hand to callers."""
    if isinstance(value, Mapping):
        return {k: _thaw(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(v) for v in value]
    return value


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    revision: str
    tokenizer_json_sha256: str
    context_window: int
    tokenizer_max_length: int
    safe_context_tokens: int
    chat_template_kwargs: Mapping[str, Any]
    generation_settings: Mapping[str, Any]
    # Output budget facts. A fixed config value must lie in [min, max]; "auto" resolves to
    # min(max_output_tokens, safe_context_tokens - prompt_tokens - output_margin_tokens).
    max_output_tokens: int = 16_384
    output_margin_tokens: int = 512
    min_output_tokens: int = 1_024
    # Runtime facts the Modal image and the loader need. They are part of the fingerprint.
    transformers_requirement: str = "4.57.3"
    huggingface_hub_requirement: str = "0.36.2"
    # transformers class used to load the weights (text-only generation for every profile).
    model_class: str = "AutoModelForCausalLM"
    # Public files fetched and checked before any weight is loaded (names in the snapshot).
    tokenizer_files: tuple[str, ...] = DEFAULT_TOKENIZER_FILES

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str)
            for value in (self.name, self.model, self.revision, self.tokenizer_json_sha256)
        ):
            raise ModelProfileError("profile name, model, revision and tokenizer hash must be text")
        if not _NAME.fullmatch(self.name):
            raise ModelProfileError("profile name invalid")
        if not self.model or "/" not in self.model or self.model != self.model.strip():
            raise ModelProfileError("profile model must be an org/name id")
        if not _REVISION.fullmatch(self.revision):
            raise ModelProfileError("profile revision must be a 40-hex pinned commit")
        if not _DIGEST.fullmatch(self.tokenizer_json_sha256):
            raise ModelProfileError("profile tokenizer hash must be sha256 hex")
        if type(self.context_window) is not int or self.context_window < 2048:
            raise ModelProfileError("profile context window invalid")
        if type(self.tokenizer_max_length) is not int or self.tokenizer_max_length < 1:
            raise ModelProfileError("profile tokenizer max length invalid")
        if type(self.safe_context_tokens) is not int or not 1024 <= self.safe_context_tokens <= min(
            self.context_window, self.tokenizer_max_length
        ):
            raise ModelProfileError("profile safe context must fit the model and tokenizer limits")
        if (
            any(
                type(value) is not int
                for value in (
                    self.max_output_tokens,
                    self.output_margin_tokens,
                    self.min_output_tokens,
                )
            )
            or not 1 <= self.min_output_tokens <= self.max_output_tokens
            or self.max_output_tokens >= self.safe_context_tokens
            or not 0 <= self.output_margin_tokens < self.safe_context_tokens
        ):
            raise ModelProfileError("profile output budget must fit inside the safe context")
        for value in (
            self.transformers_requirement,
            self.huggingface_hub_requirement,
        ):
            if not isinstance(value, str) or not re.fullmatch(r"\d+(\.\d+){1,2}", value):
                raise ModelProfileError("profile package requirement must be an exact version")
        if (
            not isinstance(self.tokenizer_files, tuple)
            or not self.tokenizer_files
            or any(
                not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9._-]*", name)
                for name in self.tokenizer_files
            )
            or len(set(self.tokenizer_files)) != len(self.tokenizer_files)
            or not {"config.json", "tokenizer.json", "tokenizer_config.json"}
            <= set(self.tokenizer_files)
        ):
            raise ModelProfileError("profile tokenizer files invalid")
        if not isinstance(self.model_class, str) or self.model_class not in MODEL_CLASSES:
            raise ModelProfileError("profile model class unsupported")
        object.__setattr__(self, "chat_template_kwargs", _freeze(self.chat_template_kwargs))
        object.__setattr__(self, "generation_settings", _freeze(self.generation_settings))

    @property
    def template_kwargs(self) -> dict[str, Any]:
        return cast(dict[str, Any], _thaw(self.chat_template_kwargs))

    @property
    def sampling(self) -> dict[str, Any]:
        return cast(dict[str, Any], _thaw(self.generation_settings))

    @property
    def fingerprint(self) -> str:
        """Stable identity of every profile fact, used to key per-profile caches."""
        material = {
            "name": self.name,
            "model": self.model,
            "revision": self.revision,
            "tokenizer_json_sha256": self.tokenizer_json_sha256,
            "context_window": self.context_window,
            "tokenizer_max_length": self.tokenizer_max_length,
            "safe_context_tokens": self.safe_context_tokens,
            "chat_template_kwargs": _thaw(self.chat_template_kwargs),
            "generation_settings": _thaw(self.generation_settings),
            "max_output_tokens": self.max_output_tokens,
            "output_margin_tokens": self.output_margin_tokens,
            "min_output_tokens": self.min_output_tokens,
        }
        # Added only when it differs from the original defaults, so an existing profile
        # keeps the fingerprint (and every cache key) it had before these fields existed.
        if (
            self.transformers_requirement,
            self.huggingface_hub_requirement,
            self.model_class,
            self.tokenizer_files,
        ) != ("4.57.3", "0.36.2", "AutoModelForCausalLM", DEFAULT_TOKENIZER_FILES):
            material["runtime"] = {
                "transformers": self.transformers_requirement,
                "huggingface_hub": self.huggingface_hub_requirement,
                "model_class": self.model_class,
                "tokenizer_files": list(self.tokenizer_files),
            }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

    def auto_output_tokens(self, prompt_tokens: int) -> int:
        """Largest finite output budget the measured prompt leaves room for."""
        if type(prompt_tokens) is not int or prompt_tokens < 1:
            raise ModelProfileError("measured prompt token count invalid")
        budget = min(
            self.max_output_tokens,
            self.safe_context_tokens - prompt_tokens - self.output_margin_tokens,
        )
        if budget < self.min_output_tokens:
            raise ModelProfileError("prompt leaves too little room for the minimum output")
        return budget

    @property
    def effective_context(self) -> int:
        """Largest prompt+output the pinned tokenizer and model both accept."""
        return min(self.context_window, self.tokenizer_max_length)


QWEN3_1_7B = ModelProfile(
    name="qwen3-1.7b",
    model="Qwen/Qwen3-1.7B",
    revision="8d4744f9e13072f4920c326350fa81eedb74eae9",
    tokenizer_json_sha256="aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    context_window=40_960,
    tokenizer_max_length=131_072,
    safe_context_tokens=32_768,
    chat_template_kwargs={"enable_thinking": False},
    generation_settings={"do_sample": True, "temperature": 0.7, "top_p": 0.8},
)

# Qwen3.5-4B is a vision-language checkpoint (Qwen3_5ForConditionalGeneration, model type
# qwen3_5). It is used text-only: the visual tower loads but no image or video is ever
# passed. Values come from the pinned snapshot: max_position_embeddings sits under
# text_config in config.json; model_max_length is in tokenizer_config.json. Sampling is
# the model card's non-thinking setting minus presence_penalty, which transformers'
# generate() does not implement. transformers 4.57.3 does not know model type qwen3_5, so
# this profile needs the 5.x line.
QWEN3_5_4B = ModelProfile(
    name="qwen3.5-4b",
    model="Qwen/Qwen3.5-4B",
    revision="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    tokenizer_json_sha256="5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42",
    context_window=262_144,
    tokenizer_max_length=262_144,
    safe_context_tokens=32_768,
    chat_template_kwargs={"enable_thinking": False},
    generation_settings={"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20},
    transformers_requirement="5.24.1",
    huggingface_hub_requirement="2.5.0",
    model_class="AutoModelForImageTextToText",
    tokenizer_files=(
        "chat_template.jinja",
        "config.json",
        "merges.txt",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ),
)

DEFAULT_PROFILE = QWEN3_1_7B.name
_PROFILES: dict[str, ModelProfile] = {
    QWEN3_1_7B.name: QWEN3_1_7B,
    QWEN3_5_4B.name: QWEN3_5_4B,
}
_active: ContextVar[ModelProfile] = ContextVar("clipper_active_model_profile", default=QWEN3_1_7B)


def profiles() -> dict[str, ModelProfile]:
    return dict(_PROFILES)


def get(name: str) -> ModelProfile:
    try:
        return _PROFILES[name]
    except (KeyError, TypeError):
        raise ModelProfileError("unknown model profile") from None


def register(profile: ModelProfile) -> None:
    if profile.name in _PROFILES and _PROFILES[profile.name] != profile:
        raise ModelProfileError("profile name already registered with different facts")
    _PROFILES[profile.name] = profile


def output_bounds() -> tuple[int, int]:
    """Inclusive limits for a fixed max_new_tokens under the active profile."""
    profile = _active.get()
    return profile.min_output_tokens, profile.max_output_tokens


def active() -> ModelProfile:
    return _active.get()


@contextmanager
def use(profile: ModelProfile | str) -> Iterator[ModelProfile]:
    """Select a profile for the duration of the block, isolated per thread and task."""
    chosen = get(profile) if isinstance(profile, str) else profile
    token = _active.set(chosen)
    try:
        yield chosen
    finally:
        _active.reset(token)
