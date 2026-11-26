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
from typing import Any

_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")


class ModelProfileError(ValueError):
    """Raised for an invalid or unknown model profile."""


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

    def __post_init__(self) -> None:
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
        object.__setattr__(
            self, "chat_template_kwargs", MappingProxyType(dict(self.chat_template_kwargs))
        )
        object.__setattr__(
            self, "generation_settings", MappingProxyType(dict(self.generation_settings))
        )

    @property
    def template_kwargs(self) -> dict[str, Any]:
        return dict(self.chat_template_kwargs)

    @property
    def sampling(self) -> dict[str, Any]:
        return dict(self.generation_settings)

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
            "chat_template_kwargs": dict(self.chat_template_kwargs),
            "generation_settings": dict(self.generation_settings),
        }
        return hashlib.sha256(json.dumps(material, sort_keys=True).encode()).hexdigest()

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

DEFAULT_PROFILE = QWEN3_1_7B.name
_PROFILES: dict[str, ModelProfile] = {QWEN3_1_7B.name: QWEN3_1_7B}
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
