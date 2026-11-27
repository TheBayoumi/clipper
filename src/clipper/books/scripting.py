"""Pinned Qwen3 script drafting, validation, bounded repair and recovery.

This is the canonical script stage; no TTS or video rendering runs here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlsplit

import yaml

from . import constrained, model_profile

# Pin the script weights; keep TTS identity versioned to invalidate cache on upgrades.
VOICE_MODEL = "hexgrad/Kokoro-82M"
VOICE_REVISION = "v1.0"
SCRIPT_CONTRACT = "niche-scenes-open-v3"
BOOK_EDIT_CONTRACT = "book-lesson-complete-sentences-v2"
MAX_SCRIPT_ATTEMPTS = 3
VOICE_CONTRACT = "kokoro-24khz-wav-v1"
VOICES = frozenset({"af_heart", "af_bella", "am_adam", "am_michael", "bf_emma"})


_PROFILE_ALIASES = {
    "SCRIPT_MODEL": lambda profile: profile.model,
    "SCRIPT_REVISION": lambda profile: profile.revision,
    "QWEN_GENERATION_SETTINGS": lambda profile: profile.sampling,
    "MAX_CHAPTER_CONTEXT": lambda profile: profile.context_window,
}


def __getattr__(name: str) -> Any:
    """Read-only views of the active model profile under the historical names."""
    if name in _PROFILE_ALIASES:
        return _PROFILE_ALIASES[name](model_profile.active())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class NicheVideoError(RuntimeError):
    """A validation or production step failed closed."""


def _text(value: object, field: str, limit: int = 500) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise NicheVideoError(f"{field} must be nonempty text of at most {limit} characters")
    return value.strip()


@dataclass(frozen=True)
class NicheConfig:
    niche: str
    audience: str
    brand: str
    voice: str
    facts: tuple[str, ...]
    topic: str
    format: str = "general"
    pillar: str = ""
    book_title: str = ""
    book_author: str = ""
    source_url: str = ""
    book_id: str = ""
    lesson_id: str = ""
    angle_id: str = ""
    episode_id: str = ""
    script_version: str = ""
    selection_fingerprint: str = ""
    hook_angle: str = ""
    visual_action: str = ""

    @classmethod
    def load(cls, path: Path, topic_override: str | None = None) -> NicheConfig:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise NicheVideoError("niche config must be an object")
        allowed = {
            "niche",
            "audience",
            "brand",
            "voice",
            "facts",
            "topic",
            "format",
            "pillar",
            "book_title",
            "book_author",
            "source_url",
            "book_id",
            "lesson_id",
            "angle_id",
            "episode_id",
            "script_version",
            "selection_fingerprint",
            "hook_angle",
            "visual_action",
        }
        if set(raw) - allowed:
            raise NicheVideoError(
                f"unsupported niche configuration keys: {sorted(set(raw) - allowed)}"
            )
        facts = raw.get("facts", [])
        if not isinstance(facts, list) or len(facts) > 12:
            raise NicheVideoError("facts must be a list of at most twelve verified statements")
        if not all(isinstance(fact, str) and 1 <= len(fact.strip()) <= 400 for fact in facts):
            raise NicheVideoError("every fact must be a nonempty string of at most 400 characters")
        voice = _text(raw.get("voice", "af_heart"), "voice", 30).lower()
        if voice not in VOICES:
            raise NicheVideoError("unsupported Kokoro voice")
        story_format = _text(raw.get("format", "general"), "format", 30)
        if story_format not in {"general", "book_lesson"}:
            raise NicheVideoError("unsupported niche video format")
        pillar = str(raw.get("pillar", ""))
        title = str(raw.get("book_title", ""))
        author = str(raw.get("book_author", ""))
        source = str(raw.get("source_url", ""))
        if story_format == "book_lesson":
            if pillar not in {"self_improvement", "psychology", "business"}:
                raise NicheVideoError("book lesson requires a recognized pillar")
            title = _text(title, "book_title", 100)
            author = _text(author, "book_author", 100)
            source = _text(source, "source_url", 500)
            parts = urlsplit(source)
            if (
                parts.scheme != "https"
                or not parts.hostname
                or parts.username is not None
                or parts.password is not None
            ):
                raise NicheVideoError("book lesson source must be an HTTPS URL")
            if not facts:
                raise NicheVideoError("book lesson requires approved, sourced facts")
        elif any((pillar, title, author, source)):
            raise NicheVideoError("book metadata requires format: book_lesson")
        identity_fields = (
            "book_id",
            "lesson_id",
            "angle_id",
            "episode_id",
            "script_version",
            "selection_fingerprint",
            "hook_angle",
            "visual_action",
        )
        if any(key in raw for key in identity_fields):
            if story_format != "book_lesson" or not all(
                isinstance(raw.get(key), str) and raw[key].strip() for key in identity_fields
            ):
                raise NicheVideoError("book episode identity must be complete")
            identity = {key: raw[key] for key in identity_fields}
            for key in ("book_id", "lesson_id", "angle_id"):
                if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", identity[key]):
                    raise NicheVideoError(f"invalid {key}")
            expected = "--".join(identity[key] for key in ("book_id", "lesson_id", "angle_id"))
            if identity["episode_id"] != expected:
                raise NicheVideoError("book episode identity mismatch")
            if identity["script_version"] != BOOK_EDIT_CONTRACT:
                raise NicheVideoError("unsupported book script version")
            if not re.fullmatch(r"[0-9a-f]{64}", identity["selection_fingerprint"]):
                raise NicheVideoError("selection fingerprint must be a SHA256")
        else:
            identity = {}
        return cls(
            niche=_text(raw.get("niche"), "niche", 150),
            audience=_text(raw.get("audience"), "audience", 200),
            brand=_text(raw.get("brand"), "brand", 60),
            voice=voice,
            facts=tuple(fact.strip() for fact in facts),
            topic=_text(topic_override or raw.get("topic"), "topic", 200),
            format=story_format,
            pillar=pillar,
            book_title=title,
            book_author=author,
            source_url=source,
            **identity,
        )


@dataclass(frozen=True)
class Scene:
    headline: str
    narration: str


@dataclass(frozen=True)
class Episode:
    title: str
    scenes: tuple[Scene, ...]

    @classmethod
    def from_json(cls, raw: object) -> Episode:
        if not isinstance(raw, dict) or set(raw) != {"title", "scenes"}:
            raise NicheVideoError("episode schema mismatch")
        scenes_raw = raw["scenes"]
        if not isinstance(scenes_raw, list) or not 3 <= len(scenes_raw) <= 5:
            raise NicheVideoError("episode requires 3-5 distinct scenes")
        scenes: list[Scene] = []
        for item in scenes_raw:
            if not isinstance(item, dict) or set(item) != {"headline", "narration"}:
                raise NicheVideoError("scene schema mismatch")
            headline = _text(item["headline"], "headline", 65)
            narration = _text(item["narration"], "narration", 400)
            if not 6 <= len(narration.split()) <= 45:
                raise NicheVideoError("scene narration must contain 6-45 words")
            scenes.append(Scene(headline=headline, narration=narration))
        if not 45 <= sum(len(scene.narration.split()) for scene in scenes) <= 125:
            raise NicheVideoError("episode narration must contain 45-125 words")
        return cls(title=_text(raw["title"], "title", 100), scenes=tuple(scenes))

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "scenes": [
                {"headline": scene.headline, "narration": scene.narration} for scene in self.scenes
            ],
        }


def _fingerprint(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temp = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temp.unlink(missing_ok=True)
            raise
    temp.replace(path)


def _script_request(
    config: NicheConfig, *, evaluation_context: dict[str, object] | None = None
) -> dict[str, object]:
    request = {
        "model": model_profile.active().model,
        "revision": model_profile.active().revision,
        "seed": 42,
        "max_new_tokens": 960,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Write exactly one valid JSON object, no markdown or commentary. "
                    'Schema: {"title": string, "scenes": '
                    '[{"headline": string, "narration": string}]}. '
                    "Write four distinct scenes, aiming for 80-100 narration words "
                    "in total, around 20-25 words each. Hard limits: 3-5 scenes, "
                    "45-125 total words, 6-45 per scene, headline <= 65 chars. "
                    "Start with a clear honest hook, develop useful original "
                    "ideas, finish with a complete payoff. Never invent statistics, "
                    "quotes, or news. Concrete claims may use approved_facts only. "
                    "With no facts, use general reproducible methods instead. "
                    "For book lessons, attribute the author naturally, explain "
                    "one precise idea, one relatable original example, and one "
                    "actionable experiment. Do not quote or reproduce book prose."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "niche": config.niche,
                        "audience": config.audience,
                        "topic": config.topic,
                        "approved_facts": config.facts,
                        "format": config.format,
                        "pillar": config.pillar,
                        "book_title": config.book_title,
                        "book_author": config.book_author,
                        "source_url": config.source_url,
                        **(
                            {
                                "book_id": config.book_id,
                                "lesson_id": config.lesson_id,
                                "angle_id": config.angle_id,
                                "episode_id": config.episode_id,
                                "script_version": config.script_version,
                                "selection_fingerprint": config.selection_fingerprint,
                                "hook_angle": config.hook_angle,
                                "visual_action": config.visual_action,
                            }
                            if config.episode_id
                            else {}
                        ),
                    },
                    ensure_ascii=False,
                ),
            },
        ],
    }
    if evaluation_context is not None:
        # Evaluation evidence is NOT an approved-facts catalog. Keep it separate
        # and content-addressed in the canonical Qwen script request/cache key.
        messages = cast(list[dict[str, str]], request["messages"])
        messages[0]["content"] += (
            " This is a private assisted evaluation, NOT an approved book fact. "
            "Use only the reviewer-attested narrow claim and exact evidence/source "
            "context; preserve all uncertainty and counterexamples. Do not convert "
            "the source into a stronger generalization or invent narrative facts. "
            "Do not imply editorial approval or permission to publish."
        )
        user = json.loads(messages[1]["content"])
        user["approved_facts"] = []
        user["evaluation_source_context"] = evaluation_context
        messages[1]["content"] = json.dumps(user, ensure_ascii=False, sort_keys=True)
    return request


class IncompleteConstrainedResponse(NicheVideoError):
    """A generation stopped before EOS; its unchanged raw text must be stored."""

    def __init__(self, raw: str) -> None:
        super().__init__("constrained book generation stopped without EOS")
        self.raw = raw


class PinnedQwenSnapshotError(NicheVideoError):
    """Missing or incomplete exact-revision model cache, without revision fallback."""


def _pinned_qwen_snapshot() -> Path:
    """Resolve the exact HF snapshot locally before calling Transformers.

    Transformers 4.57.3 can query Hub metadata while patching tokenizer
    regexes if given a repository ID, even when its files are cached.
    A local snapshot path avoids that lookup for both tokenizer and weights.
    """
    try:
        hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise NicheVideoError(
            "Install the [niche-models] runtime for pinned Qwen inference"
        ) from exc
    offline = any(
        os.environ.get(key, "").strip().casefold() in {"1", "true", "yes", "on"}
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )

    def verify(snapshot_path: str | Path) -> Path:
        try:
            path = Path(snapshot_path).resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise PinnedQwenSnapshotError("pinned Qwen snapshot is missing") from exc
        if (
            not path.is_dir()
            or path.name != model_profile.active().revision
            or path.parent.name != "snapshots"
            or path.parent.parent.name
            != "models--" + model_profile.active().model.replace("/", "--")
        ):
            raise PinnedQwenSnapshotError("pinned Qwen snapshot identity mismatch")
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            file = path / name
            if not file.is_file() or file.stat().st_size == 0:
                raise PinnedQwenSnapshotError("pinned Qwen snapshot is incomplete")
        single = path / "model.safetensors"
        if single.is_file() and single.stat().st_size:
            return path
        index = path / "model.safetensors.index.json"
        try:
            if not index.is_file() or index.stat().st_size > 2_000_000:
                raise PinnedQwenSnapshotError(
                    "pinned Qwen snapshot incomplete: weight index missing or oversized"
                )
            index_data = json.loads(index.read_text(encoding="utf-8"))
            mapping = index_data["weight_map"]
            if not isinstance(mapping, dict) or not mapping:
                raise ValueError("empty model shard mapping")
            shard_names = set(mapping.values())
            if any(
                not isinstance(name, str)
                or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]*\.safetensors", name)
                for name in shard_names
            ):
                raise ValueError("unsafe model shard reference")
            if any(
                not (path / name).is_file() or (path / name).stat().st_size == 0
                for name in shard_names
            ):
                raise ValueError("missing model weight shard")
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            raise PinnedQwenSnapshotError("pinned Qwen snapshot is incomplete") from exc
        return path

    try:
        locally_cached = hub.snapshot_download(
            repo_id=model_profile.active().model,
            revision=model_profile.active().revision,
            local_files_only=True,
        )
        return verify(locally_cached)
    except PinnedQwenSnapshotError:
        if offline:
            raise
    except Exception as exc:
        if offline:
            raise PinnedQwenSnapshotError(
                "pinned Qwen snapshot unavailable in offline cache"
            ) from exc
    # Only explicit inference can reach this branch; a network-enabled
    # invocation may acquire public pinned weights, never another revision.
    try:
        downloaded = hub.snapshot_download(
            repo_id=model_profile.active().model,
            revision=model_profile.active().revision,
            local_files_only=False,
        )
        return verify(downloaded)
    except PinnedQwenSnapshotError:
        raise
    except Exception as exc:
        raise PinnedQwenSnapshotError("pinned Qwen snapshot download failed") from exc


def _public_cpu_smoke_phase(phase: str) -> None:
    """Optional public-only phase markers, never prompts, tokens, or credentials."""
    if os.environ.get("CLIPPER_PUBLIC_BOOK_CPU_SMOKE") == "1":
        print(f"PUBLIC_BOOK_CPU_MODEL_PHASE {phase}", flush=True)
        destination = os.environ.get("CLIPPER_PUBLIC_BOOK_CPU_SMOKE_PHASE_FILE")
        if destination:
            path = Path(destination)
            if (
                path.name != "model-phases.jsonl"
                or path.is_symlink()
                or path.parent.is_symlink()
                or not path.parent.is_dir()
                or not re.fullmatch(r"[a-z_]{1,64}", phase)
            ):
                raise NicheVideoError("public CPU model phase evidence path is unsafe")
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps({"phase": phase}, sort_keys=True) + "\n")
            path.chmod(0o600)


def _public_book_context(*, local_files_only: bool = False) -> tuple[Any, int]:
    """Pinned public tokenizer and context limit, cached per model profile."""
    return _public_book_context_cached(model_profile.active().fingerprint, local_files_only)


@lru_cache(maxsize=4)
def _public_book_context_cached(
    profile_fingerprint: str, local_files_only: bool
) -> tuple[Any, int]:
    """Validate pinned public-only tokenizer/config before loading book weights.

    The profile fingerprint is only the cache key; the facts come from the active profile.
    """

    try:
        hub = importlib.import_module("huggingface_hub")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise PinnedQwenSnapshotError("pinned public book tokenizer runtime missing") from exc
    offline = local_files_only or any(
        os.environ.get(key, "").strip().casefold() in {"1", "true", "yes", "on"}
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )
    public = (
        "added_tokens.json",
        "config.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    )

    def fetch(*, local_files_only: bool) -> Path:
        return Path(
            hub.snapshot_download(
                repo_id=model_profile.active().model,
                revision=model_profile.active().revision,
                allow_patterns=list(public),
                local_files_only=local_files_only,
            )
        )

    try:
        try:
            snapshot = fetch(local_files_only=True)
        except Exception as exc:
            if offline:
                raise PinnedQwenSnapshotError(
                    "pinned public tokenizer unavailable offline"
                ) from exc
            snapshot = fetch(local_files_only=False)
        snapshot = snapshot.resolve(strict=True)
        if (
            snapshot.name != model_profile.active().revision
            or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name
            != "models--" + model_profile.active().model.replace("/", "--")
        ):
            raise PinnedQwenSnapshotError("pinned public tokenizer snapshot mismatch")
        if hashlib.sha256((snapshot / "tokenizer.json").read_bytes()).hexdigest() != (
            constrained.PINNED_TOKENIZER_SHA256
        ):
            raise PinnedQwenSnapshotError("pinned public tokenizer hash mismatch")
        if local_files_only and any(not (snapshot / name).is_file() for name in public):
            raise PinnedQwenSnapshotError("complete pinned public tokenizer unavailable offline")
        model_config = json.loads((snapshot / "config.json").read_text(encoding="utf-8"))
        tokenizer_config = json.loads(
            (snapshot / "tokenizer_config.json").read_text(encoding="utf-8")
        )
        model_limit = model_config["max_position_embeddings"]
        tokenizer_limit = tokenizer_config["model_max_length"]
        if (
            type(model_limit) is not int
            or type(tokenizer_limit) is not int
            or model_limit != model_profile.active().context_window
            or tokenizer_limit != model_profile.active().tokenizer_max_length
        ):
            raise PinnedQwenSnapshotError("pinned public context limits mismatch")
        tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
        if tokenizer.model_max_length != tokenizer_limit:
            raise PinnedQwenSnapshotError("pinned tokenizer context metadata mismatch")
        return tokenizer, min(model_limit, tokenizer_limit)
    except PinnedQwenSnapshotError:
        raise
    except Exception as exc:
        raise PinnedQwenSnapshotError("pinned public context preflight unavailable") from exc


def _qwen_device() -> str:
    """Pinned Qwen device: CPU unless a reviewed private phase selects CUDA."""
    device = os.environ.get("CLIPPER_QWEN_DEVICE", "cpu")
    if device not in ("cpu", "cuda"):
        raise NicheVideoError("unsupported pinned Qwen device")
    return device


def _device_type(model: Any) -> str:
    device = getattr(model, "device", "cpu")
    return str(getattr(device, "type", device))


def _to_model_device(inputs: Any, model: Any) -> Any:
    """Tokenized tensors follow the model; CPU inputs stay untouched."""
    device = getattr(model, "device", "cpu")
    kind = getattr(device, "type", device)
    if not isinstance(kind, str) or kind == "cpu":
        return inputs
    return inputs.to(device)


def _script_pipeline() -> tuple[Any, Any]:
    """Pinned model and tokenizer for the active profile, initialized once per profile."""
    return _script_pipeline_cached(model_profile.active().fingerprint)


@lru_cache(maxsize=1)
def _script_pipeline_cached(profile_fingerprint: str) -> tuple[Any, Any]:
    """Lazy, single initialization using an exact local snapshot path."""
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        raise NicheVideoError(
            "Install the [niche-models] runtime for open-weight inference"
        ) from exc
    _public_cpu_smoke_phase("pinned_snapshot_resolution_start")
    snapshot = _pinned_qwen_snapshot()
    _public_cpu_smoke_phase("pinned_snapshot_resolution_complete")
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    _public_cpu_smoke_phase("tokenizer_initialization_start")
    tokenizer = transformers.AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    _public_cpu_smoke_phase("tokenizer_initialization_complete")
    _public_cpu_smoke_phase("cpu_weight_loading_start")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, dtype="auto"
    ).eval()
    _public_cpu_smoke_phase("cpu_weight_loading_complete")
    if _qwen_device() == "cuda":
        if not torch.cuda.is_available():
            raise NicheVideoError("CUDA device requested but unavailable")
        model = model.to("cuda")
    return model, tokenizer


@dataclass(frozen=True, slots=True)
class PinnedQwenRawReceipt:
    """Created only from model.generate token IDs, never from injected text."""

    raw: str
    raw_sha256: str
    generated_token_count: int
    terminal_token_id: int
    eos_decision: str


def _run_pinned_qwen(
    request: dict[str, object], *, normalize_script: bool, capture_receipt: bool = False
) -> str | PinnedQwenRawReceipt:
    """Decode exact generated tokens; book mode retains unmodified raw response."""
    torch = importlib.import_module("torch")

    model, tokenizer = _script_pipeline()
    # No decoder state is shared with the independent script-generation callers.
    kwargs: dict[str, Any] = {}
    if not normalize_script:
        from . import constrained

        kwargs["prefix_allowed_tokens_fn"] = constrained.prefix_for(
            request, tokenizer, model_profile.active().revision
        )
    torch.manual_seed(int(cast(int, request["seed"])))
    inputs = tokenizer.apply_chat_template(
        request["messages"],
        tokenize=True,
        add_generation_prompt=True,
        **model_profile.active().template_kwargs,
        return_dict=True,
        return_tensors="pt",
    )
    _public_cpu_smoke_phase("real_generation_start")
    inputs = _to_model_device(inputs, model)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=int(cast(int, request["max_new_tokens"])),
            **model_profile.active().sampling,
            pad_token_id=tokenizer.eos_token_id,
            **kwargs,
        )
    _public_cpu_smoke_phase("real_generation_complete")
    output_ids = output[0][inputs["input_ids"].shape[-1] :]
    result: str = tokenizer.decode(output_ids, skip_special_tokens=True)
    if not normalize_script:
        # Match the *effective* pinned model EOS set without changing generation.
        # Qwen generation_config can carry more than tokenizer.eos_token_id.
        configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        if configured is None:
            configured = tokenizer.eos_token_id
        if isinstance(configured, int):
            end_tokens = {configured}
        elif isinstance(configured, (tuple, list)) and all(
            isinstance(token_id, int) for token_id in configured
        ):
            end_tokens = set(configured)
        else:
            end_tokens = set()
        last_token = int(output_ids[-1]) if len(output_ids) else None
        if last_token is None or last_token not in end_tokens:
            raise IncompleteConstrainedResponse(result)
    if capture_receipt:
        if normalize_script:
            raise NicheVideoError("model token receipt is only valid for original book mining")
        return PinnedQwenRawReceipt(
            raw=result,
            raw_sha256=hashlib.sha256(result.encode("utf-8")).hexdigest(),
            generated_token_count=len(output_ids),
            terminal_token_id=int(output_ids[-1]),
            eos_decision="EOS_VERIFIED",
        )
    if normalize_script:
        result = result.strip()
        if result.startswith(chr(96) * 3):
            result = "\n".join(result.splitlines()[1:-1]).strip()
    return result


def _chapter_tokenizer_material(tokenizer: Any) -> dict[str, str]:
    """Bind the effective HF chat template AND pinned tokenizer config bytes.

    Use the same exact-revision snapshot for public preflight and final CPU
    generation. Do not trust a symbolic template name or token count alone.
    """

    path = Path(str(getattr(tokenizer, "name_or_path", "")))
    try:
        snapshot = path.resolve(strict=True)
        if (
            snapshot.name != model_profile.active().revision
            or snapshot.parent.name != "snapshots"
            or snapshot.parent.parent.name
            != "models--" + model_profile.active().model.replace("/", "--")
        ):
            raise PinnedQwenSnapshotError("chapter tokenizer snapshot identity mismatch")
        raw = (snapshot / "tokenizer_config.json").read_bytes()
        if not 0 < len(raw) <= 1_000_000:
            raise PinnedQwenSnapshotError("chapter tokenizer config absent or oversized")
        token_json = (snapshot / "tokenizer.json").read_bytes()
        if hashlib.sha256(token_json).hexdigest() != constrained.PINNED_TOKENIZER_SHA256:
            raise PinnedQwenSnapshotError("chapter tokenizer JSON identity changed")
        template = getattr(tokenizer, "chat_template", None)
        if not isinstance(template, (str, dict)) or not template:
            raise PinnedQwenSnapshotError("effective chapter chat template absent")
        template_raw = json.dumps(
            template, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return {
            "tokenizer_config_sha256": hashlib.sha256(raw).hexdigest(),
            "chat_template_sha256": hashlib.sha256(template_raw).hexdigest(),
            "tokenizer_json_sha256": constrained.PINNED_TOKENIZER_SHA256,
        }
    except PinnedQwenSnapshotError:
        raise
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise PinnedQwenSnapshotError("chapter effective tokenizer provenance unavailable") from exc


def _chapter_prompt_encoding(
    messages: list[dict[str, str]], *, tokenizer: Any | None = None
) -> dict[str, object]:
    """Exact rendered token IDs plus effective-template/config digests; no weights."""
    if tokenizer is None:
        tokenizer, _limit = _public_book_context()
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        **model_profile.active().template_kwargs,
    )
    if (
        not isinstance(ids, list)
        or not ids
        or len(ids) > model_profile.active().context_window
        or any(type(value) is not int or value < 0 for value in ids)
    ):
        raise PinnedQwenSnapshotError("chapter rendered input token encoding invalid")
    return {"token_ids": ids, **_chapter_tokenizer_material(tokenizer)}


def _chapter_encoding_receipt(encoding: dict[str, object]) -> dict[str, str]:
    """Canonical SHA-256 encoding shared by preflight, stage key and generation."""
    ids = encoding["token_ids"]
    if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
        raise PinnedQwenSnapshotError("chapter verified token sequence malformed")
    digest = hashlib.sha256(
        (json.dumps(ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "input_token_ids_sha256": digest,
        "tokenizer_config_sha256": cast(str, encoding["tokenizer_config_sha256"]),
        "chat_template_sha256": cast(str, encoding["chat_template_sha256"]),
        "tokenizer_json_sha256": cast(str, encoding["tokenizer_json_sha256"]),
    }


def _cached_chapter_tokenizer_material() -> dict[str, str]:
    """Model-free producer provenance, resolved from the pinned local HF cache."""
    home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
    snapshot = (
        home
        / "hub"
        / ("models--" + model_profile.active().model.replace("/", "--"))
        / "snapshots"
        / model_profile.active().revision
    )
    config = snapshot / "tokenizer_config.json"
    tokenizer_file = snapshot / "tokenizer.json"
    if not config.is_file() or not tokenizer_file.is_file():
        return {"status": "TOKENIZER_NOT_CACHED"}
    try:
        raw = config.read_bytes()
        if not 0 < len(raw) <= 1_000_000:
            return {"status": "TOKENIZER_CONFIG_UNVERIFIED"}
        if hashlib.sha256(tokenizer_file.read_bytes()).hexdigest() != (
            constrained.PINNED_TOKENIZER_SHA256
        ):
            return {"status": "TOKENIZER_JSON_UNVERIFIED"}
        return {
            "status": "PINNED_PUBLIC_TOKENIZER",
            "tokenizer_config_sha256": hashlib.sha256(raw).hexdigest(),
            "tokenizer_json_sha256": constrained.PINNED_TOKENIZER_SHA256,
        }
    except (OSError, ValueError):
        return {"status": "TOKENIZER_CONFIG_UNVERIFIED"}


def _chapter_context_tokens(
    request: dict[str, object], *, tokenizer_context: tuple[Any, int] | None = None
) -> int:
    """Preflight the full real chapter chat with the pinned tokenizer, before weights."""
    if (
        request.get("model") != model_profile.active().model
        or request.get("revision") != model_profile.active().revision
        or type(request.get("seed")) is not int
        or request["seed"] not in (42, 43)
        or type(request.get("max_new_tokens")) is not int
        or not model_profile.output_bounds()[0]
        <= cast(int, request["max_new_tokens"])
        <= model_profile.output_bounds()[1]
    ):
        raise NicheVideoError("chapter pinned model request or output budget mismatch")
    tokenizer, limit = _public_book_context() if tokenizer_context is None else tokenizer_context
    try:
        if "verified_input_encoding" in request:
            encoded = _chapter_prompt_encoding(
                cast(list[dict[str, str]], request["messages"]), tokenizer=tokenizer
            )
            receipt = _chapter_encoding_receipt(encoded)
            if receipt != request["verified_input_encoding"]:
                raise NicheVideoError("chapter rendered input tokens or template changed")
            count = len(cast(list[int], encoded["token_ids"]))
        else:
            tokens = tokenizer.apply_chat_template(
                request["messages"],
                tokenize=True,
                add_generation_prompt=True,
                **model_profile.active().template_kwargs,
            )
            count = len(tokens)
    except NicheVideoError:
        raise
    except (ValueError, TypeError, KeyError, RuntimeError) as exc:
        raise NicheVideoError("chapter pinned tokenizer preflight failed") from exc
    if count <= 0 or count + cast(int, request["max_new_tokens"]) > limit:
        raise NicheVideoError(
            "chapter complete source, chat template and output exceed pinned token context"
        )
    return count


def _local_chapter_completion(request: dict[str, object]) -> PinnedQwenRawReceipt:
    """Real CPU chapter response with raw-bound generated token/EOS evidence."""
    _chapter_context_tokens(request)
    # Kokoro may have been used just before the direction stage. Never retain
    # both large inference graphs when switching between generation phases.
    from . import narration

    narration._kokoro_pipeline.cache_clear()
    import gc

    gc.collect()
    torch = importlib.import_module("torch")
    model, tokenizer = _script_pipeline()
    if _device_type(model) != _qwen_device():
        raise NicheVideoError("chapter requires pinned Qwen weights on the configured device")
    torch.manual_seed(int(cast(int, request["seed"])))
    inputs = tokenizer.apply_chat_template(
        request["messages"],
        tokenize=True,
        add_generation_prompt=True,
        **model_profile.active().template_kwargs,
        return_dict=True,
        return_tensors="pt",
    )
    if "verified_input_encoding" in request:
        # The very tensor passed to generate() must match the public-only
        # preflight, including equal-length but different token sequences.
        identity = _chapter_encoding_receipt(
            {
                "token_ids": inputs["input_ids"][0].tolist(),
                **_chapter_tokenizer_material(tokenizer),
            }
        )
        if identity != request["verified_input_encoding"]:
            raise NicheVideoError("chapter generator rendered tokens differ from reserved input")
    if (
        inputs["input_ids"].shape[-1] + cast(int, request["max_new_tokens"])
        > model_profile.active().context_window
    ):
        raise NicheVideoError("chapter final token context differs from preflight")
    inputs = _to_model_device(inputs, model)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=cast(int, request["max_new_tokens"]),
            **model_profile.active().sampling,
            pad_token_id=tokenizer.eos_token_id,
        )
    ids = output[0][inputs["input_ids"].shape[-1] :]
    raw: str = tokenizer.decode(ids, skip_special_tokens=True)
    configured = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if configured is None:
        configured = tokenizer.eos_token_id
    eos = (
        {configured}
        if type(configured) is int
        else (set(configured) if isinstance(configured, (list, tuple)) else set())
    )
    if len(ids) == 0 or int(ids[-1]) not in eos:
        raise IncompleteConstrainedResponse(raw)
    return PinnedQwenRawReceipt(
        raw=raw,
        raw_sha256=hashlib.sha256(raw.encode("utf-8")).hexdigest(),
        generated_token_count=len(ids),
        terminal_token_id=int(ids[-1]),
        eos_decision="EOS_VERIFIED",
    )


# Pin the actual implementation's identity; an injected callable must never
# acquire production token/EOS provenance just by replacing a module attribute.
_ORIGINAL_LOCAL_CHAPTER_COMPLETION = _local_chapter_completion


def _local_script_completion(request: dict[str, object]) -> str:
    """Generate locally; keep weights resident for bounded script repair."""
    value = _run_pinned_qwen(request, normalize_script=True)
    if not isinstance(value, str):
        raise NicheVideoError("script mode must return plain generated text")
    return value


def _local_book_generate(
    request: dict[str, object], *, capture_receipt: bool
) -> str | PinnedQwenRawReceipt:
    """Opt-in only: validate source choices before pinned CPU-only inference."""
    from . import constrained

    # No eligible source span must never quietly disable constrained decoding.
    constrained.decode_schema(request)
    constrained._decoder()
    public_tokenizer, context_limit = _public_book_context()
    constrained.check_prompt_context(request, public_tokenizer, context_limit)
    model, _tokenizer = _script_pipeline()
    if _device_type(model) != _qwen_device():
        raise NicheVideoError(
            "local book completion requires pinned Qwen weights on the configured device"
        )
    return _run_pinned_qwen(request, normalize_script=False, capture_receipt=capture_receipt)


def _local_book_completion(request: dict[str, object]) -> str:
    """Preserve existing plain-text book miner API for existing callers."""
    value = _local_book_generate(request, capture_receipt=False)
    if not isinstance(value, str):
        raise NicheVideoError("book miner expected original plain text")
    return value


# Private selected/real miner uses authenticated token IDs from this exact
# unmodified provider path, not an injected callable's claimed metadata.
_ORIGINAL_LOCAL_BOOK_COMPLETION = _local_book_completion


def _local_book_completion_with_receipt(request: dict[str, object]) -> PinnedQwenRawReceipt:
    value = _local_book_generate(request, capture_receipt=True)
    if not isinstance(value, PinnedQwenRawReceipt):
        raise NicheVideoError("real book miner expected generated token/EOS receipt")
    return value


def _script_feedback(raw: object, error: Exception) -> str:
    """Give the local model measured word counts, not only a generic error."""
    message = str(error)
    if isinstance(raw, dict) and isinstance(raw.get("scenes"), list):
        counts = [
            len(scene["narration"].split())
            if isinstance(scene, dict) and isinstance(scene.get("narration"), str)
            else None
            for scene in raw["scenes"]
        ]
        total = sum(count for count in counts if count is not None)
        message += f"; total_narration_words={total}; scene_words={counts}"
    return message[:900]


def _revision_request(original: dict[str, object], previous: str, error: str) -> dict[str, object]:
    messages = original["messages"]
    if not isinstance(messages, list):
        raise NicheVideoError("internal request lost original script context")
    return {
        **original,
        "messages": [
            *messages,
            {"role": "assistant", "content": previous[:6000]},
            {
                "role": "user",
                "content": (
                    "Revise your previous JSON to resolve this exact violation: "
                    f"{error}. Preserve the original niche, topic, approved facts, "
                    "honest hook and complete payoff. Do not invent claims. "
                    "Return JSON only; aim for four scenes of 20-25 narration "
                    "words each, totalling 80-100 narration words."
                ),
            },
        ],
    }


def _compact_book_sentences(candidate: object) -> tuple[object, bool]:
    """Tighten a book lesson using ONLY complete, original model sentences.

    No word truncation, invented facts, paraphrasing or changed scene order.
    Source and creative-quality checks remain human approval gates.
    """
    if not isinstance(candidate, dict) or set(candidate) != {"title", "scenes"}:
        return candidate, False
    raw_scenes = candidate["scenes"]
    if not isinstance(raw_scenes, list) or not 3 <= len(raw_scenes) <= 5:
        return candidate, False
    if not all(
        isinstance(scene, dict)
        and set(scene) == {"headline", "narration"}
        and isinstance(scene["narration"], str)
        for scene in raw_scenes
    ):
        return candidate, False
    counts = [len(scene["narration"].split()) for scene in raw_scenes]
    if sum(counts) <= 125 and all(6 <= count <= 45 for count in counts):
        return candidate, False

    # Mandatory content: each scene's opening and the closing scene's last
    # complete sentence. Allow variable scene budgets, then select optional
    # complete sentences without exceeding the strict episode word ceiling.
    sentence_lists = [
        re.split(r"(?<=[.!?])\s+", scene["narration"].strip()) for scene in raw_scenes
    ]
    chosen: list[set[int]] = []
    lengths: list[int] = []
    for index, sentences in enumerate(sentence_lists):
        if not sentences or any(not sentence for sentence in sentences):
            raise NicheVideoError("book script contains empty sentence boundaries")
        required = {0}
        if index == len(sentence_lists) - 1:
            required.add(len(sentences) - 1)
        word_count = sum(len(sentences[i].split()) for i in required)
        if word_count > 45:
            raise NicheVideoError("book opening or final payoff exceeds the per-scene limit")
        chosen.append(required)
        lengths.append(word_count)
    total = sum(lengths)
    if total > 125:
        raise NicheVideoError("mandatory complete sentences exceed narration budget")

    for index, sentences in enumerate(sentence_lists):
        # Nonfinal scenes target 35 words; the finale may need the full 45 to
        # preserve its spoken takeaway. Favor earlier contextual sentences.
        scene_target = 45 if index == len(sentence_lists) - 1 else 35
        for i in range(1, len(sentences)):
            if i in chosen[index]:
                continue
            count = len(sentences[i].split())
            if lengths[index] + count <= scene_target and total + count <= 120:
                chosen[index].add(i)
                lengths[index] += count
                total += count

    edited: list[dict[str, str]] = []
    for index, scene in enumerate(raw_scenes):
        if not 6 <= lengths[index] <= 45:
            raise NicheVideoError("complete-sentence edit could not preserve scene length")
        narration = " ".join(sentence_lists[index][i] for i in sorted(chosen[index]))
        edited.append({"headline": scene["headline"], "narration": narration})
    result = {"title": candidate["title"], "scenes": edited}
    if sum(len(scene["narration"].split()) for scene in edited) < 45:
        raise NicheVideoError("complete-sentence edit was too brief")
    return result, result != candidate


def _script_fingerprint(
    config: NicheConfig, *, evaluation_context: dict[str, object] | None = None
) -> str:
    return _fingerprint(
        {
            "contract": BOOK_EDIT_CONTRACT if config.format == "book_lesson" else SCRIPT_CONTRACT,
            "request": _script_request(config, evaluation_context=evaluation_context),
        }
    )


def _recover_book_script(
    config: NicheConfig, cache_root: Path, audit_path: Path, run_id: str
) -> str:
    """Validate and cache a genuine Qwen draft preserved by a failed GitHub run."""
    if config.format != "book_lesson" or not run_id.isdecimal():
        raise NicheVideoError("audit recovery requires a book lesson and numeric run ID")
    records = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(records, list) or not records:
        raise NicheVideoError("Qwen3 recovery audit must contain recorded attempts")
    first = records[0]
    if (
        not isinstance(first, dict)
        or first.get("accepted") is not False
        or not isinstance(first.get("completion"), str)
    ):
        raise NicheVideoError("recovery requires the original rejected Qwen3 completion")
    completion = first["completion"]
    candidate = json.loads(completion)
    edited_candidate, changed = _compact_book_sentences(candidate)
    episode = Episode.from_json(edited_candidate)
    digest = _script_fingerprint(config)
    path = cache_root / "scripts" / f"{digest}.json"
    if path.is_file():
        if Episode.from_json(json.loads(path.read_text(encoding="utf-8"))) != episode:
            raise NicheVideoError("recovery differs from the current validated script cache")
    else:
        _write_atomic(
            path, json.dumps(episode.to_dict(), ensure_ascii=False, indent=2).encode("utf-8")
        )
    provenance = {
        "source": "github_actions_rejected_qwen3_attempt",
        "source_run_id": run_id,
        "source_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "original_completion_sha256": hashlib.sha256(completion.encode()).hexdigest(),
        "editor_contract": BOOK_EDIT_CONTRACT,
        "edited": changed,
        "narration_words": sum(len(scene.narration.split()) for scene in episode.scenes),
        "human_factual_review": "PENDING",
    }
    _write_atomic(
        cache_root / "script_recoveries" / f"{digest}.json",
        json.dumps(provenance, indent=2).encode("utf-8"),
    )
    print(f"Recovered authenticated Qwen script from Actions run {run_id}")
    return digest


def generate_episode(
    config: NicheConfig,
    cache_root: Path,
    *,
    completion: Callable[[dict[str, object]], str] | None = None,
    evaluation_context: dict[str, object] | None = None,
) -> tuple[Episode, str]:
    original = _script_request(config, evaluation_context=evaluation_context)
    digest = _script_fingerprint(config, evaluation_context=evaluation_context)
    path = cache_root / "scripts" / f"{digest}.json"
    if path.is_file():
        return Episode.from_json(json.loads(path.read_text(encoding="utf-8"))), digest

    request = original
    audit_path = cache_root / "script_diagnostics" / f"{digest}.json"
    attempts: list[dict[str, object]] = []
    for attempt in range(1, MAX_SCRIPT_ATTEMPTS + 1):
        response = (completion or _local_script_completion)(request)
        raw: object = None
        try:
            if not response:
                raise NicheVideoError("local script model returned no content")
            raw = json.loads(response)
            edited = False
            if config.format == "book_lesson":
                raw, edited = _compact_book_sentences(raw)
            episode = Episode.from_json(raw)
        except (json.JSONDecodeError, NicheVideoError) as exc:
            error = _script_feedback(raw, exc)
            attempts.append(
                {
                    "attempt": attempt,
                    "accepted": False,
                    "feedback": error,
                    "completion": response[:8000],
                }
            )
            _write_atomic(
                audit_path,
                json.dumps(attempts, ensure_ascii=False, indent=2).encode("utf-8"),
            )
            print(
                f"Qwen3 validation attempt {attempt}/{MAX_SCRIPT_ATTEMPTS}: {error}",
                file=sys.stderr,
            )
            if attempt == MAX_SCRIPT_ATTEMPTS:
                raise NicheVideoError(
                    f"script rejected after {MAX_SCRIPT_ATTEMPTS} attempts: {error}"
                ) from exc
            request = _revision_request(original, response, error)
            continue
        attempts.append(
            {
                "attempt": attempt,
                "accepted": True,
                "complete_sentence_edit": edited if config.format == "book_lesson" else False,
                "narration_words": sum(len(scene.narration.split()) for scene in episode.scenes),
            }
        )
        _write_atomic(
            audit_path,
            json.dumps(attempts, ensure_ascii=False, indent=2).encode("utf-8"),
        )
        _write_atomic(
            path,
            json.dumps(episode.to_dict(), ensure_ascii=False, indent=2).encode("utf-8"),
        )
        print(f"Qwen3 script accepted on attempt {attempt}/{MAX_SCRIPT_ATTEMPTS}")
        return episode, digest

    raise NicheVideoError("bounded script generation unexpectedly exhausted")


def clear_model_caches() -> None:
    """Drop every cached tokenizer and model (all profiles)."""
    _public_book_context_cached.cache_clear()
    _script_pipeline_cached.cache_clear()


_public_book_context.cache_clear = _public_book_context_cached.cache_clear  # type: ignore[attr-defined]
_script_pipeline.cache_clear = _script_pipeline_cached.cache_clear  # type: ignore[attr-defined]
