from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generic, Literal, Protocol, TypeVar

from ..canonical import CanonicalTimeline
from ..stage_contracts import content_fingerprint

T = TypeVar("T")
ProfileName = Literal["local-lite", "balanced", "quality"]


class EditorialCapacityError(RuntimeError):
    """Editorial request cannot fit the active model/runtime capacity."""

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        self.details = dict(details or {})
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    model_id: str
    revision: str
    quantization: str
    inference_engine: str
    prompt_version: str = "none"
    schema_version: str = "none"

    def cache_fingerprint(self, *, sampling: dict[str, Any] | None = None) -> str:
        """Fingerprint exact execution identity and contracts without release counters."""
        return content_fingerprint({**asdict(self), "sampling": sampling or {}})

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class InferenceUsage:
    provider: str
    started_at: str
    duration_seconds: float
    gpu_type: str | None = None
    gpu_seconds: float = 0.0
    peak_vram_mb: float | None = None
    input_units: int = 0
    output_units: int = 0
    estimated_cost_usd: float = 0.0
    runtime: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProviderResult(Generic[T]):
    value: T
    model: ModelIdentity
    usage: InferenceUsage
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class ComputeProfile:
    name: ProfileName
    editorial_location: Literal["local", "modal"]
    vision_location: Literal["local", "modal"]
    allow_large_vlm_escalation: bool


def compute_profile(name: ProfileName) -> ComputeProfile:
    if name == "local-lite":
        return ComputeProfile(name, "local", "local", False)
    if name == "balanced":
        return ComputeProfile(name, "modal", "modal", False)
    if name == "quality":
        return ComputeProfile(name, "modal", "modal", True)
    raise ValueError(f"unsupported compute profile: {name}")


class TranscriptionProvider(Protocol):
    identity: ModelIdentity

    def transcribe(
        self, source: Path, *, video_id: str, source_hash: str
    ) -> ProviderResult[CanonicalTimeline]: ...


class AlignmentProvider(Protocol):
    identity: ModelIdentity

    def align(
        self, source: Path, timeline: CanonicalTimeline
    ) -> ProviderResult[CanonicalTimeline]: ...


class DiarizationProvider(Protocol):
    identity: ModelIdentity

    def diarize(
        self, source: Path, timeline: CanonicalTimeline
    ) -> ProviderResult[CanonicalTimeline]: ...


class EditorialProvider(Protocol):
    identity: ModelIdentity

    def complete_json(
        self, *, task: str, payload: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]: ...


class VisionProvider(Protocol):
    identity: ModelIdentity

    def inspect(
        self, *, task: str, frames: list[Path], context: dict[str, Any]
    ) -> ProviderResult[dict[str, Any]]: ...
