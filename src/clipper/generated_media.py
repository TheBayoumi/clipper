from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from .models import CampaignBrief, PolicyAction


class GeneratedMediaBlocked(RuntimeError):
    """Raised before a generator can run when campaign policy does not allow it."""


@dataclass(frozen=True, slots=True)
class GeneratedMediaRequest:
    quality_moment_id: str
    prompt: str
    source_evidence_ids: tuple[str, ...]
    silent: bool = True

    def __post_init__(self) -> None:
        if not self.quality_moment_id.strip() or not self.prompt.strip():
            raise ValueError("generated media request requires moment identity and prompt")
        if not self.source_evidence_ids:
            raise ValueError("generated media request requires source evidence provenance")
        if not self.silent:
            raise ValueError("generated media assets must be silent illustrations")


@dataclass(frozen=True, slots=True)
class GeneratedMediaAsset:
    path: Path
    provider: str
    model_id: str
    model_revision: str
    request: GeneratedMediaRequest
    sha256: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["path"] = str(self.path)
        return payload


class GeneratedMediaProvider(Protocol):
    @property
    def provider_name(self) -> str: ...

    @property
    def model_id(self) -> str: ...

    @property
    def model_revision(self) -> str: ...

    def generate(self, request: GeneratedMediaRequest, output_path: Path) -> Path: ...


def generated_media_policy(brief: CampaignBrief) -> PolicyAction:
    return brief.acceptance_policy.ai_generated_source_video


def generate_policy_gated_media(
    brief: CampaignBrief,
    provider: GeneratedMediaProvider,
    request: GeneratedMediaRequest,
    output_path: Path,
) -> GeneratedMediaAsset:
    """Invoke a generator only when policy explicitly ALLOWs synthetic visuals."""

    policy = generated_media_policy(brief)
    if policy == "forbid":
        raise GeneratedMediaBlocked("campaign policy forbids synthetic visual generation")
    if policy == "escalate":
        raise GeneratedMediaBlocked("synthetic visual generation requires human policy escalation")
    generated = provider.generate(request, output_path)
    if generated != output_path or not generated.is_file() or generated.stat().st_size <= 0:
        raise RuntimeError("generated media provider did not produce the requested artifact")
    digest = hashlib.sha256(generated.read_bytes()).hexdigest()
    return GeneratedMediaAsset(
        path=generated,
        provider=provider.provider_name,
        model_id=provider.model_id,
        model_revision=provider.model_revision,
        request=request,
        sha256=digest,
    )
