from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any


def content_fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def structural_contract_fingerprint(
    name: str,
    *types: type[Any],
    exclude_fields: tuple[str, ...] = (),
) -> str:
    """Fingerprint a serialized contract from its annotated structure, not a release number."""
    if not name.strip() or not types:
        raise ValueError("structural contract requires a name and at least one type")
    excluded = set(exclude_fields)
    structures: dict[str, dict[str, str]] = {}
    for value_type in types:
        annotations = getattr(value_type, "__annotations__", {})
        structures[value_type.__qualname__] = {
            str(field_name): str(annotation)
            for field_name, annotation in annotations.items()
            if field_name not in excluded
        }
    return content_fingerprint({"name": name, "types": structures})


@dataclass(frozen=True, slots=True)
class StageContract:
    name: str
    contract: dict[str, object]
    relevant_policy: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("stage contract requires a name")
        if not self.contract:
            raise ValueError("stage contract material cannot be empty")

    @property
    def contract_hash(self) -> str:
        return content_fingerprint(
            {
                "name": self.name,
                "contract": self.contract,
                "relevant_policy": self.relevant_policy,
            }
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["contract_hash"] = self.contract_hash
        return payload


@dataclass(frozen=True, slots=True)
class StageIdentity:
    stage_name: str
    source_hash: str
    contract_hash: str
    dependency_output_hashes: tuple[str, ...] = ()
    model_revision: str | None = None
    decoding_parameters: dict[str, object] = field(default_factory=dict)
    relevant_policy_hash: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.stage_name.strip()
            or not self.source_hash.strip()
            or not self.contract_hash.strip()
        ):
            raise ValueError("stage identity requires stage, source, and contract hashes")
        if any(not value.strip() for value in self.dependency_output_hashes):
            raise ValueError("dependency output fingerprints cannot be empty")

    @property
    def cache_key(self) -> str:
        return content_fingerprint(
            {
                "source_hash": self.source_hash,
                "stage_name": self.stage_name,
                "stage_contract_hash": self.contract_hash,
                "dependency_output_hashes": self.dependency_output_hashes,
                "model_revision": self.model_revision,
                "decoding_parameters": self.decoding_parameters,
                "relevant_policy_hash": self.relevant_policy_hash,
            }
        )

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["dependency_output_hashes"] = list(self.dependency_output_hashes)
        payload["cache_key"] = self.cache_key
        return payload


def stage_identity(
    contract: StageContract,
    *,
    source_hash: str,
    dependency_output_hashes: tuple[str, ...] = (),
    model_revision: str | None = None,
    decoding_parameters: dict[str, object] | None = None,
) -> StageIdentity:
    relevant_policy_hash = (
        content_fingerprint(contract.relevant_policy) if contract.relevant_policy else None
    )
    return StageIdentity(
        stage_name=contract.name,
        source_hash=source_hash,
        contract_hash=contract.contract_hash,
        dependency_output_hashes=dependency_output_hashes,
        model_revision=model_revision,
        decoding_parameters=dict(decoding_parameters or {}),
        relevant_policy_hash=relevant_policy_hash,
    )
