"""Offline, versioned stage-selection boundary for the source-driven chapter pipeline."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from clipper.stage_contracts import StageContract, content_fingerprint, stage_identity

from . import model_profile

SCHEMA = "clipper-book-pipeline-config-v1"
REPORT_SCHEMA = "clipper-book-pipeline-plan-v1"
DESCRIPTOR_SCHEMA = "clipper-book-pipeline-input-v1"
MAX_JSON = 256_000
MAX_ARTIFACT = 2_000_000
DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class PlanError(ValueError):
    """Fixed-category offline planning refusal; never includes source prose."""


@dataclass(frozen=True)
class StageSpec:
    dependencies: tuple[str, ...]
    schema: str
    provider: str
    support: str = "IMPLEMENTED"
    model: str | None = None
    revision: str | None = None
    limit_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()


def registry() -> dict[str, StageSpec]:
    """Data only: no model/provider imports, factories or initialization."""

    def text(
        deps: tuple[str, ...], schema: str, *limits: str, optional: tuple[str, ...] = ()
    ) -> StageSpec:
        return StageSpec(
            deps,
            schema,
            "pinned-qwen-cpu",
            model=model_profile.active().model,
            revision=model_profile.active().revision,
            limit_fields=("max_new_tokens", "max_attempts", "max_runtime_seconds", *limits),
            optional_fields=optional,
        )

    return {
        "source": StageSpec((), "clipper-private-selected-chapter-evidence-v1", "owner-input"),
        "argument": text(
            ("source",),
            "clipper-chapter-hybrid-argument-v1",
            optional=("argument_batch_tokens",),
        ),
        "audits": text(("source", "argument"), "clipper-chapter-hybrid-local-audit-v1"),
        "reconcile": text(
            ("source", "argument", "audits"), "clipper-chapter-hybrid-brief-v1", "max_beats"
        ),
        "human_review": StageSpec(
            ("source", "reconcile"), "clipper-chapter-hybrid-human-review-v1", "human"
        ),
        "adaptation": text(
            ("reconcile", "human_review"), "clipper-chapter-adaptation-v1", "max_beats"
        ),
        "script": text(
            ("reconcile", "human_review", "adaptation"),
            "clipper-chapter-script-graph-v1",
            "max_scenes",
            "max_words",
            "max_beats",
        ),
        "narration": StageSpec(
            ("script",),
            "kokoro-24khz-wav-v1",
            "kokoro-cpu",
            model="hexgrad/Kokoro-82M",
            revision="v1.0",
            limit_fields=("voice",),
        ),
        "direction": text(
            ("script", "adaptation", "source", "reconcile"),
            "clipper-chapter-continuity-shots-v2",
            "max_scenes",
            "max_words",
            "max_shots_per_scene",
        ),
        "visuals": StageSpec(("direction",), "unported-book-visuals", "none", "UNSUPPORTED"),
        "video": StageSpec(("visuals", "narration"), "unported-book-video", "none", "UNSUPPORTED"),
        "render": StageSpec(("video",), "unported-book-render", "none", "UNSUPPORTED"),
        "publish": StageSpec(("render",), "unported-book-publish", "none", "UNSUPPORTED"),
    }


TARGETS = {
    "source": ("source",),
    "brief": ("reconcile",),
    "script": ("script",),
    "narration": ("narration",),
    "direction": ("direction",),
    "preproduction": ("narration", "direction"),
    "visuals": ("visuals",),
    "video": ("video",),
    "render": ("render",),
    "publish": ("publish",),
}
AUTO_OUTPUT = "auto"
BATCH_TOKENS_RANGE = (512, 16384)
OPTIONAL_OFF_ZERO = frozenset({"argument_batch_tokens"})
RANGES = {
    "max_attempts": (1, 2),
    "max_runtime_seconds": (60, 900),
    "max_beats": (2, 24),
    "max_scenes": (2, 20),
    "max_words": (130, 2400),
    "max_shots_per_scene": (1, 4),
}
VOICES = {"af_heart", "af_bella", "am_adam", "am_michael", "bf_emma"}
# A verifier is trusted application code, never a name or flag supplied by the config.
# It must independently replay the stage contract and provenance against exact inputs.
Verifier = Callable[[bytes, dict[str, Any], Mapping[str, bytes]], bool]


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise PlanError(reason)


def is_digest(value: object) -> bool:
    return isinstance(value, str) and DIGEST.fullmatch(value) is not None


def strict_json(raw: bytes) -> dict[str, Any]:
    def unique(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result

    def invalid_constant(_value: str) -> None:
        raise PlanError("nonfinite JSON value")

    try:
        value = json.loads(raw, object_pairs_hook=unique, parse_constant=invalid_constant)
    except (ValueError, UnicodeError, RecursionError):
        raise PlanError("invalid or duplicate JSON") from None
    if not isinstance(value, dict):
        raise PlanError("JSON object required")
    return dict(value)


def ordered_stages(specs: Mapping[str, StageSpec]) -> list[str]:
    result: list[str] = []
    visiting: set[str] = set()

    def visit(name: str) -> None:
        require(name in specs, "unknown stage dependency")
        require(name not in visiting, "stage dependency cycle")
        if name in result:
            return
        visiting.add(name)
        deps = specs[name].dependencies
        require(len(set(deps)) == len(deps), "duplicate stage dependency")
        for dep in deps:
            visit(dep)
        visiting.remove(name)
        result.append(name)

    for name in specs:
        visit(name)
    return result


def _read(path: Path, maximum: int) -> bytes:
    require(path.is_file() and not path.is_symlink(), "artifact absent or redirected")
    with path.open("rb") as handle:
        raw = handle.read(maximum + 1)
    require(0 < len(raw) <= maximum, "artifact byte bound exceeded")
    return raw


def _descriptor(value: object) -> None:
    if not isinstance(value, dict):
        raise PlanError("invalid input descriptor")
    require(
        set(value) == {"schema", "identity", "output_sha256", "path"}
        and value["schema"] == DESCRIPTOR_SCHEMA
        and is_digest(value["output_sha256"])
        and isinstance(value["identity"], dict)
        and isinstance(value["path"], str),
        "input descriptor shape mismatch",
    )
    path = Path(value["path"])
    require(
        bool(path.parts)
        and not path.is_absolute()
        and all(part not in {"..", "."} for part in path.parts)
        and "\\" not in value["path"]
        and path.as_posix() == value["path"],
        "input descriptor path must be relative and canonical",
    )
    identity = value["identity"]
    require(
        set(identity)
        == {
            "stage_name",
            "source_hash",
            "contract_hash",
            "dependency_output_hashes",
            "model_revision",
            "decoding_parameters",
            "relevant_policy_hash",
            "cache_key",
        }
        and all(is_digest(identity[key]) for key in ("source_hash", "contract_hash", "cache_key"))
        and isinstance(identity["dependency_output_hashes"], list)
        and all(is_digest(item) for item in identity["dependency_output_hashes"])
        and isinstance(identity["decoding_parameters"], dict)
        and (identity["model_revision"] is None or isinstance(identity["model_revision"], str))
        and (
            identity["relevant_policy_hash"] is None or is_digest(identity["relevant_policy_hash"])
        ),
        "input stage identity malformed",
    )


def config_profile_name(raw: object) -> str:
    """Model profile named by a configuration; absent means the default profile."""
    if isinstance(raw, dict) and "model_profile" in raw:
        name = raw["model_profile"]
        require(isinstance(name, str) and name in model_profile.profiles(), "unknown model profile")
        return str(name)
    return model_profile.DEFAULT_PROFILE


@contextmanager
def profile_scope(raw: object) -> Iterator[None]:
    """Run a block with the model profile the configuration selects.

    An unknown name is left to validate_config to reject, so the error stays the
    configuration error rather than a lookup failure.
    """
    try:
        name = config_profile_name(raw)
    except PlanError:
        name = model_profile.DEFAULT_PROFILE
    with model_profile.use(name):
        yield


def validate_config(raw: object, specs: Mapping[str, StageSpec]) -> dict[str, Any]:
    ordered_stages(specs)
    if not isinstance(raw, dict):
        raise PlanError("pipeline config object required")
    require(
        set(raw) - {"model_profile"} == {"schema", "source_sha256", "targets", "stages"}
        and raw["schema"] == SCHEMA
        and is_digest(raw["source_sha256"]),
        "pipeline config schema mismatch",
    )
    config_profile_name(raw)
    targets = raw["targets"]
    require(
        isinstance(targets, list)
        and bool(targets)
        and all(isinstance(item, str) and item in TARGETS for item in targets)
        and len(set(targets)) == len(targets),
        "unknown or duplicate output target",
    )
    stages = raw["stages"]
    require(isinstance(stages, dict) and set(stages) == set(specs), "explicit stage map required")
    for name, spec in specs.items():
        entry = stages[name]
        require(
            isinstance(entry, dict)
            and set(entry) == {"enabled", "semantic", "input"}
            and type(entry["enabled"]) is bool
            and isinstance(entry["semantic"], dict),
            "invalid stage configuration",
        )
        semantic = entry["semantic"]
        require(
            set(spec.limit_fields) <= set(semantic) <= {*spec.limit_fields, *spec.optional_fields},
            "unknown or missing semantic feature",
        )
        for key, value in semantic.items():
            if key == "voice":
                require(isinstance(value, str) and value in VOICES, "unsupported voice")
            elif key == "argument_batch_tokens":
                # 0 switches batching off; otherwise the measured prompt size of one batch.
                low, high = BATCH_TOKENS_RANGE
                require(
                    type(value) is int and (value == 0 or low <= value <= high),
                    "invalid stage limit",
                )
            elif key == "max_new_tokens":
                # "auto" is resolved per request from the measured prompt; an int is fixed.
                low, high = model_profile.output_bounds()
                require(
                    value == AUTO_OUTPUT or (type(value) is int and low <= value <= high),
                    "invalid stage limit",
                )
            else:
                low, high = RANGES[key]
                require(type(value) is int and low <= value <= high, "invalid stage limit")
        if entry["input"] is not None:
            _descriptor(entry["input"])
            require(
                entry["input"]["identity"]["stage_name"] == f"books.pipeline.{name}.v1",
                "input descriptor stage mismatch",
            )
    return copy.deepcopy(raw)


def canonical_semantic(semantic: dict[str, Any]) -> dict[str, Any]:
    """Drop an optional feature that is switched off, so absent and zero are one identity."""
    return {k: v for k, v in semantic.items() if not (k in OPTIONAL_OFF_ZERO and v == 0)}


def stage_input_identity(
    name: str,
    spec: StageSpec,
    source: str,
    semantic: dict[str, Any],
    dependencies: tuple[str, ...],
    dependency_lineage: tuple[str, ...] = (),
) -> dict[str, object]:
    contract = StageContract(
        f"books.pipeline.{name}.v1",
        {"schema": spec.schema, "provider": spec.provider, "model": spec.model},
        {"dependency_semantic_lineage": list(dependency_lineage)},
    )
    return stage_identity(
        contract,
        source_hash=source,
        dependency_output_hashes=dependencies,
        model_revision=spec.revision,
        decoding_parameters=canonical_semantic(semantic),
    ).to_dict()


def semantic_lineage(config: dict[str, Any], specs: Mapping[str, StageSpec]) -> dict[str, str]:
    """Bind transitive semantic changes even when regenerated output bytes coincide."""
    lineage: dict[str, str] = {}
    for name in ordered_stages(specs):
        spec = specs[name]
        identity = stage_input_identity(
            name, spec, config["source_sha256"], config["stages"][name]["semantic"], ()
        )
        lineage[name] = content_fingerprint(
            {"local": identity, "upstream": [lineage[dep] for dep in spec.dependencies]}
        )
    return lineage


def _verify(
    root: Path,
    descriptor: dict[str, Any],
    identity: dict[str, object],
    verifier: Verifier | None,
    dependencies: Mapping[str, bytes],
) -> tuple[bytes | None, str]:
    if descriptor["identity"] != identity:
        return None, "INPUT_IDENTITY_MISMATCH"
    if verifier is None:
        return None, "UNVERIFIED_NO_STAGE_VALIDATOR"
    path = root / descriptor["path"]
    try:
        if path.resolve() != path or not path.is_relative_to(root):
            return None, "INPUT_PATH_REDIRECTED"
        raw = _read(path, MAX_ARTIFACT)
        if hashlib.sha256(raw).hexdigest() != descriptor["output_sha256"]:
            return None, "INPUT_BYTES_MISMATCH"
        if verifier(raw, copy.deepcopy(descriptor), dict(dependencies)) is not True:
            return None, "INPUT_CONTRACT_OR_PROVENANCE_REJECTED"
    except (ValueError, OSError, TypeError, KeyError, RuntimeError):
        return None, "INPUT_VERIFICATION_FAILED"
    return raw, "VALIDATED_INPUT"


def resolve(
    config: object,
    artifact_root: Path,
    *,
    specs: Mapping[str, StageSpec] | None = None,
    verifiers: Mapping[str, Verifier] | None = None,
) -> dict[str, object]:
    """Resolve the entire requested closure without executing any stage."""
    graph = registry() if specs is None else specs
    order = ordered_stages(graph)
    doc = validate_config(config, graph)
    checks = {} if verifiers is None else verifiers
    require(set(checks).issubset(graph), "unknown stage validator")
    root = artifact_root.absolute()
    require(root.resolve() == root and root.is_dir(), "private artifact root required")
    targets = doc["targets"]
    requested: set[str] = set()

    def include(name: str) -> None:
        require(name in graph, "target unavailable in registry")
        if name not in requested:
            requested.add(name)
            for dependency in graph[name].dependencies:
                include(dependency)

    for target in targets:
        for name in TARGETS[target]:
            include(name)
    rows: dict[str, dict[str, Any]] = {}
    verified: dict[str, bytes] = {}
    lineage = semantic_lineage(doc, graph)
    for name in order:
        spec = graph[name]
        entry = doc["stages"][name]
        deps = spec.dependencies
        row: dict[str, Any] = {
            "stage": name,
            "enabled": entry["enabled"],
            "requested": name in requested,
            "dependencies": list(deps),
            "support": spec.support,
            "provider": spec.provider,
            "identity": None,
            "output_sha256": None,
            "semantic_lineage_sha256": lineage[name],
            "human_attestation": "REQUIRED_NOT_ATTESTED" if name == "human_review" else None,
        }
        rows[name] = row
        if name not in requested:
            row.update(
                status="DISABLED" if not entry["enabled"] else "NOT_REQUESTED",
                reason="OUTSIDE_TARGET_CLOSURE",
            )
            continue
        if spec.support != "IMPLEMENTED":
            row.update(status="UNSUPPORTED", reason="BOOK_STAGE_NOT_IMPLEMENTED")
            continue
        blocked = [dep for dep in deps if rows[dep]["status"] not in {"RUN", "REUSE"}]
        if blocked:
            row.update(
                status="BLOCKED_DEPENDENCY", reason="UPSTREAM_UNAVAILABLE", blocked_by=blocked
            )
            continue
        # RUN is a future action, not an output. Never invent dependency output hashes.
        pending = [dep for dep in deps if dep not in verified]
        identity = None
        if not pending:
            identity = stage_input_identity(
                name,
                spec,
                doc["source_sha256"],
                entry["semantic"],
                tuple(rows[dep]["output_sha256"] for dep in deps),
                tuple(lineage[dep] for dep in deps),
            )
            row["identity"] = identity
        descriptor = entry["input"]
        if descriptor is not None and identity is not None:
            raw, reason = _verify(
                root,
                descriptor,
                identity,
                checks.get(name),
                {dep: verified[dep] for dep in deps},
            )
            if raw is not None:
                verified[name] = raw
                row.update(status="REUSE", reason=reason, output_sha256=descriptor["output_sha256"])
                if name == "human_review":
                    row["human_attestation"] = "HUMAN_ATTESTED_NOT_INDEPENDENTLY_PROVEN"
                continue
            # An explicit bad input is never silently replaced by paid work.
            row.update(
                status="UNVERIFIED" if entry["enabled"] else "BLOCKED_DEPENDENCY",
                reason=reason,
            )
            continue
        if name == "human_review":
            row.update(
                status="AWAITING_HUMAN_REVIEW" if entry["enabled"] else "BLOCKED_DEPENDENCY",
                reason="HUMAN_ATTESTATION_REQUIRED",
            )
        elif descriptor is not None:
            row.update(
                status="BLOCKED_DEPENDENCY",
                reason="INPUT_DEPENDENCIES_NOT_VERIFIED",
                blocked_by=pending,
            )
        elif not entry["enabled"]:
            row.update(status="BLOCKED_DEPENDENCY", reason="DISABLED_WITHOUT_VALIDATED_INPUT")
        elif spec.provider == "owner-input":
            row.update(status="BLOCKED_DEPENDENCY", reason="EXPLICIT_SOURCE_INPUT_REQUIRED")
        else:
            row.update(status="RUN", reason="PLANNED_ONLY", identity_pending_on=pending)
    terminal = [rows[name]["status"] for target in targets for name in TARGETS[target]]
    return {
        "schema": REPORT_SCHEMA,
        "mode": "DRY_RUN_ONLY",
        "execution_authorized": False,
        "publish_eligible": False,
        "targets": targets,
        "source_sha256": doc["source_sha256"],
        "configuration_sha256": content_fingerprint(doc),
        "status": "PLAN_RESOLVED"
        if all(x in {"RUN", "REUSE"} for x in terminal)
        else "PLAN_BLOCKED",
        "ordered_stages": list(rows.values()),
    }


def template(
    source_sha256: str, target: str = "script", profile: str | None = None
) -> dict[str, Any]:
    """Explicit stage selection for one named target; never grants execution authority."""
    require(is_digest(source_sha256), "invalid source digest")
    require(target in TARGETS, "unknown target")
    if profile is not None:
        require(profile in model_profile.profiles(), "unknown model profile")
    with model_profile.use(profile or model_profile.DEFAULT_PROFILE):
        specs = registry()
    selected: set[str] = set()

    def visit(name: str) -> None:
        if name not in selected:
            selected.add(name)
            for dep in specs[name].dependencies:
                visit(dep)

    for name in TARGETS[target]:
        visit(name)
    defaults: dict[str, object] = {
        # Fixed default, clamped so a small-output profile still yields a valid config.
        "max_new_tokens": min(
            4096,
            model_profile.profiles()[profile or model_profile.DEFAULT_PROFILE].max_output_tokens,
        ),
        "max_attempts": 2,
        "max_runtime_seconds": 900,
        "max_beats": 20,
        "max_scenes": 8,
        "max_words": 1000,
        "max_shots_per_scene": 3,
        "voice": "af_heart",
    }
    document: dict[str, Any] = {
        "schema": SCHEMA,
        "source_sha256": source_sha256,
        "targets": [target],
        "stages": {
            name: {
                "enabled": name in selected,
                "semantic": {field: defaults[field] for field in spec.limit_fields},
                "input": None,
            }
            for name, spec in specs.items()
        },
    }
    if profile is not None:
        document["model_profile"] = profile
    return document


def verify_selected_source(
    raw: bytes, descriptor: dict[str, Any], dependencies: Mapping[str, bytes]
) -> bool:
    """Strict selected bundle import in an isolated temporary directory, no model access."""
    from . import selected_scope

    if dependencies or hashlib.sha256(raw).hexdigest() != descriptor["identity"]["source_hash"]:
        return False
    with tempfile.TemporaryDirectory(prefix="clipper-plan-") as temporary:
        root = Path(temporary)
        folder = root / "book-candidates" / "selected-scopes"
        folder.mkdir(parents=True)
        path = folder / f"{descriptor['output_sha256']}.json"
        path.write_bytes(raw)
        path.chmod(0o600)
        verified = selected_scope.import_chapter(root, path)
        return bool(verified.bundle_sha256 == descriptor["output_sha256"])


def plan_file(config: Path, artifact_root: Path) -> dict[str, object]:
    """Offline CLI entry; only the selected-source validator is currently wired."""
    return resolve(
        strict_json(_read(config, MAX_JSON)),
        artifact_root,
        verifiers={"source": verify_selected_source},
    )
