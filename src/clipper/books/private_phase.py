"""Private configured chapter phase boundary. No SDK import or remote allocation."""

from __future__ import annotations

import hashlib
import os
import stat
import re
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from clipper.stage_contracts import content_fingerprint

from . import (
    acquisition,
    chapter,
    chapter_host,
    evaluation,
    pipeline_plan,
    scripting,
    selected_scope,
)
from .orchestrator import BookOrchestrator

MANIFEST_SCHEMA = "clipper-private-configured-phase-v1"
PACKET_SCHEMA = "clipper-private-chapter-packet-v1"
MAX_PACKET = 32_000_000
MAX_HEADER = 128_000
MAX_FILES = 512
MAX_MEMBER = 2_000_000
WORKSPACE = "mahmoud-bayoumi-mm"
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
NATIVE_PATH = re.compile(
    r"book-candidates/(?:"
    r"failed-phase\\.json|"
    r"selected-scopes/[0-9a-f]{64}\.json|"
    r"section-scopes/[0-9a-f]{64}\.json|"
    r"(?:hybrid_argument|hybrid_audit|hybrid_reconcile|adaptation|chapter_script)/"
    r"[0-9a-f]{64}/(?:validated|started-[12]|raw-[12]|failed-[12]|"
    r"log-[0-9a-f]{32})\.json|"
    r"chapter-evaluations/[0-9a-f]{64}/(?:source-ledger|"
    r"(?:human-review|chapter-result|configured-execution)-[0-9a-f]{64})\.json|"
    r"private-host/chapter-[1-9][0-9]*-review-packet\.json)"
)
INPUT_NAMES = {"config.json", "selected.json", "section.json", "review.json"}


class PrivatePhaseError(ValueError):
    """Fixed-category failure without private text or filenames."""


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise PrivatePhaseError(reason)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def allowed(name: str) -> bool:
    return name in INPUT_NAMES or NATIVE_PATH.fullmatch(name) is not None


def pack(files: dict[str, bytes]) -> bytes:
    """Canonical length-framed archive; no tar extraction or path normalization."""
    require(0 < len(files) <= MAX_FILES, "packet member count")
    rows = []
    chunks = []
    size = 0
    for name, raw in sorted(files.items()):
        require(allowed(name), "packet member outside allowlist")
        require(type(raw) is bytes and 0 < len(raw) <= MAX_MEMBER, "packet member bound")
        size += len(raw)
        require(size <= MAX_PACKET, "packet total bound")
        rows.append({"name": name, "size": len(raw), "sha256": sha(raw)})
        chunks.append(raw)
    header = evaluation._json_bytes({"schema": PACKET_SCHEMA, "members": rows})
    require(len(header) <= MAX_HEADER, "packet header bound")
    result = len(header).to_bytes(4, "big") + header + b"".join(chunks)
    require(len(result) <= MAX_PACKET, "packet total bound")
    return result


def unpack(packet: bytes) -> dict[str, bytes]:
    require(type(packet) is bytes and 4 < len(packet) <= MAX_PACKET, "packet byte bound")
    length = int.from_bytes(packet[:4], "big")
    require(0 < length <= MAX_HEADER and 4 + length < len(packet), "packet header bound")
    header = pipeline_plan.strict_json(packet[4 : 4 + length])
    require(
        set(header) == {"schema", "members"} and header["schema"] == PACKET_SCHEMA,
        "packet schema",
    )
    rows = header["members"]
    require(isinstance(rows, list) and 0 < len(rows) <= MAX_FILES, "packet member count")
    files: dict[str, bytes] = {}
    offset = 4 + length
    for row in rows:
        require(
            isinstance(row, dict) and set(row) == {"name", "size", "sha256"},
            "packet member schema",
        )
        name, size = row["name"], row["size"]
        require(
            isinstance(name, str) and allowed(name) and name not in files,
            "packet member invalid or duplicate",
        )
        require(
            type(size) is int and 0 < size <= MAX_MEMBER and offset + size <= len(packet),
            "packet member bound",
        )
        raw = packet[offset : offset + size]
        require(sha(raw) == row["sha256"], "packet member checksum mismatch")
        files[name] = raw
        offset += size
    require(offset == len(packet) and pack(files) == packet, "packet not canonical")
    return files


def validate_manifest(raw: object) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise PrivatePhaseError("manifest object required")
    require(
        set(raw)
        == {
            "schema",
            "commit",
            "workspace",
            "phase",
            "configuration_sha256",
            "bundle_sha256",
            "section_sha256",
            "review_sha256",
            "expected_brief_identity",
            "checkpoint_sha256",
            "owner",
            "cpu",
            "memory_mib",
            "timeout_seconds",
            "resource_approval",
        },
        "manifest fields",
    )
    require(raw["schema"] == MANIFEST_SCHEMA and raw["workspace"] == WORKSPACE, "manifest scope")
    require(
        isinstance(raw["commit"], str) and HEX40.fullmatch(raw["commit"]) is not None,
        "manifest commit",
    )
    for key in ("configuration_sha256", "bundle_sha256", "section_sha256", "owner"):
        require(pipeline_plan.is_digest(raw[key]), "manifest identity")
    require(
        type(raw["cpu"]) is int
        and raw["cpu"] == 4
        and type(raw["memory_mib"]) is int
        and raw["memory_mib"] == 16384
        and type(raw["timeout_seconds"]) is int
        and 60 <= raw["timeout_seconds"] <= 43200
        and raw["resource_approval"] == "EXPLICIT_PRIVATE_CPU_PHASE",
        "manifest resource bounds or approval",
    )
    require(raw["phase"] in ("brief", "script"), "manifest phase")
    for key in ("review_sha256", "expected_brief_identity", "checkpoint_sha256"):
        require(
            pipeline_plan.is_digest(raw[key]) if raw["phase"] == "script" else raw[key] is None,
            "manifest phase evidence",
        )
    return dict(raw)


def preflight(
    manifest: object, packet: bytes, checkpoint: bytes | None = None
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, bytes]]:
    """Byte-only validation, no model or provider activity."""
    run = validate_manifest(manifest)
    inputs = unpack(packet)
    expected = {"config.json", "selected.json", "section.json"}
    if run["phase"] == "script":
        expected.add("review.json")
    require(set(inputs) == expected, "phase input inventory")
    for name, field in (
        ("selected.json", "bundle_sha256"),
        ("section.json", "section_sha256"),
    ):
        require(sha(inputs[name]) == run[field], "phase input checksum mismatch")
    require(len(inputs["selected.json"]) <= selected_scope.MAX_BUNDLE_BYTES, "source bound")
    require(len(inputs["section.json"]) <= 16_384, "section bound")
    require(len(inputs["config.json"]) <= pipeline_plan.MAX_JSON, "configuration bound")
    config = pipeline_plan.validate_config(
        pipeline_plan.strict_json(inputs["config.json"]), pipeline_plan.registry()
    )
    require(
        content_fingerprint(config) == run["configuration_sha256"]
        and config["source_sha256"] == run["bundle_sha256"]
        and config["targets"] == [run["phase"]],
        "phase configuration binding",
    )
    stages = config["stages"]
    require(stages["source"]["enabled"], "native source validation required")
    require(all(entry["input"] is None for entry in stages.values()), "native input only")
    required = {"argument", "audits", "reconcile"}
    if run["phase"] == "brief":
        require(checkpoint is None, "brief cannot restore checkpoint")
        require(all(stages[name]["enabled"] for name in required), "brief analysis permission")
        restored: dict[str, bytes] = {}
    else:
        require(
            all(not stages[name]["enabled"] for name in required)
            and stages["adaptation"]["enabled"]
            and stages["script"]["enabled"],
            "script phase requires analysis replay and creative permission",
        )
        require(
            sha(inputs["review.json"]) == run["review_sha256"]
            and len(inputs["review.json"]) <= 250000,
            "review binding",
        )
        require(checkpoint is not None, "checkpoint required")
        if checkpoint is None:
            raise PrivatePhaseError("checkpoint required")
        require(sha(checkpoint) == run["checkpoint_sha256"], "checkpoint checksum mismatch")
        restored = unpack(checkpoint)
        require("book-candidates/failed-phase.json" not in restored, "failed phase cannot resume")
        require(all(NATIVE_PATH.fullmatch(name) for name in restored), "checkpoint inventory")
        require(
            not any(
                name.startswith(("book-candidates/adaptation/", "book-candidates/chapter_script/"))
                for name in restored
            ),
            "resume checkpoint must precede creative attempts",
        )
        # The exact prior brief will also be reconstructed by native raw replay.
        require(
            any(
                name.startswith("book-candidates/hybrid_reconcile/")
                and name.endswith("/validated.json")
                and pipeline_plan.strict_json(data).get("artifact_identity")
                == run["expected_brief_identity"]
                for name, data in restored.items()
            ),
            "checkpoint brief identity absent",
        )
        receipts = [
            pipeline_plan.strict_json(data)
            for name, data in restored.items()
            if "/configured-execution-" in name
        ]
        require(
            any(
                receipt.get("status") == "BRIEF_READY_UNREVIEWED"
                and receipt.get("selected_source_bundle_sha256") == run["bundle_sha256"]
                and receipt.get("artifacts", {}).get("brief_identity")
                == run["expected_brief_identity"]
                for receipt in receipts
            ),
            "checkpoint configured brief receipt absent",
        )
    allowed_enabled = required if run["phase"] == "brief" else {"adaptation", "script"}
    require(
        all(
            not entry["enabled"] or name in allowed_enabled | {"source", "human_review"}
            for name, entry in stages.items()
        ),
        "unsupported phase feature",
    )
    return run, inputs, restored


def restore(root: Path, files: dict[str, bytes]) -> None:
    """Restore only native allowlisted members into a new private root."""
    require(
        root.is_dir() and not root.is_symlink() and not any(root.iterdir()), "fresh root required"
    )
    for name, raw in files.items():
        require(NATIVE_PATH.fullmatch(name) is not None, "restore member outside allowlist")
        acquisition._private_write(root / name, raw, private_root=root)


def collect(root: Path) -> bytes:
    files: dict[str, bytes] = {}
    for path in sorted((root / "book-candidates").rglob("*")):
        require(not path.is_symlink(), "checkpoint symlink refused")
        if path.is_dir():
            continue
        name = path.relative_to(root).as_posix()
        require(NATIVE_PATH.fullmatch(name) is not None, "unexpected native checkpoint file")
        files[name] = evaluation._read(path, MAX_MEMBER)
    return pack(files)


def partial_checkpoint(root: Path) -> tuple[bytes | None, int, bool]:
    """Best effort bounded evidence, never a complete/resumable phase result."""
    files: dict[str, bytes] = {}
    omitted = 0
    truncated = False
    total = 0
    try:
        base = root / "book-candidates"
        if base.is_symlink() or not base.is_dir():
            return None, 0, True
        scanned = 0
        for directory, dirs, names in os.walk(base, followlinks=False):
            parent = Path(directory)
            for name in list(dirs):
                if (parent / name).is_symlink():
                    dirs.remove(name)
                    omitted += 1
            for name in names:
                scanned += 1
                if scanned > MAX_FILES * 4:
                    truncated = True
                    break
                path = parent / name
                relative = path.relative_to(root).as_posix()
                try:
                    if (
                        not NATIVE_PATH.fullmatch(relative)
                        or not stat.S_ISREG(path.lstat().st_mode)
                        or path.resolve() != path.absolute()
                        or len(files) >= MAX_FILES
                    ):
                        omitted += 1
                        continue
                    data = evaluation._read(path, MAX_MEMBER)
                    if not data or total + len(data) > MAX_PACKET - MAX_HEADER - 4:
                        omitted += 1
                        continue
                    files[relative] = data
                    total += len(data)
                except Exception:
                    omitted += 1
            if truncated:
                break
        if files:
            return pack(files), omitted, truncated
    except Exception:
        return None, omitted, True
    return None, omitted, truncated


def failed_phase(
    root: Path, run: dict[str, Any], result: dict[str, object]
) -> tuple[dict[str, object], bytes | None]:
    """Sanitized terminal envelope, even when archive finalization itself fails."""
    output, omitted, truncated = partial_checkpoint(root)
    if output is not None:
        try:
            files = unpack(output)
            marker = "book-candidates/failed-phase.json"
            files[marker] = evaluation._json_bytes(
                {"status": "PARTIAL_NONRESUMABLE", "manifest_sha256": content_fingerprint(run)}
            )
            output = pack(files)
        except Exception:
            output = None
            truncated = True
    report: dict[str, object] = {
        "status": "PRIVATE_PHASE_FAILED_NO_RETRY",
        "manifest_sha256": content_fingerprint(run),
        "owner": run["owner"],
        "checkpoint_sha256": sha(output) if output is not None else None,
        "checkpoint_bytes": len(output) if output is not None else 0,
        "retention": "PARTIAL_NONRESUMABLE" if output is not None else "OUTPUT_NOT_RETAINED",
        "omitted_members": omitted,
        "inventory_truncated": truncated,
        "publish_eligible": False,
    }
    for key in ("brief_identity", "configured_execution_receipt_identity", "final_result_identity"):
        if pipeline_plan.is_digest(result.get(key)):
            report[key] = result[key]
    return report, output


def persist_checkpoint(path: Path, packet: bytes, expected_sha256: str) -> None:
    """Owner-side durable write; never claims external backup."""
    require(sha(packet) == expected_sha256, "checkpoint response checksum mismatch")
    unpack(packet)
    acquisition._private_write(path, packet, private_root=path.parent)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    require(
        sha(evaluation._read(path, MAX_PACKET)) == expected_sha256, "checkpoint reread mismatch"
    )


def execute_private_phase(
    manifest: object,
    packet: bytes,
    checkpoint: bytes | None,
    *,
    verified_commit: str,
    verified_workspace: str,
) -> tuple[dict[str, object], bytes | None]:
    """Worker consumer for a future authenticated, reviewed SDK Function launcher.

    The launcher must supply independently established deployment identity and
    durable one-shot claim. This function does not establish those authorities.
    """
    run, inputs, restored = preflight(manifest, packet, checkpoint)
    require(
        verified_commit == run["commit"] and verified_workspace == run["workspace"],
        "worker identity mismatch",
    )
    with tempfile.TemporaryDirectory(prefix="clipper-private-phase-") as temporary:
        root = Path(temporary)
        restore(root, restored)
        selected = root / "book-candidates" / "selected-scopes" / f"{run['bundle_sha256']}.json"
        section = root / "book-candidates" / "section-scopes" / f"{run['section_sha256']}.json"
        for path, raw in ((selected, inputs["selected.json"]), (section, inputs["section.json"])):
            acquisition._private_write(path, raw, private_root=root)
        review = None
        if run["phase"] == "script":
            review = root / "review.json"
            acquisition._private_write(review, inputs["review.json"], private_root=root)
        config = pipeline_plan.strict_json(inputs["config.json"])
        producer = scripting._ORIGINAL_LOCAL_CHAPTER_COMPLETION
        runner = BookOrchestrator(root)
        # No source prose or model output on the provider's standard streams.
        with (
            tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as sink,
            redirect_stdout(sink),
            redirect_stderr(sink),
        ):
            result: dict[str, object] = {}
            try:
                result = chapter.evaluate_hybrid_chapter(
                    runner,
                    selected,
                    section,
                    pipeline_config=config,
                    plan=chapter.ChapterPlan(
                        brief_cpu=run["phase"] == "brief",
                        adaptation_cpu=run["phase"] == "script",
                        script_cpu=run["phase"] == "script",
                    ),
                    review_file=review,
                    expected_brief_identity=run["expected_brief_identity"],
                    argument_completion=producer,
                    audit_completion=producer,
                    reconcile_completion=producer,
                    adaptation_completion=producer,
                    script_completion=producer,
                )
                expected_status = (
                    "BRIEF_READY_UNREVIEWED"
                    if run["phase"] == "brief"
                    else "CHAPTER_SCRIPT_READY_UNREVIEWED"
                )
                require(
                    result["status"] == expected_status
                    and result["automation_claim"] == "ACTUAL_CPU",
                    "actual phase output not verified",
                )
                if run["phase"] == "brief":
                    verified = selected_scope.import_chapter(root, selected)
                    chapter_host._hybrid_review_packet(runner, verified.chapter_number, result)
                output = collect(root)
            except Exception:
                return failed_phase(root, run, result)
        return {
            "status": expected_status,
            "retention": "COMPLETE",
            "manifest_sha256": content_fingerprint(run),
            "owner": run["owner"],
            "checkpoint_sha256": sha(output),
            "checkpoint_bytes": len(output),
            "brief_identity": result["brief_identity"],
            "configured_execution_receipt_identity": result[
                "configured_execution_receipt_identity"
            ],
            "final_result_identity": result.get("final_result_identity"),
            "publish_eligible": False,
        }, output
