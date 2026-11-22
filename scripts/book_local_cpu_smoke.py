"""Dormant public-synthetic-only GitHub CPU Qwen smoke: no private inputs.

Only 'gate' permits the later single-file request commit. 'run' always uses the
production candidate-ref orchestrator and its real pinned local CPU producer.
Normal test/dry-run/collector modes never load model weights.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

LABEL = "PUBLIC_SYNTHETIC_FIXTURE_NOT_A_BOOK"
BRANCH = "review/unified-tts-cleanup"
BOOK_ID = "public-synthetic-fixture-not-a-book"
SCHEMA = "clipper-public-book-local-cpu-smoke-run-request-v1"
INTENT = "REAL_EXECUTION"
RUN_REQUEST = "acceptance/book-local-cpu-smoke/run-request.json"
FIXTURE = "acceptance/book-local-cpu-smoke/public-original-synthetic.txt"
FIXTURE_SHA256 = "eaa39fee22abeeec8df706d64a67d41f6a3c00a91add2c947ac02a1a6cce9d3d"
REPO = Path(__file__).resolve().parents[1]
SHA = re.compile(r"^[0-9a-f]{40}$")
UNIT_LIMIT = 2
MAX_FILE_BYTES = 96_000
MAX_TOTAL_BYTES = 512_000
MAX_FILES = 90
CHUNK_BYTES = 600
MIN_RAM_BYTES = 6 * 1024**3
MIN_DISK_BYTES = 10 * 1024**3
OUTPUT_NAME = "clipper-public-book-cpu-smoke"
RUNTIME = {
    "torch": "2.6.0+cpu",
    "transformers": "4.57.3",
    "huggingface-hub": "0.36.2",
}
_MODEL = "Qwen/Qwen3-1.7B"
_REVISION = "8d4744f9e13072f4920c326350fa81eedb74eae9"


class SmokeError(ValueError):
    """A bounded, public-fixture-only smoke failure."""


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SmokeError("duplicate run-request JSON field")
        result[key] = value
    return result


def _fixture_bytes() -> bytes:
    path = REPO / FIXTURE
    if path.is_symlink() or not path.is_file():
        raise SmokeError("public synthetic fixture missing or linked")
    data = path.read_bytes()
    if len(data) > 4096 or _digest(data) != FIXTURE_SHA256:
        raise SmokeError("fixed public synthetic fixture bytes changed")
    return data


def _paragraphs(data: bytes) -> tuple[str, str]:
    raw = data.decode("utf-8")
    parts = raw.strip().split("\n\n")
    if len(parts) != 3 or parts[0] != LABEL:
        raise SmokeError("public fixture label or two-paragraph layout mismatch")
    paragraphs = (parts[1].strip(), parts[2].strip())
    if any(not 50 <= len(part.split()) <= 220 for part in paragraphs):
        raise SmokeError("public fixture content must be two small substantial units")
    return paragraphs


def validate_request(
    payload: object,
    *,
    ref: str,
    event: str,
    trigger_sha: str,
    before_sha: str,
    parents: tuple[str, ...],
    changes: tuple[str, ...],
    fixture: bytes,
) -> str:
    """Validate a one-commit, one-addition, fixed-fixture request; no other inputs."""
    if ref != f"refs/heads/{BRANCH}" or event != "push":
        raise SmokeError("public smoke requires the fixed review branch and push event")
    if not SHA.fullmatch(trigger_sha) or not SHA.fullmatch(before_sha):
        raise SmokeError("GitHub push SHA is invalid")
    if len(parents) != 1 or parents[0] != before_sha or trigger_sha == before_sha:
        raise SmokeError("one trigger commit directly after the reviewed code is required")
    if changes != (f"A\t{RUN_REQUEST}",):
        raise SmokeError("trigger commit must ONLY add the dedicated run-request file")
    if not isinstance(payload, dict):
        raise SmokeError("run request must be a JSON object")
    required = {
        "schema",
        "intent",
        "branch",
        "reviewed_code_sha",
        "fixture_label",
        "fixture_sha256",
        "max_new_units",
    }
    if set(payload) != required:
        raise SmokeError("run request fields do not match strict schema")
    if (
        payload["schema"] != SCHEMA
        or payload["intent"] != INTENT
        or payload["branch"] != BRANCH
        or payload["reviewed_code_sha"] != before_sha
        or payload["fixture_label"] != LABEL
        or payload["fixture_sha256"] != FIXTURE_SHA256
        or type(payload["max_new_units"]) is not int
        or payload["max_new_units"] != UNIT_LIMIT
    ):
        raise SmokeError("run request does not match exact pinned code/fixture/budget")
    if _digest(fixture) != FIXTURE_SHA256:
        raise SmokeError("public synthetic fixture hash mismatch")
    return before_sha


def _git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=True, timeout=12
    )
    return result.stdout.strip()


def gate() -> dict[str, str]:
    """GitHub push gate: no defaults, alternate paths, commands, or overrides."""
    file = REPO / RUN_REQUEST
    if file.is_symlink() or not file.is_file() or file.stat().st_size > 2048:
        raise SmokeError("dedicated run-request file is missing, linked, or oversized")
    record = json.loads(file.read_bytes(), object_pairs_hook=_unique_pairs)
    trigger = os.environ.get("GITHUB_SHA", "")
    ref = os.environ.get("GITHUB_REF", "")
    before = os.environ.get("SMOKE_PUSH_BEFORE_SHA", "")
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    if _git("rev-parse", "HEAD") != trigger:
        raise SmokeError("checked-out trigger SHA differs from GitHub push SHA")
    parents = tuple(_git("show", "--format=%P", "--no-patch", "HEAD").split())
    changes = tuple(
        _git("diff-tree", "--no-commit-id", "--name-status", "-r", "HEAD").splitlines()
    )
    reviewed_sha = validate_request(
        record,
        ref=ref,
        event=event,
        trigger_sha=trigger,
        before_sha=before,
        parents=parents,
        changes=changes,
        fixture=_fixture_bytes(),
    )
    if _git("ls-tree", "-r", "--name-only", reviewed_sha, RUN_REQUEST):
        raise SmokeError("run-request must not pre-exist in the reviewed code commit")
    return {"trigger_sha": trigger, "reviewed_code_sha": reviewed_sha}


def _root() -> Path:
    runner_temp = os.environ.get("RUNNER_TEMP")
    if not runner_temp:
        raise SmokeError("public smoke requires an ephemeral GitHub runner temp root")
    temp = Path(runner_temp).resolve(strict=True)
    root = temp / OUTPUT_NAME
    if root.is_symlink() or root.parent != temp:
        raise SmokeError("unsafe public smoke output root")
    return root


def _write(path: Path, data: bytes) -> None:
    if path.is_symlink():
        raise SmokeError("public smoke evidence cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    temporary = path.with_name(f".{path.name}.new")
    if temporary.exists() or temporary.is_symlink():
        raise SmokeError("unexpected public smoke evidence temporary file")
    with temporary.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    temporary.chmod(0o600)
    temporary.replace(path)


def _json(path: Path, payload: object) -> None:
    _write(path, (json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n").encode())


def _append(path: Path, payload: object) -> None:
    if path.is_symlink():
        raise SmokeError("public smoke events cannot be a symlink")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("ab") as output:
        output.write((json.dumps(payload, sort_keys=True) + "\n").encode())
        output.flush()
        os.fsync(output.fileno())
    path.chmod(0o600)


def marker(root: Path, phase: str, **fields: object) -> None:
    item = {
        "schema": "clipper-public-smoke-phase-v1",
        "label": LABEL,
        "execution": INTENT,
        "phase": phase,
        "utc": datetime.now(UTC).isoformat(),
        **fields,
    }
    print("PUBLIC_BOOK_CPU_SMOKE_PHASE " + json.dumps(item, sort_keys=True), flush=True)
    _json(root / "evidence" / "latest-phase.json", item)
    _append(root / "evidence" / "phases.jsonl", item)


def _epub(paragraphs: tuple[str, str]) -> bytes:
    """Build one-spine deterministic EPUB without headings or a real book claim."""
    content = "<html><body>" + "".join(
        f"<p>{html.escape(paragraph)}</p>" for paragraph in paragraphs
    ) + "</body></html>"
    package = (
        '<package xmlns:dc="http://purl.org/dc/elements/1.1/" unique-identifier="uid">'
        f"<metadata><dc:title>{LABEL}</dc:title>"
        "<dc:creator>Clipper public synthetic test</dc:creator>"
        '<dc:identifier id="uid">urn:isbn:0000000000</dc:identifier>'
        "</metadata><manifest>"
        '<item id="content" href="source.xhtml" media-type="application/xhtml+xml"/>'
        '</manifest><spine><itemref idref="content"/></spine></package>'
    )
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "META-INF/container.xml",
            '<container><rootfile full-path="OPS/content.opf"/></container>',
        )
        archive.writestr("OPS/content.opf", package)
        archive.writestr("OPS/source.xhtml", content)
    return stream.getvalue()


def prepare(root: Path) -> dict[str, object]:
    """Use only the repository's in-memory recorded transport; no web or models."""
    from clipper.books import acquisition, scripting, whole_book
    from clipper.books.contracts import SourceUnresolved
    from clipper.books.orchestrator import BookOrchestrator
    from clipper.stage_contracts import content_fingerprint

    data = _fixture_bytes()
    epub = _epub(_paragraphs(data))
    asset_url = (
        "https://archive.org/download/clipper-public-synthetic-no-real-record/"
        "public-original-synthetic.epub"
    )
    candidate = acquisition.ReviewedCandidate(
        book_id=BOOK_ID,
        title=LABEL,
        author="Clipper public synthetic test",
        record_url="https://archive.org/metadata/clipper-public-synthetic-no-real-record",
        filename="public-original-synthetic.epub",
        format="epub",
        expected_bytes=len(epub),
        expected_sha256=_digest(epub),
        isbn="0000000000",
        edition_identity="public-synthetic-original-v1",
        title_aliases=(),
        exclusions=(),
    )

    class RecordedPublicFixtureTransport:
        def get(self, url: str, *, limit: int) -> acquisition.HTTPResult:
            if url == candidate.record_url:
                metadata = {
                    "metadata": {"identifier": "clipper-public-synthetic-no-real-record"},
                    "files": [
                        {
                            "name": candidate.filename,
                            "size": str(len(epub)),
                            "source": "original",
                        }
                    ],
                }
                payload = json.dumps(metadata, sort_keys=True).encode()
                if len(payload) > limit:
                    raise SmokeError("synthetic metadata transport exceeded limit")
                return acquisition.HTTPResult(url, url, 200, "application/json", payload)
            if url == asset_url:
                if len(epub) > limit:
                    raise SmokeError("synthetic asset transport exceeded limit")
                return acquisition.HTTPResult(url, url, 200, "application/epub+zip", epub)
            raise SmokeError("unexpected transport URL for public synthetic fixture")

    runner = BookOrchestrator(root / "private")
    ingested = runner.ingest_candidate(
        LABEL, registry=(candidate,), transport=RecordedPublicFixtureTransport()
    )
    if isinstance(ingested, SourceUnresolved):
        raise SmokeError("synthetic fixture unexpectedly failed candidate discovery")
    plan = whole_book.plan_for(ingested.book)
    if (
        len(plan.units) != UNIT_LIMIT
        or plan.skipped
        or any(unit.evidence_field != "native_text" for unit in plan.units)
        or any(len(unit.text.split()) < 50 for unit in plan.units)
        or ingested.book.source.title != LABEL
        or ingested.book.source.authorization_status != "unverified"
        or scripting.SCRIPT_MODEL != _MODEL
        or scripting.SCRIPT_REVISION != _REVISION
    ):
        raise SmokeError("synthetic work plan or pinned model identity changed")
    _write(root / "evidence" / "public-source.txt", data)
    request_rows = []
    for unit in plan.units:
        request = whole_book.request_for(ingested.book, unit)
        if request["model"] != _MODEL or request["revision"] != _REVISION:
            raise SmokeError("mining request changed pinned model identity")
        digest = content_fingerprint(request)
        _json(root / "evidence" / f"unit-{unit.ordinal:04d}-request.json", request)
        request_rows.append(
            {
                "unit": unit.ordinal,
                "request_sha256": digest,
                "text_sha256": unit.text_sha256,
                "origin": unit.origin,
                "field": unit.evidence_field,
                "word_count": len(unit.text.split()),
            }
        )
    manifest = {
        "schema": "clipper-public-book-smoke-fixture-v1",
        "label": LABEL,
        "fixture_sha256": FIXTURE_SHA256,
        "epub_sha256": _digest(epub),
        "book_id": BOOK_ID,
        "source_kind": "FABRICATED_IN_MEMORY_RECORDED_TRANSPORT_NO_NETWORK",
        "source_rights": "unverified",
        "publisher_authenticity": "not_applicable_public_synthetic",
        "analysis_ref": str(ingested.analysis_ref_path.relative_to(root)),
        "analysis_ref_sha256": _digest(ingested.analysis_ref_path.read_bytes()),
        "book_fingerprint": ingested.book.fingerprint,
        "work_plan_fingerprint": plan.fingerprint,
        "work_units": request_rows,
        "max_new_units": UNIT_LIMIT,
        "model": _MODEL,
        "revision": _REVISION,
        "review_status": "MINED_UNREVIEWED",
        "publish_eligible": False,
    }
    _json(root / "evidence" / "fixture-manifest.json", manifest)
    return manifest


def _metric_bytes() -> dict[str, int | None]:
    ram: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            field, _, value = line.partition(":")
            if field in {"MemAvailable", "MemTotal"}:
                ram[field] = int(value.strip().split()[0]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    cgroup_free: int | None = None
    try:
        maximum = Path("/sys/fs/cgroup/memory.max").read_text().strip()
        current = int(Path("/sys/fs/cgroup/memory.current").read_text().strip())
        if maximum != "max":
            cgroup_free = max(0, int(maximum) - current)
    except (OSError, ValueError):
        pass
    disk = shutil.disk_usage(_root())
    rss: int | None = None
    try:
        rss_lines = Path("/proc/self/status").read_text().splitlines()
        entry = next(row for row in rss_lines if row.startswith("VmRSS:"))
        rss = int(entry.split()[1]) * 1024
    except (OSError, ValueError, StopIteration, IndexError):
        pass
    free = ram.get("MemAvailable")
    if cgroup_free is not None:
        free = min(free, cgroup_free) if free is not None else cgroup_free
    return {
        "ram_available_bytes": free,
        "host_ram_total_bytes": ram.get("MemTotal"),
        "cgroup_available_bytes": cgroup_free,
        "disk_free_bytes": disk.free,
        "disk_total_bytes": disk.total,
        "process_rss_bytes": rss,
    }


def _runtime() -> dict[str, object]:
    found: dict[str, str] = {}
    for package, required in RUNTIME.items():
        try:
            actual = version(package)
        except PackageNotFoundError as exc:
            raise SmokeError(f"missing runtime dependency: {package}") from exc
        if actual != required:
            raise SmokeError(f"runtime pin mismatch: {package}")
        found[package] = actual
    import torch

    cuda = bool(torch.cuda.is_available())
    if cuda:
        raise SmokeError("public smoke must not run with a GPU")
    return {"versions": found, "torch_cuda_available": cuda, "device": "cpu"}


def _require_resources(root: Path) -> dict[str, int | None]:
    resources = _metric_bytes()
    _json(root / "evidence" / "resource-preflight.json", resources)
    available = resources["ram_available_bytes"]
    disk = resources["disk_free_bytes"]
    if available is None or available < MIN_RAM_BYTES:
        raise SmokeError("insufficient available CPU RAM for pinned Qwen weights")
    if disk is None or disk < MIN_DISK_BYTES:
        raise SmokeError("insufficient ephemeral disk for pinned Qwen weights")
    return resources


def _observe_memory(root: Path, stop: threading.Event) -> None:
    for index in range(160):
        if stop.is_set():
            break
        try:
            metrics = _metric_bytes()
            _append(
                root / "evidence" / "resource-samples.jsonl",
                {"sample": index, "utc": datetime.now(UTC).isoformat(), **metrics},
            )
        except OSError:
            break
        if stop.wait(12):
            break


def _capture_paths(root: Path) -> dict[str, str]:
    """Exact bounded public fixture outputs, excluding mutable temporary/lock files."""
    result: dict[str, str] = {}
    for path in _allowed_files(root):
        if "/analysis/completions/" in path.as_posix() or (
            "/analysis/progress/" in path.as_posix()
            or "/analysis/exports/" in path.as_posix()
        ):
            result[str(path.relative_to(root))] = _digest(path.read_bytes())
    return result


def _public_result(root: Path, result: dict[str, Any]) -> dict[str, Any]:
    """Make result path identities stable across fresh process replay."""
    normalized = dict(result)
    for key in ("progress_path", "export_path"):
        value = result[key]
        normalized[key] = str(Path(value).relative_to(root)) if value else None
    return normalized


def _execute(root: Path, *, replay: bool) -> None:
    from clipper.books import scripting
    from clipper.books.orchestrator import BookOrchestrator

    gate_info = gate()
    if replay:
        stored = json.loads((root / "evidence" / "outcome.json").read_bytes())
        if stored["result"]["status"] != "complete":
            raise SmokeError("cannot replay a failed or incomplete real model invocation")
        existing = _capture_paths(root)
        if not existing or scripting._script_pipeline.cache_info().misses:
            raise SmokeError("replay was not started in a new unloaded Python process")
        manifest = json.loads((root / "evidence" / "fixture-manifest.json").read_bytes())
        marker(root, "replay_cache_read_start")
    else:
        if root.exists() or root.is_symlink():
            raise SmokeError("fresh public smoke refuses any preseeded results")
        root.mkdir(mode=0o700)
        marker(root, "public_fixture_prepare_start")
        manifest = prepare(root)
        _json(root / "evidence" / "trigger.json", gate_info)
        marker(root, "public_fixture_ready", unit_count=len(manifest["work_units"]))
        _require_resources(root)
        runtime = _runtime()
        _json(root / "evidence" / "runtime.json", runtime)
        marker(root, "real_model_cpu_start", runtime=runtime)
    reference = root / str(manifest["analysis_ref"])
    if reference.is_symlink() or not reference.is_file():
        raise SmokeError("public candidate reference is missing or linked")
    runner = BookOrchestrator(root / "private")
    start = time.monotonic()
    # No injected producer or output. The production loader must execute for
    # first-pass uncached units; a completed replay must never invoke it.
    os.environ["CLIPPER_PUBLIC_BOOK_CPU_SMOKE"] = "1"
    stop = threading.Event()
    watch = threading.Thread(target=_observe_memory, args=(root, stop), daemon=True)
    watch.start()
    try:
        result = runner.mine_candidate(
            reference,
            book_id=BOOK_ID,
            local_cpu=True,
            max_new_units=UNIT_LIMIT,
        )
    finally:
        stop.set()
        watch.join(timeout=2)
    elapsed = round(time.monotonic() - start, 3)
    if result["total_work_units"] != UNIT_LIMIT:
        raise SmokeError("unexpected public smoke unit count after mining")
    if result["review_status"] != "MINED_UNREVIEWED" or result["publish_eligible"] is not False:
        raise SmokeError("public smoke must not approve mined candidates")
    if replay:
        if _public_result(root, result) != stored["result"]:
            raise SmokeError("completed replay changed the production outcome")
        if scripting._script_pipeline.cache_info().misses:
            raise SmokeError("cache replay unexpectedly initialized the model")
        if _capture_paths(root) != existing:
            raise SmokeError("cache replay changed raw/progress/export bytes")
        _json(
            root / "evidence" / "replay.json",
            {
                "label": LABEL,
                "execution": INTENT,
                "status": "CACHE_REPLAY_NO_MODEL_INITIALIZATION",
                "producer_model_initializations": 0,
                "unchanged_private_artifact_hashes": existing,
                "elapsed_seconds": elapsed,
            },
        )
        marker(root, "replay_verified", elapsed_seconds=elapsed)
    else:
        material = {
            "label": LABEL,
            "execution": INTENT,
            "result": _public_result(root, result),
            "elapsed_seconds": elapsed,
            "producer_cache_initializations": scripting._script_pipeline.cache_info().misses,
            "resources_after": _metric_bytes(),
        }
        _json(root / "evidence" / "outcome.json", material)
        marker(
            root,
            "model_mining_finished",
            status=result["status"],
            counts=result["counts"],
            elapsed_seconds=elapsed,
        )
    if result["status"] != "complete":
        raise SmokeError("real pinned model mining incomplete: inspect bounded private progress")


def _allowed_files(root: Path) -> list[Path]:
    """Strict allowlist, no cross-root glob or symlink traversal."""
    allowed = [
        (root / "evidence", re.compile(r"[a-zA-Z0-9-]+(?:\.json|\.jsonl|\.txt)")),
        (
            root / "private" / "book-candidates" / "analysis",
            re.compile(
                rf"(?:completions|failures|progress|exports)/{BOOK_ID}/"
                r"(?:[0-9a-f]{64}/)?[a-zA-Z0-9_.-]+\.(?:txt|json)"
            ),
        ),
        (
            root / "private" / "book-dag",
            re.compile(
                r"books-mining-(?:unit|local-completion)-v1/"
                r"[0-9a-f]{64}/(?:stage|output)\.json"
            ),
        ),
    ]
    found: list[Path] = []
    for directory, allowed_path in allowed:
        if not directory.exists():
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise SmokeError("public evidence allowlist directory is invalid")
        for base, dirs, files in os.walk(directory, followlinks=False):
            base_path = Path(base)
            if any((base_path / child).is_symlink() for child in dirs):
                raise SmokeError("public evidence contains a linked directory")
            for name in files:
                path = base_path / name
                relative = path.relative_to(directory).as_posix()
                if not allowed_path.fullmatch(relative):
                    continue
                if path.is_symlink() or not path.is_file():
                    raise SmokeError("public evidence contains a linked or non-file artifact")
                found.append(path)
                if len(found) > MAX_FILES:
                    raise SmokeError("public evidence exceeds file-count limit")
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


def encoded_evidence(root: Path) -> list[dict[str, object]]:
    """Reconstructable JSON/base64 records; refuse oversized or external data."""
    if root.is_symlink():
        raise SmokeError("public evidence root is a symlink")
    records: list[dict[str, object]] = []
    total = 0
    for file in _allowed_files(root):
        data = file.read_bytes()
        if len(data) > MAX_FILE_BYTES:
            raise SmokeError("public smoke individual evidence file exceeds size limit")
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise SmokeError("public smoke evidence total exceeds hard limit")
        digest = _digest(data)
        chunks = [data[i : i + CHUNK_BYTES] for i in range(0, len(data), CHUNK_BYTES)]
        if not chunks:
            chunks = [b""]
        for index, chunk in enumerate(chunks):
            records.append(
                {
                    "kind": "PUBLIC_BOOK_CPU_SMOKE_EVIDENCE_CHUNK_V1",
                    "label": LABEL,
                    "execution": INTENT,
                    "path": file.relative_to(root).as_posix(),
                    "index": index,
                    "chunk_count": len(chunks),
                    "byte_count": len(data),
                    "sha256": digest,
                    "data_b64": base64.b64encode(chunk).decode("ascii"),
                }
            )
    return records


def collect(root: Path) -> None:
    records = encoded_evidence(root)
    names = sorted({str(item["path"]) for item in records})
    summary = {
        "label": LABEL,
        "execution": INTENT,
        "schema": "clipper-public-book-cpu-smoke-collection-v1",
        "record_count": len(records),
        "file_count": len(names),
        "files": names,
        "no_uploaded_artifacts": True,
        "model_execution_is_not_accepted_by_synthetic_ci": True,
    }
    print("PUBLIC_BOOK_CPU_SMOKE_COLLECTION " + json.dumps(summary, sort_keys=True), flush=True)
    for record in records:
        print(json.dumps(record, sort_keys=True, separators=(",", ":")), flush=True)
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with Path(summary_path).open("a", encoding="utf-8") as out:
            out.write(
                "## PUBLIC_SYNTHETIC_FIXTURE_NOT_A_BOOK — CPU smoke\n"
                f"Execution: {INTENT}; evidence files: {len(names)}; "
                f"reconstruction records: {len(records)}\n"
                "No artifact upload or persistent cache. "
                "The exact evidence is in job log JSON/base64 chunks.\n"
            )


def finish(root: Path, exit_code: int) -> None:
    if not root.exists():
        return
    kind = "timeout" if exit_code in {124, 137} else ("success" if exit_code == 0 else "failure")
    _json(
        root / "evidence" / "process-exit.json",
        {
            "label": LABEL,
            "execution": INTENT,
            "exit_code": exit_code,
            "outcome": kind,
            "utc": datetime.now(UTC).isoformat(),
        },
    )
    print(f"PUBLIC_BOOK_CPU_SMOKE_PROCESS_EXIT {exit_code} {kind}", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dormant public synthetic Qwen CPU smoke")
    parser.add_argument("action", choices=("gate", "dry-run", "run", "replay", "finish", "collect"))
    parser.add_argument("--exit-code", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        if args.action == "gate":
            print("PUBLIC_BOOK_CPU_SMOKE_GATE " + json.dumps(gate(), sort_keys=True))
        elif args.action == "dry-run":
            # CI preparation only. Never initializes model, Hub, torch, or inference.
            from tempfile import TemporaryDirectory

            with TemporaryDirectory(prefix="clipper-public-fixture-") as tmp:
                root = Path(tmp)
                manifest = prepare(root)
                print(
                    "PUBLIC_BOOK_CPU_SMOKE_DRY_RUN "
                    + json.dumps(
                        {
                            "label": LABEL,
                            "unit_count": len(manifest["work_units"]),
                            "no_inference": True,
                        }
                    )
                )
        elif args.action == "run":
            _execute(_root(), replay=False)
        elif args.action == "replay":
            _execute(_root(), replay=True)
        elif args.action == "finish":
            if args.exit_code is None or not 0 <= args.exit_code <= 255:
                raise SmokeError("finish requires a valid process exit code")
            finish(_root(), args.exit_code)
        else:
            collect(_root())
    except Exception as exc:
        # Public-only diagnostics; do not leak source prompts or environment.
        print(
            "PUBLIC_BOOK_CPU_SMOKE_ERROR "
            + json.dumps(
                {"label": LABEL, "execution": INTENT, "error_type": type(exc).__name__},
                sort_keys=True,
            ),
            flush=True,
        )
        if args.action in {"run", "replay"}:
            root = _root()
            if root.exists():
                _json(
                    root / "evidence" / "failure.json",
                    {
                        "label": LABEL,
                        "execution": INTENT,
                        "failure_type": type(exc).__name__,
                        "phase": (
                            json.loads((root / "evidence" / "latest-phase.json").read_bytes())[
                                "phase"
                            ]
                            if (root / "evidence" / "latest-phase.json").is_file()
                            else "before_initial_phase"
                        ),
                    },
                )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
