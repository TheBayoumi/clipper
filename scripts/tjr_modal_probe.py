#!/usr/bin/env python3
"""Cookie-free original YouTube egress probe on the user's existing Modal account.

GitHub Actions controls the task. Modal supplies a different network path,
not a different video source. No original bytes or browser cookies are uploaded.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import modal

from scripts.tjr_media_policy import has_production_hd_video_stream, production_hd_format

app = modal.App("clipper-tjr-official-youtube-probe")
MAX_MODAL_EGRESS_ATTEMPTS = 1
# Reuse the original Clipper Modal media image/provider that successfully
# acquired YouTube masters in August, not bare yt-dlp in Debian/Deno.
image = (
    modal.Image.from_registry("node:22-bookworm-slim", add_python="3.12")
    .entrypoint([])
    .apt_install("ffmpeg", "git", "ca-certificates")
    .uv_pip_install(
        "yt-dlp[default]>=2026.7.4,<2027",
        "bgutil-ytdlp-pot-provider==1.3.1",
    )
    .run_commands(
        "git clone --depth 1 --branch 1.3.1 "
        "https://github.com/Brainicism/bgutil-ytdlp-pot-provider.git "
        "/root/bgutil-ytdlp-pot-provider",
        "cd /root/bgutil-ytdlp-pot-provider/server && npm ci && npx tsc",
    )
)

PROVIDER_HOME = "/root/bgutil-ytdlp-pot-provider/server"
PROVIDER_ARG = f"youtubepot-bgutilscript:server_home={PROVIDER_HOME}"
# Independent provider-backed and plain-client transports. A bad/expired
# PO token must not suppress a working default client on the same route.
ACQUISITION_STRATEGIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "bgutil_default_mweb",
        (
            "--extractor-args",
            "youtube:player_client=default,mweb",
            "--extractor-args",
            PROVIDER_ARG,
        ),
    ),
    # This is deliberately the second strategy: if the provider-backed mweb
    # attempt hits LOGIN_REQUIRED, test an independent, non-provider client
    # before concluding the video's regional/IP access is blocked.
    (
        "plain_tv_web_safari",
        ("--extractor-args", "youtube:player_client=tv,web_safari"),
    ),
    ("bgutil_default_clients", ("--extractor-args", PROVIDER_ARG)),
    ("plain_default", ()),
    (
        "bgutil_embedded_android_vr",
        (
            "--extractor-args",
            "youtube:player_client=web_embedded,android_vr",
            "--extractor-args",
            PROVIDER_ARG,
        ),
    ),
)


def source_download_sections(duration_seconds: float) -> list[str]:
    """Isolated Modal worker uses the same full-source limit as the runner.

    This must remain dependency-free: the remote image does not install the
    editor's Python package or its XML dependencies.
    """
    if duration_seconds <= 0:
        raise ValueError("verified YouTube duration must be positive")
    if duration_seconds > 3600:
        raise ValueError("SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT: source exceeds one hour")
    return []


def acquisition_runtime() -> dict[str, str]:
    """Record resolved worker packages; declarations alone are not provenance."""
    versions = {"python": sys.version.split()[0]}
    for package in ("yt-dlp", "yt-dlp-ejs", "bgutil-ytdlp-pot-provider"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return versions


def safe_extractor_trace(stderr: str) -> list[str]:
    """Retain client/provider errors without URLs, cookies or raw PO tokens."""
    prefixes = (
        "[debug] [youtube",
        "[youtube]",
        "[youtube:pot",
        "WARNING: [youtube",
        "ERROR:",
        "[debug] JS runtimes:",
        "[debug] yt-dlp version",
        "[debug] Plugin directories:",
    )
    trace = []
    for line in stderr.splitlines():
        if not line.startswith(prefixes):
            continue
        line = re.sub(r"https?://[^\s]+", "<url>", line)
        line = re.sub(
            r"(?i)(po_token|pot|visitor_data|authorization|cookie)(\s*[:=]\s*)[^\s,]+",
            r"\1\2<redacted>",
            line,
        )
        line = re.sub(r"[A-Za-z0-9_+/=-]{40,}", "<opaque-value>", line)
        trace.append(line[:800])
    return trace[-60:]


def worker_network_diagnostics(video_id: str) -> dict[str, Any]:
    """Independent bounded checks after acquisition; never reuse their sessions."""
    import ipaddress
    import subprocess
    import urllib.request

    evidence: dict[str, Any] = {}
    try:
        node = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=5, check=False
        )
        evidence["node_version"] = node.stdout.strip() if node.returncode == 0 else "UNAVAILABLE"
    except (OSError, subprocess.TimeoutExpired):
        evidence["node_version"] = "UNAVAILABLE"
    try:
        with urllib.request.urlopen("https://api.ipify.org?format=json", timeout=8) as response:
            address = json.loads(response.read(2048))["ip"]
        evidence["outbound_ip"] = str(ipaddress.ip_address(address))
    except Exception as exc:
        evidence["outbound_ip_check_error"] = type(exc).__name__
    try:
        request = urllib.request.Request(
            f"https://www.youtube.com/watch?v={video_id}", headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310 - fixed HTTPS host
            evidence["watch_http_status"] = response.status
            html = response.read(3_000_000).decode("utf-8", errors="replace")
        match = re.search(r"(?:var\s+)?ytInitialPlayerResponse\s*=\s*", html)
        if match:
            player, _ = json.JSONDecoder().raw_decode(html[match.end() :])
            details = player.get("videoDetails") or {}
            playback = player.get("playabilityStatus") or {}
            evidence["watch_video_id"] = details.get("videoId")
            evidence["watch_playability"] = playback.get("status")
            evidence["watch_reason"] = str(playback.get("reason") or "")[:300]
        else:
            evidence["watch_playability"] = "NO_PLAYER_RESPONSE"
    except Exception as exc:
        evidence["watch_check_error"] = type(exc).__name__
    return evidence


def _yt_command(args: tuple[str, ...], url: str) -> list[str]:
    return [
        "yt-dlp",
        *(["--verbose"] if os.getenv("TJR_ACQUISITION_DIAGNOSTICS") == "1" else []),
        "--ignore-config",
        "--no-warnings",
        "--js-runtimes",
        "node",
        "--no-playlist",
        "--socket-timeout",
        "20",
        "--retries",
        "4",
        "--fragment-retries",
        "4",
        *args,
        url,
    ]


def _transport_error(output: str) -> str:
    lower = output.lower()
    if "sign in to confirm" in lower or "not a bot" in lower:
        return "YOUTUBE_IP_OR_LOGIN_CHALLENGE"
    if "http error 403" in lower or "403 forbidden" in lower:
        return "YOUTUBE_MEDIA_HTTP_403"
    if "private video" in lower:
        return "VIDEO_NOT_PUBLIC"
    if "not available" in lower:
        return "VIDEO_UNAVAILABLE"
    return "UNCLASSIFIED_YOUTUBE_TRANSPORT_FAILURE"


volume = modal.Volume.from_name("clipper-tjr-source-transport", create_if_missing=True)

# Private qualification worker in the existing app/volume; not an HTTP endpoint.
_project_root = Path(__file__).resolve().parents[1]
reviewer_gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.12")
    .apt_install("build-essential", "cmake", "ca-certificates")
    .env(
        {
            "CMAKE_ARGS": "-DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_CUDA_ARCHITECTURES=89",
            "CMAKE_BUILD_PARALLEL_LEVEL": "2",
            "CC": "gcc",
            "CXX": "g++",
            "CUDAHOSTCXX": "g++",
            "PYTHONPATH": "/app/src:/app",
            "TJR_REVIEW_MODEL_CACHE": "/tjr-media/reviewer/weights",
        }
    )
    .run_commands("gcc --version", "g++ --version", "nvcc --version")
    .pip_install(
        "llama-cpp-python==0.3.35",
        "numpy==2.3.5",
        "PyYAML==6.0.3",
        "Pillow==11.3.0",
        "gdown==6.4.1",
        "defusedxml==0.7.1",
    )
    .add_local_dir(str(_project_root / "src"), remote_path="/app/src")
    .add_local_dir(
        str(_project_root / "scripts"), remote_path="/app/scripts", ignore=["__pycache__/"]
    )
)


@app.function(
    image=reviewer_gpu_image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    # Modal hard maximum: no shorter application-level GPU assessment deadline.
    timeout=86400,
    max_containers=1,
    retries=0,
    volumes={"/tjr-media": volume},
)
def qualify_source_reviewer_gpu(
    packed: bytes, profile: dict[str, Any], job_key: str, code_hash: str
) -> dict[str, Any]:
    """One candidate, exact production review code, durable raw proof and warm replay."""
    import gzip
    import hashlib
    import subprocess
    import threading
    import time

    from llama_cpp import llama_supports_gpu_offload

    from clipper.editorial_code_identity import qualification_code_hash
    from scripts.tjr_semantic_editor import (
        _gpu_review_profiles,
        _thinking_review_profile,
        reviewer_evidence_qualification,
    )

    actual_code_hash = qualification_code_hash(Path("/app"))
    expected_key = hashlib.sha256(
        packed + json.dumps(profile, sort_keys=True).encode() + code_hash.encode()
    ).hexdigest()
    if (
        actual_code_hash != code_hash
        or job_key != expected_key
        or profile not in [*_gpu_review_profiles(), _thinking_review_profile()]
    ):
        raise ValueError("GPU qualification code/profile/input identity mismatch")
    if not llama_supports_gpu_offload() or profile.get("gpu_layers") != -1:
        raise RuntimeError("GPU backend unavailable; CPU fallback is prohibited")
    root = Path("/tjr-media/reviewer/qualification") / job_key
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text())
        if (
            saved.get("qualification_complete") is True
            and all(
                (root / name).is_file()
                and hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                for name, digest in saved.get("file_hashes", {}).items()
            )
            and saved.get("file_hashes")
        ):
            return {**saved, "qualification_cache_hit": True}
    inputs = json.loads(gzip.decompress(packed))
    if inputs.get("worker_code_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("GPU worker code differs from the submitted qualification")
    if (
        inputs["provenance"].get("identity", {}).get("source_sha256")
        != "2a7e07b37074f3073d71b65e10a3efb4019b3cdd4277bc2d3770a99dcbc55e0a"
    ):
        raise ValueError("qualification source differs from pinned original")
    for name, value in (
        ("baseline.json", inputs["baseline"]),
        ("transcript.json", inputs["transcript"]),
        ("editorial-cache.json", inputs["provenance"]),
        ("heldout.json", inputs["heldout"]),
    ):
        (root / name).write_text(json.dumps(value))
    stop = threading.Event()
    vram = {"peak_mib": 0}

    def sample_vram() -> None:
        while not stop.is_set():
            try:
                values = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    timeout=5,
                )
                vram["peak_mib"] = max(
                    vram["peak_mib"], *(int(value) for value in values.splitlines())
                )
            except (ValueError, OSError, subprocess.SubprocessError):
                pass
            stop.wait(1)

    monitor = threading.Thread(target=sample_vram, daemon=True)
    monitor.start()
    began = time.monotonic()
    manifest: dict[str, Any] = {
        "profile": profile,
        "job_key": job_key,
        "passed": False,
        "qualification_cache_hit": False,
        "runtime": {"gpu_layers": -1, "gpu_offload_supported": True},
    }
    try:
        manifest["runtime"]["gpu"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
        cold = root / "cold"
        cold.mkdir(exist_ok=True)
        cold_output = cold / "proof.json"
        reviewer_evidence_qualification(
            root / "baseline.json",
            root / "transcript.json",
            cold_output,
            model_profile=profile,
            qualification_root=Path("/app"),
            claim_level_probe=True,
            heldout_path=root / "heldout.json",
            source_qa_probe=profile["candidate"] != "reasoning_30b_a3b",
            checkpoint_callback=volume.commit,
        )
        proof = json.loads(cold_output.read_text())
        manifest.update(
            {
                key: proof[key]
                for key in (
                    "contract_error_count",
                    "semantic_error_count",
                    "request_cache_metrics",
                    "seconds",
                    "semantic_pass",
                    "claim_level_pass",
                    "claim_level_contract_error_count",
                    "claim_level_semantic_error_count",
                    "heldout_scores",
                    "source_qa_pass",
                    "source_qa_contract_error_count",
                    "source_qa_semantic_error_count",
                )
                if key in proof
            }
        )
        manifest["warm_replay"] = {
            "performed": False,
            "reason": "contract errors must be resolved before reuse qualification",
        }
        if proof["contract_error_count"] == 0:
            replay = root / "replay"
            replay.mkdir(exist_ok=True)
            reviewer_evidence_qualification(
                cold_output,
                root / "transcript.json",
                replay / "proof.json",
                model_profile=profile,
                qualification_root=Path("/app"),
                claim_level_probe=True,
                heldout_path=root / "heldout.json",
                source_qa_probe=profile["candidate"] != "reasoning_30b_a3b",
                checkpoint_callback=volume.commit,
            )
            warm = json.loads((replay / "proof.json").read_text())
            stable = all(
                [
                    (row.get("actual_accept"), row.get("actual_supported"), row["passed"])
                    for row in proof[key]
                ]
                == [
                    (row.get("actual_accept"), row.get("actual_supported"), row["passed"])
                    for row in warm[key]
                ]
                for key in ("cases", "comparisons", "claim_comparisons", "source_qa_comparisons")
            )
            stable = stable and all(
                [row[name].get("actual_supported") for row in proof["heldout_comparisons"]]
                == [row[name].get("actual_supported") for row in warm["heldout_comparisons"]]
                for name in (
                    "existing",
                    "experimental_claim_level",
                    *(("source_first_qa",) if profile["candidate"] != "reasoning_30b_a3b" else ()),
                )
            )
            manifest["warm_replay"] = {
                "performed": True,
                "same_verdicts": stable,
                **warm["request_cache_metrics"],
            }
            manifest["passed"] = (
                proof["semantic_pass"]
                and proof["claim_level_pass"]
                and all(
                    row["experimental_claim_level"]["passed"]
                    for row in proof["heldout_comparisons"]
                )
                and (
                    profile["candidate"] == "reasoning_30b_a3b"
                    or (
                        proof["source_qa_pass"]
                        and all(
                            row["source_first_qa"]["passed"] for row in proof["heldout_comparisons"]
                        )
                    )
                )
                and stable
                and warm["request_cache_metrics"]["model_calls"] == 0
            )
            manifest["qualification_complete"] = (
                proof.get("experiment_complete") is True and warm.get("experiment_complete") is True
            )
        else:
            # A contract-invalid but completed diagnostic is still a complete
            # negative result; incomplete or errored runs are always resumable.
            manifest["qualification_complete"] = proof.get("experiment_complete") is True
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
    finally:
        stop.set()
        monitor.join(timeout=6)
        manifest["passed"] = manifest["passed"] and vram["peak_mib"] >= 512
        manifest["runtime"].update(
            peak_sampled_vram_mib=vram["peak_mib"], wall_seconds=round(time.monotonic() - began, 3)
        )
        manifest["files"] = [str(path.relative_to(root)) for path in root.glob("*/*.json")]
        manifest["file_hashes"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in manifest["files"]
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        volume.commit()
    return manifest


@app.function(
    image=reviewer_gpu_image,
    gpu="L40S",
    cpu=4,
    memory=32768,
    timeout=86400,
    max_containers=1,
    retries=0,
    volumes={"/tjr-media": volume},
)
def qualify_podcast_narrative_gpu(
    packed: bytes, profile: dict[str, Any], job_key: str, code_hash: str
) -> dict[str, Any]:
    """Diagnose source-blind story scope and event entailment on a real GPU.

    This is a frozen single-source benchmark, never a general podcast approval.
    Completed inference requests are committed individually for exact reruns.
    """
    import gzip
    import hashlib
    import importlib.metadata
    import time

    from llama_cpp import llama_supports_gpu_offload

    from clipper.editorial_answer_comparison_probe import run_answer_comparison_probe
    from clipper.editorial_benchmark import load_frozen_relations
    from clipper.editorial_code_identity import qualification_code_hash
    from clipper.editorial_source_scope_probe import run_source_scope_probe
    from scripts.tjr_semantic_editor import (
        LocalReasoningReviewer,
        ReviewRequestCache,
        _thinking_review_profile,
    )

    key_expected = hashlib.sha256(
        packed + json.dumps(profile, sort_keys=True).encode() + code_hash.encode()
    ).hexdigest()
    if (
        job_key != key_expected
        or qualification_code_hash(Path("/app")) != code_hash
        or profile != _thinking_review_profile()
        or not llama_supports_gpu_offload()
        or profile.get("gpu_layers") != -1
    ):
        raise ValueError("narrative GPU source, model or runtime identity mismatch")
    payload = json.loads(gzip.decompress(packed))
    if (
        set(payload)
        != {
            "worker_code_sha256",
            "fixture",
            "proof",
            "transcript",
            "provenance",
            "source_answer",
            "scope_gold",
        }
        or payload["worker_code_sha256"] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ):
        raise ValueError("narrative GPU worker input contract differs from approved code")
    root = Path("/tjr-media/reviewer/narrative") / job_key
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        saved = json.loads(manifest_path.read_text())
        if (
            saved.get("experiment_complete") is True
            and all(
                (root / name).is_file()
                and hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                for name, digest in saved.get("file_hashes", {}).items()
            )
            and saved.get("file_hashes")
        ):
            return {**saved, "qualification_cache_hit": True}
    paths = {}
    for name in ("fixture", "proof", "transcript", "provenance", "source_answer", "scope_gold"):
        if not isinstance(payload[name], str):
            raise ValueError("narrative GPU input must preserve exact UTF-8 source bytes")
        paths[name] = root / f"{name}.json"
        paths[name].write_text(payload[name], encoding="utf-8")
    gold = json.loads(paths["fixture"].read_text())
    relations = load_frozen_relations(
        paths["fixture"], paths["proof"], paths["transcript"], paths["provenance"]
    )
    source_answer = json.loads(paths["source_answer"].read_text())
    if (
        len(relations) != 15
        or gold.get("source_sha256")
        != "2a7e07b37074f3073d71b65e10a3efb4019b3cdd4277bc2d3770a99dcbc55e0a"
        or source_answer.get("experiment") != "claim_blind_source_answer_v1"
        or source_answer.get("experiment_complete") is not True
        or source_answer.get("production_approved") is not False
    ):
        raise ValueError("narrative GPU requires complete pinned source-only evidence")
    began = time.monotonic()
    manifest: dict[str, Any] = {
        "experiment": "podcast_narrative_gpu_v1",
        "experiment_complete": False,
        "diagnostic_only": True,
        "production_approved": False,
        "qualification_cache_hit": False,
        "source_sha256": gold["source_sha256"],
        "model_profile": profile,
        "job_key": job_key,
    }
    reviewer = None
    runtime = importlib.metadata.version("llama-cpp-python")

    def factory():
        nonlocal reviewer
        if reviewer is None:
            reviewer = LocalReasoningReviewer(profile)
        return reviewer

    def model_cache(stage: str, *, replay: bool):
        return ReviewRequestCache(
            root / stage / "review-request-cache.json",
            (lambda: (_ for _ in ()).throw(RuntimeError("replay attempted new model inference")))
            if replay
            else factory,
            {
                "experiment": f"podcast_narrative_{stage}_v1",
                "fixture_sha256": hashlib.sha256(paths["fixture"].read_bytes()).hexdigest(),
                "baseline_proof_sha256": hashlib.sha256(paths["proof"].read_bytes()).hexdigest(),
                "source_answer_sha256": hashlib.sha256(
                    paths["source_answer"].read_bytes()
                ).hexdigest(),
                "model_profile": profile,
            },
            replay_only=replay,
            recorded_runtime=runtime if replay else None,
            reasoning=True,
            on_response_persisted=None if replay else volume.commit,
        )

    try:
        scores = {}
        for replay in (False, True):
            prefix = "warm" if replay else "cold"
            scope_dir = root / "scope"
            compare_dir = root / "comparison"
            scope_dir.mkdir(exist_ok=True)
            compare_dir.mkdir(exist_ok=True)
            scope_cache = model_cache("scope", replay=replay)
            comparison_cache = model_cache("comparison", replay=replay)
            scope_out = scope_dir / f"{prefix}-proof.json"
            compare_out = compare_dir / f"{prefix}-proof.json"
            # The underlying diagnostics return 1 deliberately because the
            # evaluator cannot grant automated production approval.
            run_source_scope_probe(
                paths["fixture"],
                paths["proof"],
                paths["transcript"],
                paths["provenance"],
                paths["source_answer"],
                scope_out,
                completion=scope_cache._review_completion,
                scope_model_profile=profile,
                request_metrics=lambda cache=scope_cache: cache.metrics,
                scope_gold_path=paths["scope_gold"],
            )
            volume.commit()
            run_answer_comparison_probe(
                paths["fixture"],
                paths["proof"],
                paths["transcript"],
                paths["provenance"],
                scope_out,
                compare_out,
                completion=comparison_cache._review_completion,
                comparison_model_profile=profile,
                request_metrics=lambda cache=comparison_cache: cache.metrics,
                scope_gold_path=paths["scope_gold"],
            )
            volume.commit()
            scope_report = json.loads(scope_out.read_text())
            compare_report = json.loads(compare_out.read_text())
            if (
                scope_report.get("experiment_complete") is not True
                or compare_report.get("experiment_complete") is not True
            ):
                raise RuntimeError("narrative GPU report incomplete")
            scores[prefix] = {
                "scope": scope_report.get("scope_benchmark"),
                "comparison": compare_report.get("benchmark"),
                "scope_cache": scope_cache.metrics,
                "comparison_cache": comparison_cache.metrics,
                "scope_cases": [row.get("scope_review") for row in scope_report["cases"]],
                "comparison_cases": [row.get("verdict") for row in compare_report["cases"]],
            }
        same = (
            scores["cold"]["scope_cases"] == scores["warm"]["scope_cases"]
            and scores["cold"]["comparison_cases"] == scores["warm"]["comparison_cases"]
            and scores["warm"]["scope_cache"]["model_calls"] == 0
            and scores["warm"]["comparison_cache"]["model_calls"] == 0
        )
        manifest.update(
            experiment_complete=True,
            semantic_pass=(
                same
                and scores["cold"]["scope"]["exact_pair_matches"] == 15
                and scores["cold"]["comparison"]["exact_binary_matches"] == 15
                and not scores["cold"]["scope"]["false_responsive_case_ids"]
            ),
            warm_replay_same_verdicts=same,
            scoring={
                k: {j: v for j, v in row.items() if j not in {"scope_cases", "comparison_cases"}}
                for k, row in scores.items()
            },
        )
    except Exception as error:
        manifest["error"] = f"{type(error).__name__}: {error}"
    finally:
        if reviewer is not None:
            reviewer.close()
        manifest["wall_seconds"] = round(time.monotonic() - began, 3)
        manifest["files"] = [
            str(path.relative_to(root)) for path in root.rglob("*.json") if path != manifest_path
        ]
        manifest["file_hashes"] = {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in manifest["files"]
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))
        volume.commit()
    return manifest


@app.function(image=image, volumes={"/tjr-media": volume}, timeout=1600, cpu=2, memory=2048)
def inspect_original_youtube(candidates: list[dict[str, str]], run_key: str = "") -> dict[str, Any]:
    result = _inspect_original_youtube(candidates, run_key)
    result["worker_runtime"] = acquisition_runtime()
    if os.getenv("TJR_ACQUISITION_DIAGNOSTICS") == "1" and candidates:
        result["network_diagnostics"] = worker_network_diagnostics(candidates[0]["video_id"])
    return result


def _inspect_original_youtube(candidates: list[dict[str, str]], run_key: str) -> dict[str, Any]:
    """Verify source identity and transfer real HD bytes on one Modal egress.

    Reuses Clipper's successful BgUtils strategy family. Metadata-only
    successes are insufficient: an actual signed HD CDN URL must serve bytes.
    """
    import subprocess
    import tempfile

    approved = {
        "UCf1q6dhccWr6eQEcFFnJSbA",
    }
    attempts: list[dict[str, str]] = []
    total_ip_challenges = 0
    blocked_video_count = 0
    for candidate in candidates[:6]:
        ip_challenges = 0
        video_id = candidate["video_id"]
        channel_id = candidate["channel_id"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        if channel_id not in approved or len(video_id) != 11:
            attempts.append({"url": url, "reason": "SOURCE_NOT_ALLOWLISTED"})
            continue
        for strategy_name, strategy_args in ACQUISITION_STRATEGIES:
            try:
                metadata_run = subprocess.run(
                    [
                        *_yt_command(strategy_args, url)[:-1],
                        "--dump-single-json",
                        "--skip-download",
                        url,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=100,
                    check=False,
                )
                if metadata_run.returncode:
                    reason = _transport_error(metadata_run.stderr)
                    attempts.append(
                        {
                            "url": url,
                            "strategy": strategy_name,
                            "stage": "metadata",
                            "reason": reason,
                            "extractor_trace": safe_extractor_trace(metadata_run.stderr),
                        }
                    )
                    if reason == "YOUTUBE_IP_OR_LOGIN_CHALLENGE":
                        ip_challenges += 1
                        total_ip_challenges += 1
                    continue
                metadata = json.loads(metadata_run.stdout)
                if not isinstance(metadata, dict):
                    raise RuntimeError("YouTube metadata is not an object")
                if metadata.get("id") != video_id or metadata.get("channel_id") != channel_id:
                    attempts.append(
                        {"url": url, "strategy": strategy_name, "reason": "VIDEO_OWNER_MISMATCH"}
                    )
                    break
                live_status = str(metadata.get("live_status") or "")
                if (
                    metadata.get("is_live")
                    or metadata.get("is_upcoming")
                    or live_status in {"is_live", "is_upcoming"}
                ):
                    attempts.append(
                        {
                            "url": url,
                            "strategy": strategy_name,
                            "reason": "LIVE_OR_UPCOMING_STREAM",
                        }
                    )
                    break
                duration = float(metadata.get("duration") or 0)
                if duration < 90 or duration > 3600:
                    attempts.append(
                        {
                            "url": url,
                            "strategy": strategy_name,
                            "reason": "SHORT_VIDEO"
                            if duration < 90
                            else "SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT",
                        }
                    )
                    break
                formats = metadata.get("formats") or []
                hd = any(isinstance(item, dict) and production_hd_format(item) for item in formats)
                if not hd:
                    attempts.append(
                        {"url": url, "strategy": strategy_name, "reason": "NO_HD_FORMAT"}
                    )
                    continue
                # Verify actual bytes, not just the extractor's advertised formats.
                with tempfile.TemporaryDirectory(prefix="tjr-modal-real-youtube-") as temp:
                    media_command = [
                        *_yt_command(strategy_args, url)[:-1],
                        "--test",
                        "--no-part",
                        "-f",
                        "bv*[width>=1280][height>=720]/bv*[width>=720][height>=1280]/b[width>=1280][height>=720]/b[width>=720][height>=1280]",
                        "-o",
                        str(Path(temp) / "source.%(ext)s"),
                        url,
                    ]
                    transfer = subprocess.run(
                        media_command, capture_output=True, text=True, timeout=145, check=False
                    )
                    media_bytes = sum(
                        file.stat().st_size
                        for file in Path(temp).glob("source.*")
                        if file.is_file()
                    )
                if transfer.returncode or media_bytes < 1024:
                    attempts.append(
                        {
                            "url": url,
                            "strategy": strategy_name,
                            "stage": "actual_hd_media_transfer",
                            "reason": _transport_error(transfer.stderr)
                            if transfer.returncode
                            else "NO_ORIGINAL_HD_BYTES",
                        }
                    )
                    continue
                result: dict[str, Any] = {
                    "status": "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED",
                    "source_url": url,
                    "source_video_id": video_id,
                    "source_channel_id": channel_id,
                    "duration": float(metadata.get("duration") or 0),
                    "title": str(metadata.get("title") or "")[:160],
                    "transport_strategy": strategy_name,
                    "extractor_trace": safe_extractor_trace(metadata_run.stderr),
                    "attempts": attempts,
                }
                # The full original MUST be downloaded in the same Modal
                # worker, with the same verified extractor, before any handoff.
                if run_key:
                    result["staging"] = stage_official_original.local(result, run_key)
                return result
            except (OSError, ValueError, subprocess.TimeoutExpired, RuntimeError) as exc:
                # Preserve the verified staging failure type in durable JSON.
                # A corrupt downloaded file is not a YouTube bot challenge.
                detail = str(exc)[:360] if isinstance(exc, RuntimeError) else ""
                attempts.append(
                    {
                        "url": url,
                        "strategy": strategy_name,
                        "reason": f"{type(exc).__name__}: {detail}"
                        if detail
                        else type(exc).__name__,
                    }
                )
        # A challenge from two clients does not establish that all configured
        # clients fail. Exhaust this exact video's bounded strategy set first.
        if ip_challenges == len(ACQUISITION_STRATEGIES):
            blocked_video_count += 1
            if blocked_video_count >= 2:
                return {"status": "YOUTUBE_EGRESS_BOT_CHALLENGE", "attempts": attempts}
    status = (
        "YOUTUBE_EGRESS_BOT_CHALLENGE"
        if total_ip_challenges and total_ip_challenges >= len(attempts) - 1
        else "NO_ACCESSIBLE_ORIGINAL_YOUTUBE"
    )
    return {"status": status, "attempts": attempts}


def verify_staged_media_probe(
    *, returncode: int, stdout: str, stderr: str, expected_seconds: float
) -> float:
    """Check actual HD bytes before publishing them to the shared Modal volume."""
    if returncode:
        raise RuntimeError(
            "CORRUPT_OR_INCOMPLETE_HD_TRANSFER: ffprobe failed: " + stderr.strip()[-400:]
        )
    try:
        media_info = json.loads(stdout)
        streams = media_info["streams"]
        staged_seconds = float(media_info["format"]["duration"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("CORRUPT_OR_INCOMPLETE_HD_TRANSFER: invalid ffprobe metadata") from exc
    if not isinstance(streams, list) or not 0 < staged_seconds < float("inf"):
        raise RuntimeError("CORRUPT_OR_INCOMPLETE_HD_TRANSFER: invalid stream or duration")
    if expected_seconds <= 0 or expected_seconds > 3600:
        raise RuntimeError("SOURCE_EXCEEDS_FULL_ANALYSIS_LIMIT")
    if staged_seconds + 30 < expected_seconds:
        raise RuntimeError(
            "SOURCE_DURATION_INCOMPLETE: staged media ends before the verified scan window"
        )
    if not has_production_hd_video_stream(streams):
        raise RuntimeError("MISSING_PRODUCTION_HD_VIDEO_STREAM")
    if not any(
        isinstance(stream, dict) and stream.get("codec_type") == "audio" for stream in streams
    ):
        raise RuntimeError("MISSING_AUDIO_STREAM")
    return staged_seconds


@app.function(image=image, volumes={"/tjr-media": volume}, timeout=1600, cpu=2, memory=2048)
def stage_official_original(selected: dict[str, Any], run_key: str) -> dict[str, Any]:
    """Download actual allowlisted original bytes via the verified Modal network."""
    import hashlib
    import re
    import subprocess
    import tempfile
    from pathlib import Path

    video_id = str(selected["source_video_id"])
    channel_id = str(selected["source_channel_id"])
    if (
        channel_id not in {"UCf1q6dhccWr6eQEcFFnJSbA"}
        or not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id)
        or not re.fullmatch(r"\d{4,20}-\d{1,4}", run_key)
    ):
        raise RuntimeError("staging rejected an unapproved channel, ID or run identifier")
    url = f"https://www.youtube.com/watch?v={video_id}"
    if url != selected.get("source_url"):
        raise RuntimeError("original source URL mismatch")
    expected_seconds = float(selected.get("duration") or 0)
    folder = Path("/tjr-media") / "runs" / run_key / video_id
    folder.mkdir(parents=True, exist_ok=True)
    verified_strategy = str(selected.get("transport_strategy") or "")
    extractor_args = next(
        (args for label, args in ACQUISITION_STRATEGIES if label == verified_strategy),
        None,
    )
    if extractor_args is None:
        raise RuntimeError("unrecognized previously verified original YouTube transport")
    # Each source/transport attempt receives a fresh private directory.
    # Never let a partial prior download look like a successful new original.
    with tempfile.TemporaryDirectory(prefix="acquire-", dir=folder) as scratch:
        scratch_dir = Path(scratch)
        command = [
            "yt-dlp",
            "--ignore-config",
            "--js-runtimes",
            "node",
            *extractor_args,
            "--retries",
            "10",
            "--fragment-retries",
            "10",
            "--abort-on-unavailable-fragments",
            "--no-playlist",
            "--no-warnings",
            "--merge-output-format",
            "mp4",
            *source_download_sections(expected_seconds),
            "-f",
            "bv*[width>=1280][height>=720][height<=1080]+ba/"
            "bv*[width>=720][height>=1280][width<=1080]+ba/"
            "b[width>=1280][height>=720]/b[width>=720][height>=1280]",
            "--no-part",
            "-o",
            str(scratch_dir / "original.%(ext)s"),
            url,
        ]
        try:
            outcome = subprocess.run(command, capture_output=True, timeout=1330, check=False)
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("ORIGINAL_TRANSFER_TIMEOUT") from exc
        if outcome.returncode:
            reason = _transport_error(outcome.stderr.decode("utf-8", errors="replace"))
            raise RuntimeError(f"ORIGINAL_TRANSFER_FAILED: exit={outcome.returncode} {reason}")
        source_files = [
            item
            for item in scratch_dir.glob("original.*")
            if item.is_file() and item.suffix.lower() in {".mp4", ".mkv", ".webm"}
        ]
        if len(source_files) != 1 or source_files[0].stat().st_size < 1024:
            raise RuntimeError("ORIGINAL_TRANSFER_MISSING_OR_TOO_SMALL")
        downloaded = source_files[0]
        inspect = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(downloaded),
            ],
            capture_output=True,
            text=True,
            timeout=50,
            check=False,
        )
        staged_seconds = verify_staged_media_probe(
            returncode=inspect.returncode,
            stdout=inspect.stdout,
            stderr=inspect.stderr,
            expected_seconds=expected_seconds,
        )
        with downloaded.open("rb") as media:
            digest = hashlib.file_digest(media, "sha256").hexdigest()
        # Publish only verified media; failed attempts leave no corrupt original.
        original = folder / f"original{downloaded.suffix.lower()}"
        os.replace(downloaded, original)
    volume.commit()
    return {
        "status": "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED",
        "video_id": video_id,
        "channel_id": channel_id,
        "public_video_url": url,
        "source_remote_path": f"runs/{run_key}/{video_id}/{original.name}",
        "source_sha256": digest,
        "size_bytes": original.stat().st_size,
        "title": str(selected.get("title") or ""),
        "duration": expected_seconds,
        "staged_duration_seconds": staged_seconds,
        "source_scan_complete": staged_seconds + 30 >= expected_seconds,
    }


@app.local_entrypoint()
def main() -> None:
    from clipper.brief import load_brief
    from scripts.tjr_youtube_preview import (
        constrain_official_sources,
        discover_official_uploads,
    )

    root = Path("tjr-modal-probe")
    root.mkdir(parents=True, exist_ok=True)
    output = root / "verified-original-egress.json"
    try:
        candidates, discovery_failures = discover_official_uploads()
        channel_id = os.getenv("TJR_MODAL_CHANNEL_ID", "").strip()
        if channel_id and channel_id not in {"UCf1q6dhccWr6eQEcFFnJSbA"}:
            raise RuntimeError(
                "requested channel is not one of the Reach-approved YouTube channels"
            )
        requested_video = os.getenv("TJR_SOURCE_VIDEO_ID", "").strip() or None
        published_after = load_brief(
            Path("campaigns/reach-double-coverage-dedicated.yaml")
        ).published_after
        official = constrain_official_sources(
            candidates,
            requested_id=requested_video,
            published_after=published_after,
            target_channel_id=channel_id or None,
        )
        excluded = {
            item.strip()
            for item in os.getenv("TJR_MODAL_EXCLUDE_VIDEO_IDS", "").split(",")
            if item.strip()
        }
        if requested_video and requested_video in excluded:
            raise RuntimeError("explicit requested video cannot also be excluded")
        official = [item for item in official if item.video_id not in excluded]
        # The target-channel constraint already orders direct discovery newest-first.
        # Metadata verification below skips live/upcoming/short/inaccessible candidates.
        inputs = [
            {"video_id": item.video_id, "channel_id": item.channel_id} for item in official[:8]
        ]
        if not inputs:
            raise RuntimeError("No feed-confirmed campaign YouTube videos")
        stage_media = os.getenv("TJR_MODAL_STAGE_ORIGINAL") == "1"
        run_key = (
            os.environ["GITHUB_RUN_ID"] + "-" + os.environ.get("GITHUB_RUN_ATTEMPT", "1")
            if stage_media
            else ""
        )
        region_attempts: list[dict[str, str]] = []
        result: dict[str, Any] = {
            "status": "NO_ACCESSIBLE_ORIGINAL_YOUTUBE",
            "attempts": [],
        }
        # One explicitly selected Modal route. Do not switch clouds automatically.
        routes = (("cloud:gcp", "cloud", "gcp"),)
        for label, kind, value in routes[:MAX_MODAL_EGRESS_ATTEMPTS]:
            provider = (
                inspect_original_youtube.with_options(cloud=value)
                if kind == "cloud"
                else inspect_original_youtube.with_options(region=value)
                if kind == "region"
                else inspect_original_youtube
            )
            provider = provider.with_options(env={"TJR_ACQUISITION_DIAGNOSTICS": "1"})
            try:
                candidate = provider.remote(inputs, run_key)
                result = candidate
                if candidate.get("status") == "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED" and (
                    not stage_media
                    or candidate.get("staging", {}).get("status")
                    == "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED"
                ):
                    result["selected_modal_egress"] = label
                    break
                region_attempts.append({"egress": label, "status": str(candidate.get("status"))})
            except Exception as exc:
                region_attempts.append({"egress": label, "error": type(exc).__name__})
        result["region_attempts"] = region_attempts
        result["egress_attempt_limit"] = MAX_MODAL_EGRESS_ATTEMPTS
        result["discovery_failures"] = discovery_failures
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if result["status"] != "EXACT_OFFICIAL_YOUTUBE_HD_MEDIA_BYTES_VERIFIED":
            raise RuntimeError(
                "No tested Modal region could fetch HD bytes from the official YouTube video"
            )
        print("Verified original YouTube URL:", result["source_url"])
        if stage_media:
            staging = result.get("staging")
            if (
                not isinstance(staging, dict)
                or staging.get("status") != "REAL_OFFICIAL_YOUTUBE_ORIGINAL_STAGED"
            ):
                raise RuntimeError("No same-worker verified HD original was staged")
            (root / "staged-original.json").write_text(
                json.dumps(staging, indent=2) + "\n",
                encoding="utf-8",
            )
            print("Verified original media staged on the same Modal worker")
    except Exception:
        if not output.is_file():
            output.write_text(
                json.dumps({"status": "MODAL_PROBE_FAILED_BEFORE_REMOTE_RESULT"}, indent=2) + "\n",
                encoding="utf-8",
            )
        raise
