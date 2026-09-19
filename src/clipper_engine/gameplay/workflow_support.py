from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _runner(source: str) -> str:
    return "ubuntu-22.04" if source == "batch2" else "ubuntu-24.04"


def _source_order(allocation: dict[str, Any]) -> list[str]:
    explicit = [str(item) for item in allocation.get("source_order") or []]
    if explicit:
        return explicit
    return [str(item) for item in allocation.get("source_allocations", {})]


def _clip_ffv1_nut_contract_failures(item: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    canonical = item.get("canonical_master") or {}
    fidelity = item.get("source_fidelity") or {}
    fidelity_checks = fidelity.get("checks") or {}
    canonical_transport = canonical.get("transport_architecture_qa") or {}
    canonical_transport_checks = canonical_transport.get("checks") or {}
    staging = fidelity.get("staging_fidelity") or []

    actual_container = str(canonical.get("actual_container") or "")
    actual_containers = {
        part.strip().lower() for part in actual_container.split(",") if part.strip()
    }
    if (
        str(canonical.get("canonical_container") or "").lower() != "nut"
        or "nut" not in actual_containers
    ):
        failures.append("canonical container is not verified NUT")
    if str(canonical.get("canonical_codec") or "").lower() != "ffv1 level=3 lossless":
        failures.append("canonical declared codec is not FFV1 level 3 lossless")
    if str(canonical.get("actual_video_codec") or "").lower() != "ffv1":
        failures.append("canonical actual video codec is not FFV1")
    if str(canonical.get("actual_audio_codec") or "").lower() != "pcm_s16le":
        failures.append("canonical actual audio codec is not PCM s16le")
    if not canonical_transport_checks or not all(
        bool(value) for value in canonical_transport_checks.values()
    ):
        failures.append("canonical FFV1/NUT transport QA is missing or failed")
    if canonical.get("source_contract_metadata_authority") != "original_input_probe":
        failures.append("original source media contract is not authoritative")
    if canonical.get("nut_color_metadata_authoritative") is not False:
        failures.append("NUT color metadata authority is not explicitly disabled")

    if not staging:
        failures.append("staging fidelity evidence is missing")
    for index, stage in enumerate(staging, 1):
        transport = stage.get("transport") or {}
        checks = transport.get("checks") or {}
        format_name = str(transport.get("format_name") or "")
        containers = {part.strip().lower() for part in format_name.split(",") if part.strip()}
        video = transport.get("video_profile") or {}
        audio = transport.get("audio_profile") or {}
        if "nut" not in containers:
            failures.append(f"stage {index} actual container is not NUT")
        if str(video.get("codec_name") or "").lower() != "ffv1":
            failures.append(f"stage {index} actual video codec is not FFV1")
        if str(audio.get("codec_name") or "").lower() != "pcm_s16le":
            failures.append(f"stage {index} actual audio codec is not PCM s16le")
        if not checks or not all(bool(value) for value in checks.values()):
            failures.append(f"stage {index} transport QA is missing or failed")
        if not bool((stage.get("checks") or {}).get("exact_decoded_frame_hash_match")):
            failures.append(f"stage {index} exact decoded-frame hash identity is missing or failed")

    if not bool(fidelity_checks.get("source_media_contract_reapplied_to_final_h264")):
        failures.append("final H.264 did not prove source media-contract reapplication")
    if not bool(fidelity_checks.get("ssim_encoder_fidelity")):
        failures.append("final H.264 SSIM fidelity gate did not pass")
    if not bool(fidelity_checks.get("psnr_encoder_fidelity")):
        failures.append("final H.264 PSNR fidelity gate did not pass")
    return failures


def render_matrix(allocation_path: Path, github_output: Path) -> list[dict[str, Any]]:
    allocation = _read(allocation_path)
    source_allocations = dict(allocation.get("source_allocations") or {})
    include: list[dict[str, Any]] = []
    distribution: dict[str, int] = {}

    for source in _source_order(allocation):
        source_allocation = dict(source_allocations.get(source) or {})
        keys = [str(item) for item in source_allocation.get("plan_keys") or []]
        declared_count = int(source_allocation.get("count", -1))
        if declared_count != len(keys):
            raise RuntimeError(
                f"{source}: orchestrator declared {declared_count} clips but emitted "
                f"{len(keys)} plan keys"
            )
        distribution[source] = len(keys)
        for ordinal, key_text in enumerate(keys, 1):
            include.append(
                {
                    "source": source,
                    "plan_key": key_text,
                    "ordinal": ordinal,
                    "runner": _runner(source),
                    "clip_id": f"{source}-{ordinal:02d}-{key_text[:8]}",
                }
            )

    derived_count = int(allocation.get("target_count", -1))
    selected_count = int(allocation.get("selected_count", -1))
    if derived_count < 0 or selected_count != derived_count:
        raise RuntimeError(
            f"orchestrator cardinality mismatch: target={derived_count} selected={selected_count}"
        )
    if len(include) != derived_count:
        raise RuntimeError(
            "render matrix does not match orchestrator-derived cardinality: "
            f"matrix={len(include)} target={derived_count}"
        )
    if not include:
        raise RuntimeError("orchestrator derived no renderable clips")

    value = json.dumps({"include": include}, separators=(",", ":"))
    distribution_value = json.dumps(distribution, separators=(",", ":"))
    with github_output.open("a", encoding="utf-8") as output:
        output.write(f"render_matrix={value}\n")
        output.write(f"derived_clip_count={derived_count}\n")
        output.write(f"source_distribution={distribution_value}\n")
    print(
        json.dumps(
            {
                "orchestrator_render_matrix": "PASS",
                "derived_clip_count": derived_count,
                "source_distribution": distribution,
            }
        )
    )
    return include


def batch_summaries(
    allocation_path: Path, config_path: Path, clip_meta: Path, output_dir: Path
) -> None:
    allocation = _read(allocation_path)
    config = _read(config_path)
    results = [_read(path) for path in clip_meta.rglob("*_clip_result.json")]
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in _source_order(allocation):
        expected = [
            str(item)
            for item in allocation.get("source_allocations", {})
            .get(source, {})
            .get("plan_keys", [])
        ]
        actual = sorted(
            [item for item in results if item.get("source_key") == source],
            key=lambda item: int(item.get("ordinal", 0)),
        )
        actual_keys = [str(item.get("plan_key")) for item in actual]
        failures: list[str] = []
        architecture_failures: list[str] = []
        if actual_keys != expected:
            failures.append(
                f"{source}: rendered clip keys {actual_keys} != allocated keys {expected}"
            )
        if any(item.get("status") != "PASS" for item in actual):
            failures.append(f"{source}: one or more clip results did not pass")
        for item in actual:
            clip_failures = _clip_ffv1_nut_contract_failures(item)
            architecture_failures.extend(
                f"{source} clip {item.get('ordinal', '?')}: {failure}" for failure in clip_failures
            )
        failures.extend(architecture_failures)
        architecture_passed = (
            bool(actual or not expected)
            and not architecture_failures
            and len(actual) == len(expected)
        )
        summary = {
            "mode": "shadow",
            "source_key": source,
            "selected_count": len(actual),
            "rendered_count": len(actual),
            "selected_plan_keys": actual_keys,
            "stories": [str(item.get("story_type")) for item in actual],
            "verified_finishing_move_count": len(
                config.get("finishing_move_detector", {}).get("verified_spans", {}).get(source, [])
            ),
            "selected_finishing_move_count": sum(
                1 for item in actual if bool(item.get("finishing_move"))
            ),
            "unplanned_source_cut_count": sum(
                int(item.get("unplanned_source_cut_count", 0)) for item in actual
            ),
            "ffv1_nut_transport_contract_passed": architecture_passed,
            "technical_qa_passed": (
                bool(actual or not expected)
                and len(actual) == len(expected)
                and architecture_passed
                and all(bool(item.get("technical_qa_passed")) for item in actual)
                and all(bool(item.get("source_fidelity_qa_passed")) for item in actual)
            ),
            "failure": failures or None,
        }
        (output_dir / f"{source}_pipeline_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    matrix = sub.add_parser("render-matrix")
    matrix.add_argument("--allocation", type=Path, required=True)
    matrix.add_argument(
        "--github-output", type=Path, default=Path(os.environ.get("GITHUB_OUTPUT", ""))
    )
    summary = sub.add_parser("batch-summaries")
    summary.add_argument("--allocation", type=Path, required=True)
    summary.add_argument("--config", type=Path, required=True)
    summary.add_argument("--clip-meta", type=Path, required=True)
    summary.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "render-matrix":
        if not str(args.github_output):
            raise RuntimeError("GITHUB_OUTPUT is not available")
        render_matrix(args.allocation, args.github_output)
    else:
        batch_summaries(args.allocation, args.config, args.clip_meta, args.output_dir)


if __name__ == "__main__":
    main()
