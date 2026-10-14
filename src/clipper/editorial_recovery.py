"""Read-only retrieval of interrupted GPU qualification checkpoints."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, cast


def recover_gpu_evidence(report: dict[str, Any], output: Path, volume: Any) -> dict[str, Any]:
    """Retrieve fixed evidence files for submitted job IDs; never start inference."""
    if report.get("experiment") != "source_position_gpu_qualification":
        raise ValueError("recovery requires a GPU qualification report")
    rows = report.get("profiles")
    if not isinstance(rows, list) or not rows:
        raise ValueError("recovery requires submitted job identities")
    keys = [row.get("job_key") for row in rows if isinstance(row, dict)]
    if len(keys) != len(rows) or any(
        not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None for key in keys
    ):
        raise ValueError("recovery job key must be a SHA-256 identifier")
    result: dict[str, Any] = {
        "experiment": "gpu_checkpoint_recovery",
        "production_approved": False,
        "inference_started": False,
        "jobs": [],
    }
    for key in cast(list[str], keys):
        row: dict[str, Any] = {"job_key": key, "files": [], "errors": []}
        for name in ("cold/proof.json", "cold/review-request-cache.json", "manifest.json"):
            target = output.parent / "reviewer-gpu-evidence" / "recovered" / key / name
            try:
                data = b"".join(volume.read_file(f"reviewer/qualification/{key}/{name}"))
                json.loads(data)  # Do not label incomplete JSON as a recovered checkpoint.
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                row["files"].append(
                    {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                )
            except Exception as error:
                row["errors"].append({"name": name, "error": f"{type(error).__name__}: {error}"})
        result["jobs"].append(row)
    try:
        result["weight_files"] = [
            {"path": entry.path, "bytes": entry.size}
            for entry in volume.listdir("reviewer/weights")
        ]
    except Exception as error:
        result["weight_listing_error"] = f"{type(error).__name__}: {error}"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
