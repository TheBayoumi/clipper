from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable, Literal

from .stage_contracts import StageIdentity, content_fingerprint

StageStatus = Literal["RUNNING", "PASS", "FAILED"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class StageResult:
    output: object
    usage: dict[str, object] = field(default_factory=dict)
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if self.cost_usd < 0:
            raise ValueError("stage cost cannot be negative")


@dataclass(frozen=True, slots=True)
class StageRecord:
    identity: StageIdentity
    status: StageStatus
    attempt_count: int
    started_at: str
    completed_at: str | None
    output_fingerprint: str | None
    usage: dict[str, object]
    cost_usd: float
    error_type: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.attempt_count <= 0:
            raise ValueError("stage attempt_count must be positive")
        if self.cost_usd < 0:
            raise ValueError("stage cost cannot be negative")
        if self.status == "PASS" and (not self.completed_at or not self.output_fingerprint):
            raise ValueError("passing stage requires completion and output evidence")
        if self.status == "FAILED" and (not self.completed_at or not self.error_type):
            raise ValueError("failed stage requires completion and error evidence")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["identity"] = self.identity.to_dict()
        return payload


class DagStore:
    """Persistent content-addressed stage store with exact dependency reuse."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _directory(self, identity: StageIdentity) -> Path:
        safe_stage = "".join(
            character if character.isalnum() or character in {"-", "_"} else "-"
            for character in identity.stage_name
        )
        return self.root / safe_stage / identity.cache_key

    def _record_path(self, identity: StageIdentity) -> Path:
        return self._directory(identity) / "stage.json"

    def _output_path(self, identity: StageIdentity) -> Path:
        return self._directory(identity) / "output.json"

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)

    def _read_record_payload(self, identity: StageIdentity) -> dict[str, Any] | None:
        path = self._record_path(identity)
        if not path.is_file():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return value if isinstance(value, dict) else None

    def attempt_count(self, identity: StageIdentity) -> int:
        payload = self._read_record_payload(identity)
        if payload is None:
            return 0
        raw = payload.get("attempt_count")
        return int(raw) if isinstance(raw, int) and not isinstance(raw, bool) and raw >= 0 else 0

    def cached_output(self, identity: StageIdentity) -> object | None:
        record = self._read_record_payload(identity)
        output_path = self._output_path(identity)
        if record is None or record.get("status") != "PASS" or not output_path.is_file():
            return None
        if record.get("identity", {}).get("cache_key") != identity.cache_key:
            return None
        try:
            output: object = json.loads(output_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        expected = record.get("output_fingerprint")
        if not isinstance(expected, str) or content_fingerprint(output) != expected:
            return None
        return output

    def execute(
        self,
        identity: StageIdentity,
        operation: Callable[[], StageResult | object],
    ) -> tuple[object, bool]:
        cached = self.cached_output(identity)
        if cached is not None:
            return cached, True

        attempt = self.attempt_count(identity) + 1
        started = _now()
        running = StageRecord(
            identity=identity,
            status="RUNNING",
            attempt_count=attempt,
            started_at=started,
            completed_at=None,
            output_fingerprint=None,
            usage={},
            cost_usd=0.0,
        )
        self._write_json(self._record_path(identity), running.to_dict())

        try:
            raw_result = operation()
            result = raw_result if isinstance(raw_result, StageResult) else StageResult(raw_result)
            fingerprint = content_fingerprint(result.output)
            self._write_json(self._output_path(identity), result.output)
            passed = StageRecord(
                identity=identity,
                status="PASS",
                attempt_count=attempt,
                started_at=started,
                completed_at=_now(),
                output_fingerprint=fingerprint,
                usage=dict(result.usage),
                cost_usd=result.cost_usd,
            )
            self._write_json(self._record_path(identity), passed.to_dict())
            return result.output, False
        except Exception as exc:
            failed = StageRecord(
                identity=identity,
                status="FAILED",
                attempt_count=attempt,
                started_at=started,
                completed_at=_now(),
                output_fingerprint=None,
                usage={},
                cost_usd=0.0,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            self._write_json(self._record_path(identity), failed.to_dict())
            raise
