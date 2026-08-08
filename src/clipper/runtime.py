from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from .providers.base import InferenceUsage

StageStatus = Literal["PENDING", "RUNNING", "SUCCESS", "FAILED"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True, slots=True)
class StageState:
    name: str
    status: StageStatus
    started_at: str | None
    updated_at: str
    completed: int = 0
    total: int | None = None
    timeout_seconds: float | None = None
    checkpoint: str | None = None
    message: str | None = None
    failure_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StageJournal:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.states: dict[str, StageState] = {}
        if path.is_file():
            self._load()

    def _load(self) -> None:
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        raw = payload.get("stages") if isinstance(payload, dict) else None
        if not isinstance(raw, dict):
            raise ValueError("progress journal stages must be an object")
        states: dict[str, StageState] = {}
        for name, value in raw.items():
            if not isinstance(value, dict):
                raise ValueError("progress journal stage must be an object")
            status = str(value.get("status") or "")
            if status not in {"PENDING", "RUNNING", "SUCCESS", "FAILED"}:
                raise ValueError(f"invalid progress stage status: {status}")
            states[str(name)] = StageState(
                name=str(name),
                status=status,  # type: ignore[arg-type]
                started_at=str(value["started_at"]) if value.get("started_at") else None,
                updated_at=str(value.get("updated_at") or _now()),
                completed=int(value.get("completed") or 0),
                total=int(value["total"]) if value.get("total") is not None else None,
                timeout_seconds=(
                    float(value["timeout_seconds"])
                    if value.get("timeout_seconds") is not None
                    else None
                ),
                checkpoint=str(value["checkpoint"]) if value.get("checkpoint") else None,
                message=str(value["message"]) if value.get("message") else None,
                failure_reason=(
                    str(value["failure_reason"]) if value.get("failure_reason") else None
                ),
            )
        self.states = states

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "clipper-progress-v1",
            "updated_at": _now(),
            "stages": {name: state.to_dict() for name, state in sorted(self.states.items())},
        }
        temp = self.path.with_name(f".{self.path.name}.{os.getpid()}.tmp")
        temp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        os.replace(temp, self.path)

    def start(
        self,
        name: str,
        *,
        total: int | None = None,
        timeout_seconds: float | None = None,
        checkpoint: str | None = None,
        message: str | None = None,
    ) -> StageState:
        if not name.strip():
            raise ValueError("progress stage name cannot be empty")
        now = _now()
        previous = self.states.get(name)
        state = StageState(
            name=name,
            status="RUNNING",
            started_at=previous.started_at if previous and previous.started_at else now,
            updated_at=now,
            completed=previous.completed if previous else 0,
            total=total if total is not None else (previous.total if previous else None),
            timeout_seconds=(
                timeout_seconds
                if timeout_seconds is not None
                else (previous.timeout_seconds if previous else None)
            ),
            checkpoint=checkpoint
            if checkpoint is not None
            else (previous.checkpoint if previous else None),
            message=message,
        )
        self.states[name] = state
        self._write()
        return state

    def progress(
        self,
        name: str,
        completed: int,
        *,
        total: int | None = None,
        checkpoint: str | None = None,
        message: str | None = None,
    ) -> StageState:
        previous = self.states.get(name)
        if previous is None:
            previous = self.start(name, total=total)
        resolved_total = total if total is not None else previous.total
        if completed < 0 or (resolved_total is not None and completed > resolved_total):
            raise ValueError("progress completed count is invalid")
        state = StageState(
            name=name,
            status="RUNNING",
            started_at=previous.started_at,
            updated_at=_now(),
            completed=completed,
            total=resolved_total,
            timeout_seconds=previous.timeout_seconds,
            checkpoint=checkpoint if checkpoint is not None else previous.checkpoint,
            message=message,
        )
        self.states[name] = state
        self._write()
        return state

    def complete(
        self, name: str, *, checkpoint: str | None = None, message: str | None = None
    ) -> StageState:
        previous = self.states.get(name)
        if previous is None:
            previous = self.start(name)
        completed = previous.total if previous.total is not None else previous.completed
        state = StageState(
            name=name,
            status="SUCCESS",
            started_at=previous.started_at,
            updated_at=_now(),
            completed=completed,
            total=previous.total,
            timeout_seconds=previous.timeout_seconds,
            checkpoint=checkpoint if checkpoint is not None else previous.checkpoint,
            message=message,
        )
        self.states[name] = state
        self._write()
        return state

    def fail(self, name: str, reason: str, *, checkpoint: str | None = None) -> StageState:
        if not reason.strip():
            raise ValueError("progress failure reason cannot be empty")
        previous = self.states.get(name)
        if previous is None:
            previous = self.start(name)
        state = StageState(
            name=name,
            status="FAILED",
            started_at=previous.started_at,
            updated_at=_now(),
            completed=previous.completed,
            total=previous.total,
            timeout_seconds=previous.timeout_seconds,
            checkpoint=checkpoint if checkpoint is not None else previous.checkpoint,
            failure_reason=reason,
        )
        self.states[name] = state
        self._write()
        return state


class ComputeBudget:
    def __init__(self, max_cost_usd: float = 1.0, *, large_vlm_fraction: float = 0.25) -> None:
        if max_cost_usd <= 0:
            raise ValueError("compute budget must be positive")
        if not 0 < large_vlm_fraction <= 1:
            raise ValueError("large VLM budget fraction must be in (0, 1]")
        self.max_cost_usd = max_cost_usd
        self.large_vlm_fraction = large_vlm_fraction
        self.usage: list[InferenceUsage] = []

    @property
    def estimated_cost_usd(self) -> float:
        return round(sum(item.estimated_cost_usd for item in self.usage), 6)

    @property
    def gpu_seconds(self) -> float:
        return round(sum(item.gpu_seconds for item in self.usage), 4)

    def record(self, usage: InferenceUsage) -> None:
        self.usage.append(usage)

    def record_mapping(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        try:
            self.record(
                InferenceUsage(
                    provider=str(payload.get("provider") or "unknown"),
                    started_at=str(payload.get("started_at") or "unknown"),
                    duration_seconds=float(payload.get("duration_seconds") or 0.0),
                    gpu_type=str(payload["gpu_type"]) if payload.get("gpu_type") else None,
                    gpu_seconds=float(payload.get("gpu_seconds") or 0.0),
                    peak_vram_mb=(
                        float(payload["peak_vram_mb"])
                        if payload.get("peak_vram_mb") is not None
                        else None
                    ),
                    input_units=int(payload.get("input_units") or 0),
                    output_units=int(payload.get("output_units") or 0),
                    estimated_cost_usd=float(payload.get("estimated_cost_usd") or 0.0),
                )
            )
        except (TypeError, ValueError):
            return

    def allow_large_vlm(self, *, estimated_next_cost_usd: float = 0.0) -> bool:
        if estimated_next_cost_usd < 0:
            raise ValueError("estimated next cost cannot be negative")
        escalation_cutoff = self.max_cost_usd * (1.0 - self.large_vlm_fraction)
        return self.estimated_cost_usd + estimated_next_cost_usd <= escalation_cutoff

    def to_dict(self) -> dict[str, object]:
        return {
            "max_cost_usd": self.max_cost_usd,
            "estimated_cost_usd": self.estimated_cost_usd,
            "gpu_seconds": self.gpu_seconds,
            "large_vlm_fraction": self.large_vlm_fraction,
            "large_vlm_allowed": self.allow_large_vlm(),
            "usage": [asdict(item) for item in self.usage],
        }
