from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any


_MAX_INT_FIELDS = frozenset(
    {
        "largest_good_input_tokens",
        "largest_dynamic_good_input_tokens",
        "largest_offloaded_good_input_tokens",
    }
)
_MIN_INT_FIELDS = frozenset(
    {
        "smallest_bad_input_tokens",
        "smallest_dynamic_oom_input_tokens",
        "smallest_offloaded_oom_input_tokens",
    }
)
_MAX_FLOAT_FIELDS = frozenset({"output_tokens_per_input_token"})


def _valid_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    parsed = float(value)
    return parsed if parsed > 0 else None


def merge_editorial_capacity_state(
    current: dict[str, Any],
    incoming: dict[str, Any],
) -> dict[str, Any]:
    """Merge independently learned capacity observations conservatively."""

    merged: dict[str, Any] = {
        str(family): dict(entry) if isinstance(entry, dict) else entry
        for family, entry in current.items()
    }
    for family, raw_incoming in incoming.items():
        if not isinstance(raw_incoming, dict):
            merged[str(family)] = raw_incoming
            continue

        raw_current = merged.get(str(family))
        entry = dict(raw_current) if isinstance(raw_current, dict) else {}
        for key, value in raw_incoming.items():
            if key in _MAX_INT_FIELDS:
                candidate = _valid_number(value)
                existing = _valid_number(entry.get(key))
                if candidate is not None:
                    entry[key] = int(candidate if existing is None else max(existing, candidate))
                continue
            if key in _MIN_INT_FIELDS:
                candidate = _valid_number(value)
                existing = _valid_number(entry.get(key))
                if candidate is not None:
                    entry[key] = int(candidate if existing is None else min(existing, candidate))
                continue
            if key in _MAX_FLOAT_FIELDS:
                candidate = _valid_number(value)
                existing = _valid_number(entry.get(key))
                if candidate is not None:
                    entry[key] = candidate if existing is None else max(existing, candidate)
                continue
            entry[key] = value
        merged[str(family)] = entry
    return merged


def load_editorial_capacity_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def write_editorial_capacity_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
