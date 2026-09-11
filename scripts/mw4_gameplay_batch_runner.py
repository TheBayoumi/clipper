from __future__ import annotations

import mw4_gameplay_batch as batch

_original_build_filter = batch.build_filter


def build_filter(
    candidate: batch.Candidate,
    config: dict[str, object],
    *,
    logo_enabled: bool,
) -> tuple[str, float]:
    graph, duration = _original_build_filter(
        candidate,
        config,
        logo_enabled=logo_enabled,
    )
    unused_by_profile = {
        "precision_punch": ("z3", "shake"),
        "impact_flash": ("z1", "z2", "z3"),
        "chain_escalation": (),
        "cold_open_teaser": ("z1", "z3"),
    }
    unused = unused_by_profile[candidate.effect_profile]
    if unused:
        graph += ";" + ";".join(f"[{label}]nullsink" for label in unused)
    return graph, duration


batch.build_filter = build_filter

if __name__ == "__main__":
    batch.main()
