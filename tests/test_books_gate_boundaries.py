import dataclasses

import pytest
from test_books_chapter_hybrid import _keep_only, _synthetic_ledger

from clipper.books import chapter_hybrid as hybrid
from clipper.books import model_profile


def _links(n_units: list[int], roles: list[str]) -> list[dict[str, object]]:
    return [
        {"id": f"s{u:04d}", "start": 0, "end": 3, "role": r}
        for u, r in zip(n_units, roles, strict=True)
    ]


def _accepts(links: list[dict[str, object]]) -> bool:
    try:
        hybrid.reject_degenerate_links(links, "source_links")
    except hybrid.HybridStop:
        return False
    return True


R = ["ASSERTION", "AUTHOR_EXAMPLE", "COUNTEREXAMPLE", "QUALIFICATION"]


def test_gpt1_20_degeneracy_boundaries() -> None:
    # count boundary: 5 homogeneous accepted, 6 rejected (MIN_LINKS=6)
    assert _accepts(_links([2, 3, 4, 5, 6], ["ASSERTION"] * 5))
    assert not _accepts(_links([2, 3, 4, 5, 6, 7], ["ASSERTION"] * 6))
    # unit share: 3 of 6 = 50% accepted, 4 of 6 rejected
    assert _accepts(_links([2, 2, 2, 3, 4, 5], [R[i % 4] for i in range(6)]))
    assert not _accepts(_links([2, 2, 2, 2, 3, 4], [R[i % 4] for i in range(6)]))
    # role share: 8 of 10 = 80% accepted, 9 of 10 rejected
    assert _accepts(_links(list(range(2, 12)), ["ASSERTION"] * 8 + ["COUNTEREXAMPLE"] * 2))
    assert not _accepts(_links(list(range(2, 12)), ["ASSERTION"] * 9 + ["COUNTEREXAMPLE"]))


def test_gpt1_21_retention_boundary() -> None:
    ledger = _synthetic_ledger(128)

    def span(sid: str) -> dict[str, object]:
        return {"id": sid, "start": 0, "end": 10, "role": "ASSERTION"}

    def run(k: int) -> bool:
        ids = [f"s{n:04d}" for n in range(1, k + 1)]
        resp = {
            "argument": "a",
            "amendments": [],
            "resolutions": [],
            "beats": [
                {
                    "beat_id": f"beat-{i:03d}",
                    "spans": [span(s)],
                    "meaning": "m",
                    "qualification": "",
                    "causal_link": "c",
                    "essential": True,
                }
                for i, s in enumerate(ids, 1)
            ],
        }
        arg = {"hypothesis": "h", "questions": [], "source_links": [span("s0001")]}
        try:
            hybrid.reconcile(resp, ledger, arg, _keep_only(ledger, set(ids)))
        except hybrid.HybridStop as exc:
            assert "omits nearly all" in str(exc)
            return False
        return True

    assert not run(6)  # 6/128 = 4.69% < 5%
    assert run(7)  # 7/128 = 5.47%


def test_gpt1_22_nested_profile_mutation_cannot_change_fingerprint() -> None:
    base = model_profile.QWEN3_1_7B
    profile = dataclasses.replace(
        base, generation_settings={**base.sampling, "bad_words_ids": [[1, 2]]}
    )
    before = profile.fingerprint
    with pytest.raises((AttributeError, TypeError)):
        profile.generation_settings["bad_words_ids"][0].append(11)
    profile.sampling["bad_words_ids"][0].append(11)  # a returned copy only
    assert profile.fingerprint == before
    assert profile.sampling["bad_words_ids"] == [[1, 2]]
