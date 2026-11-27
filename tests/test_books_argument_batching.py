"""Token-budgeted batching of the argument input. Synthetic text only, model-free."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from public_book_fixture_support import _injected_hybrid_response, _selected_hybrid_source
from test_books_chapter_hybrid import _ledger

from clipper.books import chapter, pipeline_plan
from clipper.books import chapter_hybrid as hybrid
from clipper.books.pipeline_plan import PlanError

ROLES = ["ASSERTION", "AUTHOR_EXAMPLE", "COUNTEREXAMPLE", "QUALIFICATION"]


def _entries(count: int = 101) -> list[dict]:
    return hybrid.validate_ledger(_ledger(count))


def _units(count: int = 101) -> list[dict]:
    return [e for e in _entries(count) if e["kind"] == "CANONICAL_WORK_UNIT"]


def _by_chars(window: list[dict]) -> int:
    """A stand-in tokenizer: fixed overhead plus text length. Real code is injected."""
    return 100 + sum(len(e["text"]) // 4 + 8 for e in window)


# 1. windows are chosen by measured tokens, not by a unit count --------------------------


def test_windows_cover_every_text_unit_once_in_order_and_fit_the_budget() -> None:
    units = _units()
    windows = hybrid.argument_batches(_entries(), budget=600, count=_by_chars)
    flat = [e["id"] for w in windows for e in w]
    assert flat == [e["id"] for e in units]
    assert len(windows) > 1
    assert all(_by_chars(w) <= 600 or len(w) == 1 for w in windows)


def test_a_larger_budget_gives_fewer_windows_and_a_huge_one_gives_one() -> None:
    small = hybrid.argument_batches(_entries(), budget=400, count=_by_chars)
    mid = hybrid.argument_batches(_entries(), budget=1200, count=_by_chars)
    huge = hybrid.argument_batches(_entries(), budget=10**9, count=_by_chars)
    assert len(small) > len(mid) > len(huge) == 1


def test_window_edges_follow_the_tokenizer_not_a_fixed_unit_count() -> None:
    entries = _entries()
    plain = hybrid.argument_batches(entries, budget=600, count=_by_chars)
    # A tokenizer that finds every unit twice as expensive must cut earlier.
    doubled = hybrid.argument_batches(
        entries, budget=600, count=lambda w: 100 + 2 * (_by_chars(w) - 100)
    )
    assert len(doubled) > len(plain)
    assert [len(w) for w in plain] != [len(w) for w in doubled]


def test_gaps_are_not_batched_and_one_oversized_unit_still_gets_its_own_window() -> None:
    entries = _entries()
    assert any(e["kind"] != "CANONICAL_WORK_UNIT" for e in entries)
    windows = hybrid.argument_batches(entries, budget=1, count=_by_chars, max_batches=500)
    assert all(len(w) == 1 for w in windows)
    assert all(e["kind"] == "CANONICAL_WORK_UNIT" for w in windows for e in w)


def test_too_many_windows_is_refused() -> None:
    with pytest.raises(hybrid.HybridStop):
        hybrid.argument_batches(_entries(), budget=1, count=_by_chars, max_batches=5)


# 2. one batch reply -------------------------------------------------------------------------


def _window(count: int = 6) -> list[dict]:
    return _units()[3 : 3 + count]


def _batch(window: list[dict], n: int) -> dict[str, object]:
    links = [{"id": e["id"], "role": ROLES[i % 4]} for i, e in enumerate(window[:n])]
    return {"points": ["The author claims something here."], "source_links": links}


def test_valid_batch_reply_is_accepted_and_ids_are_checked_against_the_window() -> None:
    window = _window()
    out = hybrid.validate_argument_batch(_batch(window, 3), _ledger(), [e["id"] for e in window])
    assert [link["id"] for link in out["source_links"]] == [e["id"] for e in window[:3]]
    outside = _units()[50]["id"]
    bad = _batch(window, 2)
    bad["source_links"].append({"id": outside, "role": "ASSERTION"})
    with pytest.raises(hybrid.HybridStop) as caught:
        hybrid.validate_argument_batch(bad, _ledger(), [e["id"] for e in window])
    assert caught.value.diagnostics and caught.value.diagnostics[0]["id"] == outside


def test_batch_reply_may_exceed_its_quota_because_code_selects_later() -> None:
    window = _window(6)
    out = hybrid.validate_argument_batch(_batch(window, 6), _ledger(), [e["id"] for e in window])
    assert len(out["source_links"]) == 6


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(points=[hybrid.POINT_PLACEHOLDER]),
        lambda d: d.update(points=[]),
        lambda d: d.update(points=["x"] * 9),
        lambda d: d.update(source_links=[]),
        lambda d: d.update(extra="no"),
        lambda d: d.pop("points"),
        lambda d: d.update(stage="argument_batch", response_contract={}),
    ],
)
def test_malformed_batch_replies_are_refused(mutate) -> None:
    window = _window()
    data = _batch(window, 2)
    mutate(data)
    with pytest.raises(hybrid.HybridStop):
        hybrid.validate_argument_batch(data, _ledger(), [e["id"] for e in window])


def test_one_wrapper_is_removed_from_a_batch_reply_too() -> None:
    window = _window()
    out = hybrid.validate_argument_batch(
        {"answer": _batch(window, 2)}, _ledger(), [e["id"] for e in window]
    )
    assert len(out["source_links"]) == 2


# 3. merge in code ---------------------------------------------------------------------------


def _results(per_window: list[list[str]]) -> list[dict[str, object]]:
    return [
        {
            "points": [f"point {i}"],
            "source_links": [
                {"id": sid, "start": 0, "end": 1, "role": ROLES[j % 4]} for j, sid in enumerate(ids)
            ],
        }
        for i, ids in enumerate(per_window)
    ]


def test_quota_is_the_cap_split_over_the_windows() -> None:
    assert hybrid.argument_batch_quota(4) == 6
    assert hybrid.argument_batch_quota(1) == hybrid.MAX_ARGUMENT_LINKS
    assert hybrid.argument_batch_quota(5) == 5
    assert hybrid.argument_batch_quota(40) == 1


def test_merge_keeps_order_dedupes_and_never_exceeds_the_cap() -> None:
    ids = [e["id"] for e in _units()]
    windows = [ids[0:10], ids[10:20], ids[20:30], ids[30:40]]
    merged = hybrid.merge_argument_batches(_results(windows), quota=6)
    assert len(merged["candidate_ids"]) <= hybrid.MAX_ARGUMENT_LINKS
    assert merged["candidate_ids"] == sorted(merged["candidate_ids"], key=ids.index)
    assert len(set(merged["candidate_ids"])) == len(merged["candidate_ids"])
    # Every window with links contributes, even when it over-cites.
    assert all(any(c in w for c in merged["candidate_ids"]) for w in windows)
    # Over-citing cannot crowd out another window: at most quota per window.
    assert all(len([c for c in merged["candidate_ids"] if c in w]) <= 6 for w in windows)


def test_merge_spreads_a_selection_evenly_when_windows_outnumber_the_cap() -> None:
    ids = [e["id"] for e in _units()]
    windows = [[ids[i]] for i in range(40)]
    merged = hybrid.merge_argument_batches(_results(windows), quota=1)
    got = merged["candidate_ids"]
    assert len(got) == hybrid.MAX_ARGUMENT_LINKS
    assert got[0] == ids[0] and got[-1] in ids[30:40]


def test_merge_is_deterministic_and_collects_the_points_in_window_order() -> None:
    ids = [e["id"] for e in _units()]
    results = _results([ids[0:5], ids[5:10]])
    a = hybrid.merge_argument_batches(results, quota=12)
    b = hybrid.merge_argument_batches(json.loads(json.dumps(results)), quota=12)
    assert a == b and a["points"] == ["point 0", "point 1"]


def test_merge_with_no_links_at_all_is_a_clear_stop() -> None:
    with pytest.raises(hybrid.HybridStop):
        hybrid.merge_argument_batches(_results([[], []]), quota=12)


# 4. prompts ---------------------------------------------------------------------------------


def test_batch_prompt_states_its_quota_and_the_plain_argument_prompt_is_unchanged() -> None:
    window = _window()
    payload = {
        "schema": hybrid.ARGUMENT_BATCH_SCHEMA,
        "original_source": hybrid.projection(window),
        "max_source_links": 6,
        "response_contract": {
            "points": [hybrid.POINT_PLACEHOLDER],
            "source_links": [{"id": "<id>", "role": "<role>"}],
        },
    }
    text = hybrid._messages("argument_batch", payload)[1]["content"]
    assert "max_source_links" in text and "at most" in text
    plain = {
        "schema": hybrid.ARGUMENT_SCHEMA,
        "original_source": hybrid.projection(window),
        "response_contract": {"hypothesis": "h"},
    }
    with_notes = {**plain, "batch_points": ["a point"]}
    base = hybrid._messages("argument", plain)[1]["content"]
    noted = hybrid._messages("argument", with_notes)[1]["content"]
    assert "batch_points" not in base and "selection" not in base
    assert "selection" in noted and "a point" in noted


# 5. configuration switch ----------------------------------------------------------------------


def _config(**semantic: object) -> dict:
    config = pipeline_plan.template("a" * 64, "brief")
    config["stages"]["argument"]["semantic"].update(semantic)
    return config


def test_batching_is_off_by_default_and_the_field_is_optional() -> None:
    config = pipeline_plan.template("a" * 64, "brief")
    assert "argument_batch_tokens" not in config["stages"]["argument"]["semantic"]
    pipeline_plan.validate_config(config, pipeline_plan.registry())
    pipeline_plan.validate_config(_config(argument_batch_tokens=0), pipeline_plan.registry())
    pipeline_plan.validate_config(_config(argument_batch_tokens=2048), pipeline_plan.registry())


@pytest.mark.parametrize("value", [1, 511, 16385, -1, True, "2048", 2048.0, None])
def test_batch_budget_values_outside_the_range_are_refused(value: object) -> None:
    with pytest.raises(PlanError):
        pipeline_plan.validate_config(
            _config(argument_batch_tokens=value), pipeline_plan.registry()
        )


def test_the_batch_field_is_only_allowed_on_the_argument_stage() -> None:
    config = pipeline_plan.template("a" * 64, "brief")
    config["stages"]["audits"]["semantic"]["argument_batch_tokens"] = 2048
    with pytest.raises(PlanError):
        pipeline_plan.validate_config(config, pipeline_plan.registry())


# 6. end to end with injected replies ------------------------------------------------------------


def _producer(seen: list[dict]):
    def run(req: dict[str, object]) -> str:
        context = hybrid.input_payload(req["messages"])  # type: ignore[arg-type]
        if "max_source_links" in context:
            units = [i for i in context["original_source"] if i[1] == "text"]
            seen.append({"kind": "batch", "units": [u[0] for u in units]})
            return json.dumps(
                {
                    "points": ["A point about these units."],
                    "source_links": [
                        {"id": u[0], "role": ROLES[i % 4]} for i, u in enumerate(units[:3])
                    ],
                }
            )
        if "batch_points" in context:
            units = [i for i in context["original_source"] if i[1] == "text"]
            seen.append({"kind": "final", "units": [u[0] for u in units]})
            return json.dumps(
                {
                    "hypothesis": "A revisable reading of the whole chapter.",
                    "questions": ["Does the evidence hold?"],
                    "source_links": [
                        {"id": u[0], "role": ROLES[i % 4]} for i, u in enumerate(units)
                    ],
                }
            )
        seen.append({"kind": "other"})
        return _injected_hybrid_response("argument", req)

    return run


def _drive(tmp_path: Path, tokens, batch_tokens: int | None, seen: list[dict]) -> dict:
    runner, selected, section = _selected_hybrid_source(tmp_path)
    config = pipeline_plan.template(selected.stem, "brief")
    if batch_tokens is not None:
        config["stages"]["argument"]["semantic"]["argument_batch_tokens"] = batch_tokens
    return chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        plan=chapter.ChapterPlan(brief_cpu=True),
        pipeline_config=config,
        argument_completion=_producer(seen),
        audit_completion=lambda r: _injected_hybrid_response("audit", r),
        reconcile_completion=lambda r: _injected_hybrid_response("reconcile", r),
        token_counter=tokens,
    )


def _by_text(messages: list[dict[str, str]]) -> int:
    return 300 + len(messages[1]["content"]) // 4


def test_batching_off_makes_exactly_one_argument_call(tmp_path: Path) -> None:
    seen: list[dict] = []
    result = _drive(tmp_path, _by_text, None, seen)
    assert result["status"] == "BRIEF_READY_UNREVIEWED"
    assert [s["kind"] for s in seen] == ["other"]


def test_batching_on_runs_batches_then_one_small_final_argument_call(tmp_path: Path) -> None:
    seen: list[dict] = []
    result = _drive(tmp_path, _by_text, 700, seen)
    assert result["status"] == "BRIEF_READY_UNREVIEWED"
    kinds = [s["kind"] for s in seen]
    assert kinds.count("batch") >= 2 and kinds[-1] == "final" and kinds.count("final") == 1
    batched = [u for s in seen if s["kind"] == "batch" for u in s["units"]]
    assert len(batched) == len(set(batched))
    final_units = seen[-1]["units"]
    assert 1 <= len(final_units) <= hybrid.MAX_ARGUMENT_LINKS
    assert set(final_units) <= set(batched)


def test_a_smaller_token_budget_makes_more_batches(tmp_path: Path) -> None:
    few: list[dict] = []
    many: list[dict] = []
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _drive(tmp_path / "a", _by_text, 1600, few)
    _drive(tmp_path / "b", _by_text, 600, many)
    assert [s["kind"] for s in many].count("batch") > [s["kind"] for s in few].count("batch")


def test_a_second_run_replays_every_batch_from_cache(tmp_path: Path) -> None:
    runner, selected, section = _selected_hybrid_source(tmp_path)
    config = pipeline_plan.template(selected.stem, "brief")
    config["stages"]["argument"]["semantic"]["argument_batch_tokens"] = 700
    seen: list[dict] = []

    def go() -> dict:
        return chapter.evaluate_hybrid_chapter(
            runner,
            selected,
            section,
            plan=chapter.ChapterPlan(brief_cpu=True),
            pipeline_config=config,
            argument_completion=_producer(seen),
            audit_completion=lambda r: _injected_hybrid_response("audit", r),
            reconcile_completion=lambda r: _injected_hybrid_response("reconcile", r),
            token_counter=_by_text,
        )

    first = go()
    calls = len(seen)
    second = go()
    assert len(seen) == calls
    assert second["brief_identity"] == first["brief_identity"]
    assert second["configured_stage_results"]["argument"] == "REUSED"


def test_resume_after_review_replays_batched_argument_when_the_script_config_matches(
    tmp_path: Path,
) -> None:
    from public_book_fixture_support import _hybrid_creative_output

    runner, selected, section = _selected_hybrid_source(tmp_path)
    brief_config = pipeline_plan.template(selected.stem, "brief")
    brief_config["stages"]["argument"]["semantic"]["argument_batch_tokens"] = 700
    seen: list[dict] = []
    waiting = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        plan=chapter.ChapterPlan(brief_cpu=True),
        pipeline_config=brief_config,
        argument_completion=_producer(seen),
        audit_completion=lambda r: _injected_hybrid_response("audit", r),
        reconcile_completion=lambda r: _injected_hybrid_response("reconcile", r),
        token_counter=_by_text,
    )
    assert [s["kind"] for s in seen].count("batch") >= 2
    base = runner.root / "book-candidates"
    ledger = json.loads(
        next((base / "chapter-evaluations").glob("*/source-ledger.json")).read_bytes()
    )
    brief = json.loads(next((base / "hybrid_reconcile").glob("*/validated.json")).read_bytes())[
        "payload"
    ]
    review = {
        "schema": hybrid.REVIEW_SCHEMA,
        "brief_sha256": brief["brief_sha256"],
        "brief_artifact_identity": waiting["brief_identity"],
        "reviewer": "Synthetic test human",
        "unit_decisions": [
            {"id": e["id"], "review": "accept", "reason": "Original source considered"}
            for e in ledger["entries"]
        ],
        "beat_decisions": [
            {
                "beat_id": b["beat_id"],
                "review": "support_attested",
                "reason": "Synthetic source-support check only",
            }
            for b in brief["beats"]
        ],
    }
    review_file = tmp_path / "review.json"
    review_file.write_text(json.dumps(review), encoding="utf-8")
    script_config = pipeline_plan.template(selected.stem, "script")
    script_config["stages"]["argument"]["semantic"] = dict(
        brief_config["stages"]["argument"]["semantic"]
    )
    for name in ("argument", "audits", "reconcile", "human_review"):
        script_config["stages"][name]["enabled"] = False
    calls = len(seen)

    def creative(kind: str):
        return lambda r: _hybrid_creative_output(kind, r)

    result = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        review_file=review_file,
        expected_brief_identity=waiting["brief_identity"],
        plan=chapter.ChapterPlan(adaptation_cpu=True, script_cpu=True),
        pipeline_config=script_config,
        argument_completion=_producer(seen),
        audit_completion=lambda r: _injected_hybrid_response("audit", r),
        reconcile_completion=lambda r: _injected_hybrid_response("reconcile", r),
        adaptation_completion=creative("adaptation"),
        script_completion=creative("chapter_script"),
        token_counter=_by_text,
    )
    assert len(seen) == calls
    assert result["configured_stage_results"]["argument"] == "REUSED"
    assert result["brief_identity"] == waiting["brief_identity"]


def test_resume_fails_closed_when_the_script_config_drops_the_batch_setting(
    tmp_path: Path,
) -> None:
    from public_book_fixture_support import _hybrid_creative_output

    runner, selected, section = _selected_hybrid_source(tmp_path)
    brief_config = pipeline_plan.template(selected.stem, "brief")
    brief_config["stages"]["argument"]["semantic"]["argument_batch_tokens"] = 700
    seen: list[dict] = []
    waiting = chapter.evaluate_hybrid_chapter(
        runner,
        selected,
        section,
        plan=chapter.ChapterPlan(brief_cpu=True),
        pipeline_config=brief_config,
        argument_completion=_producer(seen),
        audit_completion=lambda r: _injected_hybrid_response("audit", r),
        reconcile_completion=lambda r: _injected_hybrid_response("reconcile", r),
        token_counter=_by_text,
    )
    assert [s["kind"] for s in seen].count("batch") >= 2
    base = runner.root / "book-candidates"
    ledger = json.loads(
        next((base / "chapter-evaluations").glob("*/source-ledger.json")).read_bytes()
    )
    brief = json.loads(next((base / "hybrid_reconcile").glob("*/validated.json")).read_bytes())[
        "payload"
    ]
    review = {
        "schema": hybrid.REVIEW_SCHEMA,
        "brief_sha256": brief["brief_sha256"],
        "brief_artifact_identity": waiting["brief_identity"],
        "reviewer": "Synthetic test human",
        "unit_decisions": [
            {"id": e["id"], "review": "accept", "reason": "Original source considered"}
            for e in ledger["entries"]
        ],
        "beat_decisions": [
            {
                "beat_id": b["beat_id"],
                "review": "support_attested",
                "reason": "Synthetic source-support check only",
            }
            for b in brief["beats"]
        ],
    }
    review_file = tmp_path / "review.json"
    review_file.write_text(json.dumps(review), encoding="utf-8")
    script_config = pipeline_plan.template(selected.stem, "script")
    for name in ("argument", "audits", "reconcile", "human_review"):
        script_config["stages"][name]["enabled"] = False
    calls = len(seen)

    def creative(kind: str):
        return lambda r: _hybrid_creative_output(kind, r)

    with pytest.raises(chapter.ChapterStop):
        chapter.evaluate_hybrid_chapter(
            runner,
            selected,
            section,
            review_file=review_file,
            expected_brief_identity=waiting["brief_identity"],
            plan=chapter.ChapterPlan(adaptation_cpu=True, script_cpu=True),
            pipeline_config=script_config,
            argument_completion=_producer(seen),
            audit_completion=lambda r: _injected_hybrid_response("audit", r),
            reconcile_completion=lambda r: _injected_hybrid_response("reconcile", r),
            adaptation_completion=creative("adaptation"),
            script_completion=creative("chapter_script"),
            token_counter=_by_text,
        )
    assert len(seen) == calls
