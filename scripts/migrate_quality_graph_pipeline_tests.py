from pathlib import Path

path = Path("tests/test_pipeline.py")
text = path.read_text(encoding="utf-8")

if not text.startswith("import hashlib\n"):
    text = "import hashlib\n" + text

marker = '''        if task == "episode_editorial_profile":
'''
new_cases = '''        if task.startswith("semantic_cores:"):
            words = payload["words"]
            assert isinstance(words, list) and words
            refs = [item["word_ref"] for item in words if isinstance(item, dict)]
            value: dict[str, object] = {
                "cores": [
                    {
                        "core_id": "provider-core-id-is-not-authoritative",
                        "start_word_id": refs[0],
                        "end_word_id": refs[-1],
                        "semantic_summary": "A complete explanation of saving time",
                        "editorial_reason": "The source idea is independently worthwhile",
                        "confidence": 0.95,
                    }
                ]
            }
        elif task.startswith("narrative_envelope:"):
            core = payload["core"]
            words = payload["source_context_words"]
            assert isinstance(core, dict)
            assert isinstance(words, list) and words
            refs = [item["word_ref"] for item in words if isinstance(item, dict)]
            value = {
                "envelope_id": "provider-envelope-id-is-not-authoritative",
                "core_id": core["core_id"],
                "start_word_id": refs[0],
                "end_word_id": refs[-1],
                "required_prior_context": "",
                "required_followup_context": "",
                "setup_resolved": True,
                "payoff_resolved": True,
                "reference_resolution": [],
                "confidence": 0.95,
            }
        elif task.startswith("quality_windows:"):
            core = payload["core"]
            windows = payload["feasible_windows"]
            assert isinstance(core, dict)
            assert isinstance(windows, list) and windows and isinstance(windows[0], dict)
            value = {
                "core_id": core["core_id"],
                "selected_window_id": windows[0]["window_id"],
                "decision": "PASS",
                "quality_score": 0.95,
                "rationale": "Complete, specific, and worth publishing",
                "confidence": 0.95,
            }
        elif task == "episode_editorial_profile":
'''
if 'if task.startswith("semantic_cores:"):' not in text:
    if marker not in text:
        raise SystemExit("FakeOpenEditorialProvider task marker missing")
    text = text.replace(marker, new_cases, 1)

old_calls = '''    assert editorial.calls == [
        "episode_editorial_profile",
        "story_moments:0",
        "clip_concepts",
        "global_concept_comparison",
        "hook_variants:concept-1",
        "edit_plans:concept-1",
        "boundary_audit:plan-1",
    ]
    assert embedding.calls == 1
'''
new_calls = '''    assert len(editorial.calls) == 3
    assert editorial.calls[0] == "semantic_cores:0"
    assert editorial.calls[1].startswith("narrative_envelope:core-")
    assert editorial.calls[2].startswith("quality_windows:core-")
    assert embedding.calls == 0
'''
if old_calls in text:
    text = text.replace(old_calls, new_calls, 1)
elif new_calls not in text:
    raise SystemExit("open-pipeline call assertion marker missing")

old_hash = '''    visual = VisualTimeline(
        "allowed",
        "visual-hash",
'''
new_hash = '''    visual = VisualTimeline(
        "allowed",
        hashlib.sha256(media.read_bytes()).hexdigest(),
'''
if old_hash in text:
    text = text.replace(old_hash, new_hash, 1)
elif new_hash not in text:
    raise SystemExit("visual source hash marker missing")

old_visual_assertions = '''    profile_visual = editorial.payloads["episode_editorial_profile"]["visual_evidence"]
    story_visual = editorial.payloads["story_moments:0"]["visual_evidence"]
    assert isinstance(profile_visual, list) and profile_visual[0]["scene_id"] == "scene-1"
    assert isinstance(story_visual, list) and story_visual[0]["event_labels"] == ["demonstration"]
'''
new_visual_assertions = '''    semantic_visual = editorial.payloads["semantic_cores:0"]["multimodal_evidence"]
    assert isinstance(semantic_visual, list)
    assert any("scene-1" in item["scene_ids"] for item in semantic_visual)
    assert any(
        "The guest visibly demonstrates an object while speaking." in item["visual_summaries"]
        for item in semantic_visual
    )
'''
if old_visual_assertions in text:
    text = text.replace(old_visual_assertions, new_visual_assertions, 1)
elif new_visual_assertions not in text:
    raise SystemExit("visual assertion marker missing")

path.write_text(text, encoding="utf-8")
