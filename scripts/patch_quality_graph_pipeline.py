from pathlib import Path

path = Path("src/clipper/pipeline.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    "from .autonomous_editor import AutonomousEditorialPlanner, OpenVideoAnalysis\n",
    "from .autonomous_editor import AutonomousEditorialPlanner\n",
)
anchor = "from .qc import run_technical_qc\n"
replacement = anchor + "from .quality_batch import plan_quality_batch\n"
if "from .quality_batch import plan_quality_batch\n" not in text:
    if anchor not in text:
        raise SystemExit("quality_batch import anchor missing")
    text = text.replace(anchor, replacement, 1)

text = text.replace("    open_analyses: list[OpenVideoAnalysis] = []\n", "")

old_source = '''            telemetry.start(f"editorial_analysis:{video.video_id}")
            if open_planner is not None:
                analysis = open_planner.analyze_video(brief, canonical, visual_timeline)
                open_analyses.append(analysis)
                moments = analysis.moments
                concepts = analysis.concepts
                source_rejections = analysis.rejections
                source_stats = {
                    "candidate_starts": 0,
                    "eligible_endpoints": 0,
                    "concepts_after_quality": len(concepts),
                    "concepts_after_moment_dedup": len(moments),
                    "semantic_representatives": len(concepts),
                }
            else:
'''
new_source = '''            telemetry.start(f"editorial_analysis:{video.video_id}")
            if open_planner is not None:
                # Open production defers semantic discovery to the authoritative quality graph
                # after every source has canonical + visual evidence. No legacy concept planning
                # is allowed to determine production yield.
                moments = []
                concepts = []
                source_rejections = []
                source_stats = {
                    "candidate_starts": 0,
                    "eligible_endpoints": 0,
                    "concepts_after_quality": 0,
                    "concepts_after_moment_dedup": 0,
                    "semantic_representatives": 0,
                }
            else:
'''
if old_source not in text:
    raise SystemExit("open source-analysis anchor missing")
text = text.replace(old_source, new_source, 1)

text = text.replace(
    "    if open_planner is not None and not open_analyses and manifest.errors:\n",
    "    if open_planner is not None and not canonical_timelines and manifest.errors:\n",
    1,
)
text = text.replace(
    '        reason = f"all authorized source analyses failed: {source_errors}"\n',
    '        reason = f"all authorized source processing failed: {source_errors}"\n',
    1,
)

start_marker = "    if open_planner is not None:\n        open_batch = open_planner.plan_batch"
end_marker = "    else:\n        selected_concepts = select_distinct_concepts("
start = text.find(start_marker)
if start < 0:
    raise SystemExit("legacy open plan_batch start marker missing")
end = text.find(end_marker, start)
if end < 0:
    raise SystemExit("legacy open plan_batch end marker missing")
new_open = '''    if open_planner is not None:
        if editorial_provider is None:
            raise RuntimeError("open editorial provider is unavailable for quality graph planning")
        quality_batch = plan_quality_batch(
            brief,
            canonical_timelines,
            visual_timelines,
            editorial_provider,
            dag_root=run_dir / "dag",
            max_words_per_chunk=cfg.editorial_chunk_words,
            chunk_overlap_words=cfg.editorial_chunk_overlap_words,
            progress_callback=_model_progress,
        )
        all_moments = list(quality_batch.story_moments)
        all_concepts = list(quality_batch.concepts)
        selected_concepts = list(quality_batch.concepts)
        variants = list(quality_batch.variants)
        plans = list(quality_batch.plans)
        manifest.rejections.extend(quality_batch.rejections)
        manifest.run_metadata["editorial_inference"]["model_invocations"] = list(
            quality_batch.model_invocations
        )
        manifest.run_metadata["pre_render_boundary_audits"] = list(
            quality_batch.boundary_audits
        )
        manifest.run_metadata["pre_render_campaign_policy_audits"] = list(
            quality_batch.campaign_policy_audits
        )
        manifest.run_metadata["source_hazards"] = list(quality_batch.source_hazards)
        manifest.run_metadata["quality_planning"] = {
            "architecture": "semantic-core-envelope-window-quality-moment",
            "yield_is_quota_independent": True,
            "eligible_quality_moments": len(quality_batch.quality_moments),
            "stage_cache_hits": quality_batch.stage_cache_hits,
            "stage_executions": quality_batch.stage_executions,
            "source_evidence": quality_batch.source_evidence,
        }
        for invocation in quality_batch.model_invocations:
            compute_budget.record_mapping(invocation.get("usage"))
        quality_count = len(quality_batch.quality_moments)
        mining_stats.update(
            {
                "candidate_starts": quality_count,
                "eligible_endpoints": quality_count,
                "concepts_after_quality": quality_count,
                "concepts_after_moment_dedup": quality_count,
                "semantic_representatives": quality_count,
            }
        )
        _write_json(
            run_dir / "open-model" / "model-invocations.json",
            list(quality_batch.model_invocations),
        )
        _write_json(
            run_dir / "open-model" / "discovered-concepts.json",
            [item.to_dict() for item in all_concepts],
        )
        _write_json(
            run_dir / "open-model" / "boundary-audits.json",
            list(quality_batch.boundary_audits),
        )
        _write_json(
            run_dir / "open-model" / "campaign-policy-audits.json",
            list(quality_batch.campaign_policy_audits),
        )
        _write_json(
            run_dir / "open-model" / "source-hazards.json",
            list(quality_batch.source_hazards),
        )
        _write_json(run_dir / "quality-graph" / "batch.json", quality_batch.to_dict())
        for video_id, evidence in quality_batch.source_evidence.items():
            _write_json(run_dir / "quality-graph" / f"{video_id}.json", evidence)
'''
text = text[:start] + new_open + text[end:]

old_group = '''    quality_groups = group_quality_plans(plans)
    primary_plans = [group.primary for group in quality_groups]
'''
new_group = '''    quality_groups = group_quality_plans(plans)
    if open_planner is not None and len(quality_groups) != len(plans):
        raise RuntimeError(
            "authoritative quality graph must compile to exactly one render plan per quality moment"
        )
    primary_plans = [group.primary for group in quality_groups]
'''
if old_group not in text:
    raise SystemExit("quality group anchor missing")
text = text.replace(old_group, new_group, 1)

if "open_planner.analyze_video(" in text or "open_planner.plan_batch(" in text:
    raise SystemExit("legacy open planner remains authoritative")
path.write_text(text, encoding="utf-8")
