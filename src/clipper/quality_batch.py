from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .autonomous_quality_planner import AutonomousQualityPlanner
from .canonical import CanonicalTimeline
from .dag import DagStore
from .modality_profile import infer_source_modality_profile
from .models import CampaignBrief, ClipConcept, EditPlan, HookVariant, StoryMoment
from .multimodal_timeline import build_multimodal_timeline
from .providers.base import EditorialProvider, ProviderResult
from .quality_moments import QualityMoment
from .quality_pipeline import (
    AdaptedQualityMoment,
    adapt_quality_moment,
    forbidden_spans_for_campaign,
    source_branding_evidence,
)
from .source_hazards import SourceHazardClassifier, campaign_context
from .visual import VisualTimeline

ProgressCallback = Callable[[str, str], None]


class RecordingEditorialProvider:
    """Record only real inference calls; content-addressed DAG cache hits cost nothing."""

    def __init__(
        self,
        provider: EditorialProvider,
        *,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        self.provider = provider
        self.identity = provider.identity
        self.progress_callback = progress_callback
        self.invocations: list[dict[str, object]] = []

    def complete_json(
        self,
        *,
        task: str,
        payload: dict[str, Any],
    ) -> ProviderResult[dict[str, Any]]:
        if self.progress_callback is not None:
            self.progress_callback(task, "running")
        try:
            result = self.provider.complete_json(task=task, payload=payload)
        except Exception:
            if self.progress_callback is not None:
                self.progress_callback(task, "failed")
            raise
        if self.progress_callback is not None:
            self.progress_callback(task, "success")
        self.invocations.append(
            {
                "stage": task,
                "cache_hit": False,
                "model": result.model.to_dict(),
                "usage": asdict(result.usage),
                "degraded": result.degraded,
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class QualityBatchResult:
    story_moments: tuple[StoryMoment, ...]
    concepts: tuple[ClipConcept, ...]
    variants: tuple[HookVariant, ...]
    plans: tuple[EditPlan, ...]
    quality_moments: tuple[QualityMoment, ...]
    rejections: tuple[dict[str, object], ...]
    model_invocations: tuple[dict[str, object], ...]
    boundary_audits: tuple[dict[str, object], ...]
    campaign_policy_audits: tuple[dict[str, object], ...]
    source_hazards: tuple[dict[str, object], ...]
    source_evidence: dict[str, dict[str, object]]
    stage_cache_hits: int
    stage_executions: int
    stage_dag_root: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "story_moments": [item.to_dict() for item in self.story_moments],
            "concepts": [item.to_dict() for item in self.concepts],
            "variants": [item.to_dict() for item in self.variants],
            "plans": [item.to_dict() for item in self.plans],
            "quality_moments": [item.to_dict() for item in self.quality_moments],
            "rejections": list(self.rejections),
            "model_invocations": list(self.model_invocations),
            "boundary_audits": list(self.boundary_audits),
            "campaign_policy_audits": list(self.campaign_policy_audits),
            "source_hazards": list(self.source_hazards),
            "source_evidence": self.source_evidence,
            "stage_cache_hits": self.stage_cache_hits,
            "stage_executions": self.stage_executions,
            "stage_dag_root": str(self.stage_dag_root),
        }


def _story_moment(adapted: AdaptedQualityMoment) -> StoryMoment:
    concept = adapted.concept
    return StoryMoment(
        moment_id=adapted.quality_moment.quality_moment_id,
        video_id=concept.video_id,
        start=concept.source_start,
        end=concept.source_end,
        text=concept.text,
        moment_type="quality_moment",
        topic=concept.topic,
        setup=concept.setup,
        payoff=concept.payoff,
        scores=concept.scores,
        score=concept.score,
        transcript_fingerprint=concept.transcript_fingerprint,
    )


def _requires_source_visual_policy(brief: CampaignBrief) -> bool:
    policy = brief.acceptance_policy
    return policy.enabled and policy.branding.foreign_logos != "allow"


def plan_quality_batch(
    brief: CampaignBrief,
    timelines: dict[str, CanonicalTimeline],
    visual_timelines: dict[str, VisualTimeline],
    editorial: EditorialProvider,
    *,
    dag_root: str | Path,
    progress_callback: ProgressCallback | None = None,
) -> QualityBatchResult:
    """Plan evidence-derived quality yield without any clip-count or render-budget quota."""

    recorder = RecordingEditorialProvider(editorial, progress_callback=progress_callback)
    root = Path(dag_root)
    story_moments: list[StoryMoment] = []
    concepts: list[ClipConcept] = []
    variants: list[HookVariant] = []
    plans: list[EditPlan] = []
    quality_moments: list[QualityMoment] = []
    rejections: list[dict[str, object]] = []
    boundary_audits: list[dict[str, object]] = []
    campaign_policy_audits: list[dict[str, object]] = []
    source_hazards: list[dict[str, object]] = []
    source_evidence: dict[str, dict[str, object]] = {}
    cache_hits = 0
    executions = 0
    successful_sources = 0
    source_failures: list[dict[str, object]] = []

    for video_id in sorted(timelines):
        timeline = timelines[video_id]
        visual = visual_timelines.get(video_id)
        try:
            if _requires_source_visual_policy(brief) and visual is None:
                raise RuntimeError(
                    "campaign branding policy requires source visual evidence but visual scouting "
                    "is unavailable"
                )
            multimodal = build_multimodal_timeline(timeline, visual)
            modality_profile = infer_source_modality_profile(multimodal)
            if (
                _requires_source_visual_policy(brief)
                and modality_profile.source_policy_visual_coverage < 0.5
            ):
                raise RuntimeError(
                    "campaign branding policy requires broader source visual evidence coverage: "
                    f"{modality_profile.source_policy_visual_coverage:.3f}"
                )

            hazard_classifier = SourceHazardClassifier(
                recorder,
                DagStore(root / video_id / "source-hazards"),
            )
            hazard_result = hazard_classifier.classify(
                brief,
                timeline,
                multimodal=multimodal,
            )
            rejections.extend(hazard_result.rejections)
            hazards = hazard_result.hazards
            branded = source_branding_evidence(multimodal)
            forbidden = forbidden_spans_for_campaign(brief, hazards, branded)

            planner = AutonomousQualityPlanner(
                recorder,
                DagStore(root / video_id / "quality"),
            )
            planning = planner.plan(
                timeline,
                multimodal=multimodal,
                modality_profile=modality_profile,
                campaign_context=campaign_context(brief),
                relevant_policy=brief.acceptance_policy.to_dict(),
                min_seconds=brief.min_clip_seconds,
                max_seconds=brief.max_clip_seconds,
                forbidden_spans=forbidden,
            )
            successful_sources += 1
            rejections.extend(planning.rejections)
            adapted_ids: list[str] = []
            for moment in planning.quality_moments:
                adapted = adapt_quality_moment(
                    brief,
                    timeline,
                    moment,
                    hazards=hazards,
                    branding=branded,
                )
                if adapted is None:
                    rejections.append(
                        {
                            "video_id": video_id,
                            "quality_moment_id": moment.quality_moment_id,
                            "stage": "quality_moment_pre_render_eligibility",
                            "decision": "REJECT",
                            "reasons": ["quality_moment_failed_campaign_or_boundary_policy"],
                        }
                    )
                    continue
                adapted_ids.append(moment.quality_moment_id)
                quality_moments.append(moment)
                story_moments.append(_story_moment(adapted))
                concepts.append(adapted.concept)
                variants.append(adapted.variant)
                plans.append(adapted.plan)
                boundary_audits.append(
                    {
                        "video_id": video_id,
                        "quality_moment_id": moment.quality_moment_id,
                        **adapted.boundary_audit.to_dict(brief.acceptance_policy),
                    }
                )
                campaign_policy_audits.append(
                    {
                        "video_id": video_id,
                        "quality_moment_id": moment.quality_moment_id,
                        **adapted.policy_audit.to_dict(),
                    }
                )

            source_hazards.extend({"video_id": video_id, **hazard.to_dict()} for hazard in hazards)
            source_cache_hits = hazard_result.stage_cache_hits + planning.stage_cache_hits
            source_executions = hazard_result.stage_executions + planning.stage_executions
            cache_hits += source_cache_hits
            executions += source_executions
            source_evidence[video_id] = {
                "status": "PASS",
                "multimodal_timeline": multimodal.to_dict(),
                "modality_profile": modality_profile.to_dict(),
                "visual_policy_coverage": (
                    visual.coverage_summary("source_policy", duration=multimodal.duration)
                    if visual is not None
                    else None
                ),
                "source_hazards": hazard_result.to_dict(),
                "forbidden_spans": [{"start": item.start, "end": item.end} for item in forbidden],
                "quality_planning": planning.to_dict(),
                "adapted_quality_moment_ids": adapted_ids,
                "semantic_cores": len(planning.cores),
                "stage_cache_hits": source_cache_hits,
                "stage_executions": source_executions,
            }
        except Exception as exc:
            failure: dict[str, object] = {
                "video_id": video_id,
                "stage": "quality_graph_planning",
                "decision": "FAILED",
                "reasons": ["quality_graph_source_failed"],
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            source_failures.append(failure)
            rejections.append(failure)
            source_evidence[video_id] = {"status": "FAILED", **failure}

    if source_failures:
        diagnostics = "; ".join(
            f"{item['video_id']}: {item['error_type']}: {item['error']}" for item in source_failures
        )
        raise RuntimeError(f"quality graph planning failed for explicit source(s): {diagnostics}")

    if len({item.concept_id for item in concepts}) != len(concepts):
        raise RuntimeError("quality graph produced duplicate compatibility concept identities")
    if len({item.plan_id for item in plans}) != len(plans):
        raise RuntimeError("quality graph produced duplicate compatibility plan identities")

    return QualityBatchResult(
        story_moments=tuple(story_moments),
        concepts=tuple(concepts),
        variants=tuple(variants),
        plans=tuple(plans),
        quality_moments=tuple(quality_moments),
        rejections=tuple(rejections),
        model_invocations=tuple(recorder.invocations),
        boundary_audits=tuple(boundary_audits),
        campaign_policy_audits=tuple(campaign_policy_audits),
        source_hazards=tuple(source_hazards),
        source_evidence=source_evidence,
        stage_cache_hits=cache_hits,
        stage_executions=executions,
        stage_dag_root=root,
    )
