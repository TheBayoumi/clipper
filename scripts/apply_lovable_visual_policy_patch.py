from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one replacement anchor, found {count}")
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def insert_before(path: str, anchor: str, block: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if block.strip() in text:
        return
    count = text.count(anchor)
    if count != 1:
        raise RuntimeError(f"{path}: expected one insertion anchor, found {count}")
    target.write_text(text.replace(anchor, block + anchor, 1), encoding="utf-8")


def main() -> None:
    models = "src/clipper/models.py"
    insert_before(
        models,
        "@dataclass(frozen=True, slots=True)\nclass EditorialAcceptancePolicy:",
        '''@dataclass(frozen=True, slots=True)
class VisualPresencePolicy:
    require_any: tuple[str, ...] = ()
    minimum_confidence: float = 0.75

    @classmethod
    def from_dict(cls, value: object) -> VisualPresencePolicy:
        if value is None:
            return cls()
        if not isinstance(value, dict):
            raise BriefValidationError("acceptance_policy.visual_presence must be an object")
        unknown_fields = set(value) - {"require_any", "minimum_confidence"}
        if unknown_fields:
            raise BriefValidationError(
                "unsupported acceptance_policy.visual_presence rule: "
                f"{sorted(unknown_fields)[0]}"
            )
        raw = value.get("require_any", [])
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise BriefValidationError(
                "acceptance_policy.visual_presence.require_any must be a list of strings"
            )
        required = tuple(dict.fromkeys(item.strip() for item in raw if item.strip()))
        confidence = float(value.get("minimum_confidence", 0.75))
        if not 0.0 <= confidence <= 1.0:
            raise BriefValidationError(
                "acceptance_policy.visual_presence.minimum_confidence must be between 0 and 1"
            )
        return cls(require_any=required, minimum_confidence=confidence)


''',
    )
    replace_once(
        models,
        '    branding: BrandingPolicy = field(default_factory=BrandingPolicy)\n    ai_generated_source_video: PolicyAction = "escalate"\n',
        '    branding: BrandingPolicy = field(default_factory=BrandingPolicy)\n    visual_presence: VisualPresencePolicy = field(default_factory=VisualPresencePolicy)\n    ai_generated_source_video: PolicyAction = "escalate"\n',
    )
    replace_once(
        models,
        '            "branding",\n            "generated_media",\n',
        '            "branding",\n            "visual_presence",\n            "generated_media",\n',
    )
    replace_once(
        models,
        '            branding=BrandingPolicy.from_dict(value.get("branding")),\n            ai_generated_source_video=_policy_action(\n',
        '            branding=BrandingPolicy.from_dict(value.get("branding")),\n            visual_presence=VisualPresencePolicy.from_dict(value.get("visual_presence")),\n            ai_generated_source_video=_policy_action(\n',
    )
    replace_once(
        models,
        '            "branding": asdict(self.branding),\n            "generated_media": {\n',
        '            "branding": asdict(self.branding),\n            "visual_presence": {\n                "require_any": list(self.visual_presence.require_any),\n                "minimum_confidence": self.visual_presence.minimum_confidence,\n            },\n            "generated_media": {\n',
    )

    integrity = "src/clipper/editorial_integrity.py"
    replace_once(
        integrity,
        'from .models import AcceptancePolicy, CampaignBrief\nfrom .stage_contracts import structural_contract_fingerprint\n',
        'from .models import AcceptancePolicy, CampaignBrief\nfrom .multimodal_timeline import MultimodalTimeline\nfrom .stage_contracts import structural_contract_fingerprint\n',
    )
    insert_before(
        integrity,
        "def evaluate_campaign_policy(\n",
        '''def _normalize_visual_presence(value: str) -> str:
    normalized = "".join(
        character.casefold() if character.isalnum() else " " for character in value
    )
    return " ".join(normalized.split())


def _required_visual_presence_gate(
    brief: CampaignBrief,
    source_start: float,
    source_end: float,
    multimodal: MultimodalTimeline | None,
) -> tuple[GateDecision, tuple[str, ...], dict[str, object]]:
    visual_policy = brief.acceptance_policy.visual_presence
    required = tuple(item for item in visual_policy.require_any if item.strip())
    if not required:
        return GateDecision.PASS, (), {"enabled": False, "require_any": []}
    checks: dict[str, object] = {
        "enabled": True,
        "require_any": list(required),
        "minimum_confidence": visual_policy.minimum_confidence,
        "matched_terms": [],
        "matched_evidence": [],
    }
    if multimodal is None:
        checks["visual_evidence_available"] = False
        return GateDecision.ESCALATE, ("required_visual_presence_uncertain",), checks

    required_normalized = {item: _normalize_visual_presence(item) for item in required}
    overlapping = tuple(
        event
        for event in multimodal.events
        if event.end > source_start and event.start < source_end and event.visual_salience > 0
    )
    matches: list[dict[str, object]] = []
    low_confidence_matches: list[dict[str, object]] = []
    for event in overlapping:
        evidence_values = tuple(
            dict.fromkeys(
                (
                    *event.visible_people,
                    *event.branding,
                    *event.ocr_text,
                    *event.visual_summaries,
                )
            )
        )
        for evidence in evidence_values:
            normalized_evidence = _normalize_visual_presence(evidence)
            for required_text, normalized_required in required_normalized.items():
                if not normalized_required or normalized_required not in normalized_evidence:
                    continue
                item = {
                    "required": required_text,
                    "evidence": evidence,
                    "start": event.start,
                    "end": event.end,
                    "confidence": event.visual_salience,
                }
                if event.visual_salience >= visual_policy.minimum_confidence:
                    matches.append(item)
                else:
                    low_confidence_matches.append(item)

    if matches:
        checks["matched_terms"] = list(
            dict.fromkeys(str(item["required"]) for item in matches)
        )
        checks["matched_evidence"] = matches
        checks["visual_evidence_available"] = True
        return GateDecision.PASS, (), checks

    spans = sorted(
        (
            max(source_start, span.start),
            min(source_end, span.end),
        )
        for span in multimodal.visual_evidence_spans
        if span.scope == "source_policy" and span.end > source_start and span.start < source_end
    )
    merged: list[tuple[float, float]] = []
    for start, end in spans:
        if end <= start:
            continue
        if merged and start <= merged[-1][1] + 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    covered = sum(end - start for start, end in merged)
    duration = source_end - source_start
    coverage_complete = covered + 0.05 >= duration
    high_confidence_events = sum(
        1
        for event in overlapping
        if event.visual_salience >= visual_policy.minimum_confidence
    )
    checks.update(
        {
            "visual_evidence_available": bool(overlapping),
            "source_policy_coverage_seconds": round(covered, 6),
            "source_policy_coverage_complete": coverage_complete,
            "high_confidence_visual_events": high_confidence_events,
            "low_confidence_matches": low_confidence_matches,
        }
    )
    if low_confidence_matches or not coverage_complete or high_confidence_events == 0:
        return GateDecision.ESCALATE, ("required_visual_presence_uncertain",), checks
    return GateDecision.REJECT, ("required_visual_presence_missing",), checks


''',
    )
    replace_once(
        integrity,
        '    branding: tuple[BrandingEvidence, ...],\n) -> PolicyAudit:\n',
        '    branding: tuple[BrandingEvidence, ...],\n    multimodal: MultimodalTimeline | None = None,\n) -> PolicyAudit:\n',
    )
    replace_once(
        integrity,
        "    if GateDecision.REJECT in decisions:\n",
        '''    presence_decision, presence_reasons, presence_checks = _required_visual_presence_gate(
        brief, source_start, source_end, multimodal
    )
    if presence_decision != GateDecision.PASS:
        decisions.append(presence_decision)
        reasons.extend(presence_reasons)

    if GateDecision.REJECT in decisions:
''',
    )
    replace_once(
        integrity,
        '            "branding_policy": asdict(policy.branding),\n            "generated_media_policy": policy.ai_generated_source_video,\n',
        '            "branding_policy": asdict(policy.branding),\n            "visual_presence_policy": presence_checks,\n            "generated_media_policy": policy.ai_generated_source_video,\n',
    )

    quality_pipeline = "src/clipper/quality_pipeline.py"
    replace_once(
        quality_pipeline,
        '    branding: tuple[BrandingEvidence, ...],\n) -> AdaptedQualityMoment | None:\n',
        '    branding: tuple[BrandingEvidence, ...],\n    multimodal: MultimodalTimeline | None = None,\n) -> AdaptedQualityMoment | None:\n',
    )
    replace_once(
        quality_pipeline,
        '        hazards,\n        branding,\n    )\n',
        '        hazards,\n        branding,\n        multimodal=multimodal,\n    )\n',
    )

    quality_batch = "src/clipper/quality_batch.py"
    replace_once(
        quality_batch,
        '    return policy.enabled and policy.branding.foreign_logos != "allow"\n',
        '    return policy.enabled and (\n        policy.branding.foreign_logos != "allow" or bool(policy.visual_presence.require_any)\n    )\n',
    )
    replace_once(
        quality_batch,
        '                    hazards=hazards,\n                    branding=branded,\n                )\n',
        '                    hazards=hazards,\n                    branding=branded,\n                    multimodal=multimodal,\n                )\n',
    )

    visual_ai = "src/clipper/visual_ai.py"
    replace_once(
        visual_ai,
        '    recovery_attempt: int,\n    generation_capacity: dict[str, object] | None = None,\n) -> dict[str, object]:\n',
        '    recovery_attempt: int,\n    generation_capacity: dict[str, object] | None = None,\n    required_visual_entities: tuple[str, ...] = (),\n) -> dict[str, object]:\n',
    )
    replace_once(
        visual_ai,
        '    context: dict[str, object] = {\n        "video_id": video_id,\n',
        '''    if required_visual_entities:
        instruction += (
            " Campaign-required visual entities are supplied in required_visual_entities. "
            "When one is visibly evidenced, preserve the exact supplied term in "
            "visible_speakers for a person or in a branding:/ocr: event label as appropriate. "
            "Never infer a person or brand from audio, transcript, filename, or campaign context."
        )
    context: dict[str, object] = {
        "video_id": video_id,
''',
    )
    replace_once(
        visual_ai,
        '        "source_policy_recovery_attempt": recovery_attempt,\n        "instruction": instruction,\n    }\n',
        '        "source_policy_recovery_attempt": recovery_attempt,\n        "required_visual_entities": list(required_visual_entities),\n        "instruction": instruction,\n    }\n',
    )
    replace_once(
        visual_ai,
        '    requested_identity: ModelIdentity,\n) -> str:\n',
        '    requested_identity: ModelIdentity,\n    required_visual_entities: tuple[str, ...] = (),\n) -> str:\n',
    )
    replace_once(
        visual_ai,
        '            recovery_attempt=0,\n        )["instruction"]\n',
        '            recovery_attempt=0,\n            required_visual_entities=required_visual_entities,\n        )["instruction"]\n',
    )
    replace_once(
        visual_ai,
        '            "inspection_contract": instruction,\n            "frame_contract": {"max_edge": VISUAL_SAMPLE_MAX_EDGE},\n',
        '            "inspection_contract": instruction,\n            "required_visual_entities": list(required_visual_entities),\n            "frame_contract": {"max_edge": VISUAL_SAMPLE_MAX_EDGE},\n',
    )
    replace_once(
        visual_ai,
        '    generation_capacity: dict[str, object] | None = None,\n) -> tuple[tuple[VisualEvent, ...], list[ProviderResult[dict[str, Any]]]]:\n',
        '    generation_capacity: dict[str, object] | None = None,\n    required_visual_entities: tuple[str, ...] = (),\n) -> tuple[tuple[VisualEvent, ...], list[ProviderResult[dict[str, Any]]]]:\n',
    )
    replace_once(
        visual_ai,
        '                recovery_attempt=recovery_attempt,\n                generation_capacity=generation_capacity,\n            ),\n',
        '                recovery_attempt=recovery_attempt,\n                generation_capacity=generation_capacity,\n                required_visual_entities=required_visual_entities,\n            ),\n',
    )
    replace_once(
        visual_ai,
        '    checkpoint_dir: Path | None = None,\n    checkpoint_commit: Callable[[], None] | None = None,\n) -> tuple[VisualTimeline, ProviderResult[dict[str, Any]]]:\n',
        '    checkpoint_dir: Path | None = None,\n    checkpoint_commit: Callable[[], None] | None = None,\n    required_visual_entities: tuple[str, ...] = (),\n) -> tuple[VisualTimeline, ProviderResult[dict[str, Any]]]:\n',
    )
    replace_once(
        visual_ai,
        '        requested_identity=requested_identity,\n    )\n',
        '        requested_identity=requested_identity,\n        required_visual_entities=required_visual_entities,\n    )\n',
    )
    replace_once(
        visual_ai,
        '                    generation_capacity=(\n                        {"observed_output_tokens_per_item": observed_output_tokens_per_item}\n                        if observed_output_tokens_per_item is not None\n                        else None\n                    ),\n                )\n',
        '                    generation_capacity=(\n                        {"observed_output_tokens_per_item": observed_output_tokens_per_item}\n                        if observed_output_tokens_per_item is not None\n                        else None\n                    ),\n                    required_visual_entities=required_visual_entities,\n                )\n',
    )

    pipeline = "src/clipper/pipeline.py"
    replace_once(
        pipeline,
        'downloaded = gdown.download(url=url, output=str(temporary), quiet=True)  # type: ignore[attr-defined]',
        'downloaded = gdown.download(url=url, output=str(temporary), quiet=True)',
    )
    replace_once(
        pipeline,
        '    provider: VisionProvider,\n    run_dir: Path,\n) -> tuple[VisualTimeline, dict[str, object]]:\n',
        '    provider: VisionProvider,\n    run_dir: Path,\n    required_visual_entities: tuple[str, ...] = (),\n) -> tuple[VisualTimeline, dict[str, object]]:\n',
    )
    replace_once(
        pipeline,
        '        checkpoint_dir=_VISUAL_CHECKPOINT_DIR.get(),\n        checkpoint_commit=_VISUAL_CHECKPOINT_COMMIT.get(),\n    )\n',
        '        checkpoint_dir=_VISUAL_CHECKPOINT_DIR.get(),\n        checkpoint_commit=_VISUAL_CHECKPOINT_COMMIT.get(),\n        required_visual_entities=required_visual_entities,\n    )\n',
    )
    replace_once(
        pipeline,
        '                    scout,\n                    run_dir,\n                )\n',
        '                    scout,\n                    run_dir,\n                    required_visual_entities=brief.acceptance_policy.visual_presence.require_any,\n                )\n',
    )

    Path("tests/test_visual_presence_policy.py").write_text(
        '''from __future__ import annotations

from clipper.editorial_integrity import (
    GateDecision,
    HazardClassification,
    SourceHazardSegment,
    evaluate_campaign_policy,
)
from clipper.models import AcceptancePolicy, CampaignBrief
from clipper.multimodal_timeline import MultimodalEvent, MultimodalTimeline
from clipper.visual import VisualEvidenceSpan


def _brief(*required: str, minimum_confidence: float = 0.75) -> CampaignBrief:
    policy = AcceptancePolicy.from_dict(
        {
            "enabled": True,
            "source_segments": {
                "allow": ["editorial_content"],
                "forbid": [],
                "unknown": "escalate",
            },
            "branding": {"foreign_logos": "allow"},
            "visual_presence": {
                "require_any": list(required),
                "minimum_confidence": minimum_confidence,
            },
        }
    )
    return CampaignBrief(
        campaign_id="lovable-test",
        title="Lovable test",
        objective="Test required visual presence",
        allowed_video_ids=["video"],
        acceptance_policy=policy,
    )


def _hazards() -> tuple[SourceHazardSegment, ...]:
    return (
        SourceHazardSegment(
            start=0.0,
            end=10.0,
            classification=HazardClassification.EDITORIAL_CONTENT,
            confidence=0.99,
            evidence=("editorial",),
            model_identity={"model": "test"},
        ),
    )


def _timeline(
    *,
    visible_people: tuple[str, ...] = (),
    branding: tuple[str, ...] = (),
    ocr_text: tuple[str, ...] = (),
    confidence: float = 0.95,
    covered: bool = True,
) -> MultimodalTimeline:
    spans = (
        (VisualEvidenceSpan(0.0, 10.0, 5.0, "source_policy"),)
        if covered
        else (VisualEvidenceSpan(0.0, 4.0, 2.0, "source_policy"),)
    )
    return MultimodalTimeline(
        video_id="video",
        source_hash="a" * 64,
        duration=10.0,
        events=(
            MultimodalEvent(
                start=0.0,
                end=10.0,
                visible_people=visible_people,
                branding=branding,
                ocr_text=ocr_text,
                visual_summaries=("inspected source frame",),
                visual_salience=confidence,
                confidence=confidence,
            ),
        ),
        visual_evidence_spans=spans,
    )


def test_visual_presence_policy_parses_and_serializes() -> None:
    policy = AcceptancePolicy.from_dict(
        {"visual_presence": {"require_any": ["Anton", "Lovable"], "minimum_confidence": 0.8}}
    )
    assert policy.visual_presence.require_any == ("Anton", "Lovable")
    assert policy.to_dict()["visual_presence"] == {
        "require_any": ["Anton", "Lovable"],
        "minimum_confidence": 0.8,
    }


def test_required_visual_presence_passes_on_visible_person() -> None:
    audit = evaluate_campaign_policy(
        _brief("Anton", "Lovable"),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(visible_people=("Anton Osika",)),
    )
    assert audit.decision == GateDecision.PASS
    checks = audit.campaign_policy_checks["visual_presence_policy"]
    assert checks["matched_terms"] == ["Anton"]


def test_required_visual_presence_passes_on_branding() -> None:
    audit = evaluate_campaign_policy(
        _brief("Anton", "Lovable"),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(branding=("Lovable",)),
    )
    assert audit.decision == GateDecision.PASS


def test_required_visual_presence_rejects_when_absent_with_full_coverage() -> None:
    audit = evaluate_campaign_policy(
        _brief("Anton", "Lovable"),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(visible_people=("Ben",)),
    )
    assert audit.decision == GateDecision.REJECT
    assert "required_visual_presence_missing" in audit.reasons


def test_required_visual_presence_escalates_low_confidence_match() -> None:
    audit = evaluate_campaign_policy(
        _brief("Anton", "Lovable", minimum_confidence=0.8),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(branding=("Lovable",), confidence=0.6),
    )
    assert audit.decision == GateDecision.ESCALATE
    assert "required_visual_presence_uncertain" in audit.reasons


def test_required_visual_presence_escalates_incomplete_visual_coverage() -> None:
    audit = evaluate_campaign_policy(
        _brief("Anton", "Lovable"),
        0.0,
        10.0,
        _hazards(),
        (),
        multimodal=_timeline(visible_people=("Ben",), covered=False),
    )
    assert audit.decision == GateDecision.ESCALATE
    assert "required_visual_presence_uncertain" in audit.reasons
''',
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
