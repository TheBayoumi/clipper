# CLIPPER AUTONOMOUS MULTIMODAL EDITOR — IMPLEMENTATION CONTRACT

This document is the authoritative implementation contract for the current development line `feat/word-reveal-face-tracking` until the redesign is fully implemented and validated.

## Super Goal

Transform Clipper from a transcript-first podcast clipper into a source-grounded, multimodal, content-agnostic autonomous short-form editor that can ingest explicitly targeted videos, understand audio and visuals, identify every genuinely worthwhile narrative/event opportunity, construct campaign-valid clips without arbitrary output quotas, enrich visuals only when policy permits, recover locally from failures, reuse already-computed evidence, and never publish an output that has not passed deterministic, multimodal, technical, policy, and human-review gates.

## Central invariant

A clip exists because the source contains a worthwhile complete moment, not because configuration requests a fixed number of clips.

```python
eligible_clip = (
    target_source_authorized
    and semantic_core_is_valuable
    and narrative_envelope_complete
    and feasible_window_exists
    and duration_is_campaign_valid
    and source_grounded
    and hook_truthful
    and no_required_context_missing
    and no_forbidden_source_overlap
    and campaign_policy_safe
    and visual_strategy_safe
    and technical_qc_pass
    and multimodal_final_review_pass
)
```

The production yield is the set of unique eligible quality moments. Zero eligible moments is a valid completed run.

## Non-negotiable architecture rules

1. Production runs process only explicitly targeted videos. Discovery is a separate workflow.
2. Campaign briefs contain source authorization, campaign constraints and delivery/policy requirements; they do not configure editorial quotas or scoring internals.
3. Output count is quality-derived. No minimum clip count, finalist count, shortlist count, concept count, or per-source quota may force weak outputs.
4. Semantic core, narrative envelope and final delivery window are separate concepts.
5. Deterministic code enumerates legal duration/policy windows. Models rank/select legal options; models do not invent legality.
6. Vision is first-class evidence for visually dependent sources. A visually dependent source cannot silently degrade to text-only planning.
7. Production editorial decisions must not depend on podcast/domain lexical word lists.
8. Every expensive stage is independently resumable and content-addressed.
9. A contract change invalidates only dependent stages, not unrelated paid inference.
10. Generated media is optional, provenance-tracked, truthfulness-reviewed and unreachable when campaign policy forbids it.
11. Rendering begins only after grounding, narrative, duration, boundary and campaign policy eligibility pass.
12. Actual rendered MP4 review remains a final acceptance gate.

## Required target data model

Replace the overloaded story/concept/edit boundary model with:

```text
MultimodalEvent
  -> SemanticCore
  -> NarrativeEnvelope
  -> FeasibleDeliveryWindow[]
  -> QualityMoment
  -> VisualStrategy
  -> EditPlan
```

### SemanticCore
The smallest source-grounded interval containing the interesting idea/event. It may be shorter than campaign delivery duration.

### NarrativeEnvelope
The complete source interval required to understand setup, references, causality and payoff around a SemanticCore.

### FeasibleDeliveryWindow
A deterministic candidate span that contains the SemanticCore, stays inside the NarrativeEnvelope or explicitly model-approved contextual evidence, passes hard source/policy constraints, and is already inside campaign duration bounds.

### QualityMoment
A unique eligible moment that passes hard evidence gates and calibrated editorial quality assessment. Quality determines yield; quota does not.

## Explicit source targeting

Production campaign schema must use explicit targets, for example:

```yaml
targets:
  mode: explicit
  videos:
    - video_id: 2Y4LP85PTak
      url: https://www.youtube.com/watch?v=2Y4LP85PTak
      channel_id: UCf1q6dhccWr6eQEcFFnJSbA
```

An authorized channel is a rights constraint, not implicit permission for a production run to discover/process additional videos.

`clipper run` must never perform channel/search discovery. Discovery/recommendation must be a separate command/workflow whose selected outputs are written into the explicit target list.

## Campaign brief responsibilities

Campaign configuration may describe:

- campaign identity/objective
- explicit target videos
- authorized channels/rights
- min/max delivery duration
- language/region where required by campaign
- watermark/hashtags/platform requirements
- source-segment policy
- branding policy
- portrayal/safety policy
- generated-media policy
- editorial completeness requirements

Campaign configuration must not contain production editorial quotas/knobs such as:

- clip_count
- max_clips_per_source
- candidate_pool_size
- concept_count
- variants_per_concept
- final_render_budget
- minimum_distinct_finalist_concepts
- fixed shortlist counts
- heuristic score weights
- hardcoded hook-mode menus required for output count

Resource/compute limits may exist as operational guardrails but must be reported as budget-limited yield, never as editorial targets.

## Dynamic yield semantics

For N independent eligible quality moments, the pipeline may produce N outputs. Examples:

```text
0 eligible moments -> COMPLETED_NO_ELIGIBLE_MOMENTS, 0 clips
2 eligible moments -> 2 clips
7 eligible moments -> 7 clips
```

A failed render/QC does not cause promotion of a weaker moment merely to preserve a number.

If budget prevents processing all eligible moments, status must report `PARTIAL_BUDGET_LIMIT` with eligible, processed and unprocessed counts.

## Multimodal evidence

Add a canonical multimodal timeline aligned on source timestamps. It must be capable of carrying:

- transcript words and confidence
- speaker identity/turns
- shot/scene boundaries
- visible people
- actions/motion
- objects
- OCR/screen text
- source branding/logos
- visual salience
- source hazard classification
- relevant audio events/energy
- model confidence/provenance

Add a learned/evidence-based `SourceModalityProfile` describing dependency on speech, visuals, motion, screen text, speaker identity and action. Do not route by hardcoded categories such as podcast/game/tutorial.

If visual evidence is necessary for safe/correct planning and VLM perception fails, planning must block/escalate rather than silently continue from transcript only.

## Feasible-window solver

Models identify SemanticCores and context requirements. Deterministic code then enumerates legal windows before final editorial ranking.

Every emitted feasible window must satisfy at construction time:

```python
campaign.min_clip_seconds <= window.duration <= campaign.max_clip_seconds
window.contains(semantic_core)
window.is_chronological
window.has_no_forbidden_source_overlap
window.has_no_known_policy_violation
```

Boundary/narrative completeness is then audited and localized repairs may produce new feasible windows. An LLM must not return an arbitrary illegal start/end pair as a final EditPlan.

The historical regression where all nine EditPlans were 1–11 seconds for a 20-second campaign minimum is a mandatory regression test.

## Production editorial path

The open-model production editor must remain model-driven and source-grounded. Existing heuristic lexical sets may remain temporarily for legacy compatibility/tests, but they must not participate in the production autonomous path. Eventually move them behind an explicitly legacy module/path and delete once parity is proven.

## Visual strategy

After a QualityMoment is eligible, derive a `VisualStrategy` using original source evidence first:

1. relevant original source/action
2. another authorized portion of the same target source when semantically truthful
3. deterministic text/graphics/diagram
4. synthetic illustrative media only when campaign policy explicitly permits
5. remain on original speaker/action footage

Generated scenes must be silent illustrative assets, provenance-marked, truthfulness-reviewed by a VLM and discarded when they imply facts/real footage not supported by the source.

For the current Double Coverage campaign, synthetic visual generation is forbidden and the generator must be unreachable by policy.

## Durable execution DAG

Replace monolithic late-failure behavior with independently persisted stage nodes. Each node records:

- stage/contract identity
- input fingerprints
- dependency output fingerprints
- model identity/revision and decoding parameters where applicable
- output fingerprint
- status
- attempt count
- usage/cost
- timestamps

A failure/retry in a downstream stage must not repeat unrelated upstream stages.

Expected dependency direction:

```text
source
 -> transcription
 -> alignment
 -> diarization
 -> multimodal perception
 -> story/event graph
 -> semantic cores
 -> narrative envelopes
 -> feasible windows
 -> ranking/quality moments
 -> boundary/policy eligibility
 -> visual strategy
 -> render per candidate
 -> technical QC per candidate
 -> multimodal review per candidate
```

## Cache contract

Do not use manually incremented runtime/prompt version strings as broad cache invalidators. Cache identity must be content-addressed by the contract actually used by the stage.

Conceptually:

```python
cache_key = hash(
    source_hash,
    stage_name,
    stage_contract_hash,
    dependency_output_hashes,
    model_revision,
    decoding_parameters,
    relevant_campaign_policy_subset,
)
```

Examples:

- watermark change -> render/QC/review only
- generated-media policy change -> visual strategy/render/review
- feasible-window contract change -> window/ranking/downstream
- ASR model change -> canonical speech timeline/downstream

Preserve compatibility with already-paid cache entries during migrations where possible.

## Acceptance semantics

Acceptance derives expectations from the evidence graph, not externally supplied clip counts.

For each rendered candidate require exactly one consistent record for:

- grounding/source evidence
- boundary eligibility
- campaign policy eligibility
- render artifact
- technical QC
- final multimodal review

A run may complete successfully with zero eligible moments. It must never manufacture clips to satisfy a quota.

## Compute budget semantics

Cost constraints are operational constraints, not output targets. Track expected/actual marginal cost per stage and candidate. Expensive stages should not start when prerequisite evidence is already invalid. If budget ends before all eligible moments are processed, preserve their evidence and report budget-limited partial yield.

Prefer warm persistent Modal model services for repeated structured editorial/vision calls to avoid repeated large checkpoint initialization.

## Required modules / responsibilities

Target additions:

- `src/clipper/multimodal_timeline.py`
- `src/clipper/modality_profile.py`
- `src/clipper/story_graph.py`
- `src/clipper/quality_moments.py`
- `src/clipper/window_solver.py`
- `src/clipper/visual_strategy.py`
- `src/clipper/generated_media.py`
- `src/clipper/dag.py`
- `src/clipper/stage_contracts.py`
- `src/clipper/yield_policy.py`

Preserve proven responsibilities in canonical speech grounding, tracking, rendering, technical QC, campaign/editorial integrity, VLM review and content-addressed storage.

## Preserve regressions

The redesign must not regress:

- word-level reveal captions
- first-caption alignment
- edge-to-edge 9:16 without blurred filler
- speaker-aware framing without hand chasing
- camera-cut-aware stable reframing
- transition stability
- source-master quality preservation
- AV sync/audio QC
- required campaign watermark handling
- foreign/source-logo policy handling
- sponsor/ad/promo exclusion
- source-grounded truthful hooks
- semantic start/end closure
- campaign-policy audits
- multimodal final review
- technical media QC
- package coverage floor >=95%

## Implementation phases

### Phase A — Contract and campaign semantics

- add this authoritative contract to the repository
- explicit target-video schema
- separate rights authorization from run targets
- remove production discovery from `clipper run`
- remove fixed clip/finalist/shortlist quota fields from campaign semantics
- remove mandatory fake keyword compatibility from the new production schema
- update campaign templates and tests

Gate: an unlisted video cannot be processed, and a target list cannot silently expand via authorized channel search.

### Phase B — Dynamic quality-derived yield

- remove fixed quota stopping from production selection
- replace fixed finalist/shortlist targets with evidence-derived yield
- remove exact-count acceptance arguments/assertions
- remove hardcoded six/three behavior from Modal recovery and GitHub production workflow
- support zero eligible moments as a valid completed outcome

Gate: fixture runs with 0/1/2/7 eligible moments produce 0/1/2/7 eligible outputs without filler promotion.

### Phase C — Canonical multimodal evidence

- add multimodal timeline and modality profile
- enrich visual perception with scene/action/object/OCR/branding evidence
- make visual failure policy depend on source modality requirements

Gate: podcast, gameplay/action and screen/tutorial fixtures route correctly without hardcoded source-type keywords.

### Phase D — SemanticCore / NarrativeEnvelope

- separate semantic nucleus from complete contextual envelope
- retain source-word/timestamp provenance

Gate: a short semantic core may own a longer coherent narrative envelope without arbitrary padding.

### Phase E — Deterministic feasible-window solver

- enumerate legal duration/policy windows
- make historical under-duration EditPlan failure impossible by construction
- property tests for duration/chronology/core containment

### Phase F — Model ranking and quality moments

- Qwen ranks legal windows and decides quality, not legality
- constrained structured output
- no fixed output count

### Phase G — Visual strategy

- first-class per-candidate visual plan
- source-first visual decisions
- no synthetic generation yet

### Phase H — Generated media subsystem

- generator interface/provider
- policy gate ALLOW/FORBID/ESCALATE
- provenance/truthfulness review
- Double Coverage regression proving zero generator calls

### Phase I — Durable DAG / resume

- independently persisted stage nodes
- dependency fingerprints
- targeted retries
- cost ledger

Gate: terminate after planning and resume without repeating source/ASR/alignment/diarization/multimodal evidence.

### Phase J — Dynamic acceptance

- evidence-consistency acceptance
- no expected clip-count parameters
- completed-zero-yield state
- budget-limited partial state

### Phase K — Cross-content validation

Validate at minimum:

- two-person podcast
- single-person talking head
- screen tutorial
- gameplay
- sports/action
- visual demonstration
- low-speech video
- sponsor/advertisement region
- logo/branding hazard
- source with no worthwhile moments

## Autonomous implementation loop

For every implementation iteration:

1. fetch exact `feat/word-reveal-face-tracking` head
2. inspect this contract and current implementation
3. select the earliest incomplete phase
4. record starting SHA, phase, affected invariants and expected files
5. implement only work required by that phase plus necessary compatibility changes
6. add regressions/fault-injection tests
7. run Ruff lint, Ruff format, strict mypy, pytest and >=95% coverage
8. verify preserved regressions
9. update the implementation matrix/evidence
10. commit atomically to the same branch
11. do not create another branch
12. do not merge PR #2 until all phases and real MP4 acceptance pass

## No-drift rules

Never:

- weaken the Super Goal because a test is inconvenient
- lower quality to hit a clip count
- restore fixed production quotas
- add podcast/domain lexical rules to the autonomous production path
- silently downgrade visually dependent media to transcript-only editing
- broaden target sources without explicit target entries
- generate synthetic visuals when campaign policy forbids them
- pad boundaries with arbitrary context to satisfy duration
- disable a QC/policy gate merely to make CI pass
- invalidate unrelated paid cache stages
- restart from `main` or discard validated V7–V11 behavior
- claim readiness without actual MP4 inspection
- publish/submit during implementation

## Final Definition of Done

All must pass:

- explicit-target source execution
- no implicit production discovery
- no fixed clip-count requirement
- quality-derived yield, including zero-yield completion
- no weak filler promotion
- canonical multimodal evidence
- modality-aware visual requirements
- SemanticCore/NarrativeEnvelope separation
- deterministic duration-feasible windows
- structured model generation
- no production domain lexical heuristics
- boundary/payoff closure
- source hazard/campaign policy enforcement
- content-addressed per-stage cache
- interrupted-run resume without unrelated recomputation
- cost accounting
- adaptive visual strategy
- generated-media policy gating
- synthetic generation forbidden for Double Coverage
- proven 9:16/caption/tracking/render regressions
- technical QC
- multimodal final QC
- actual MP4 review
- Ruff PASS
- Ruff format PASS
- strict mypy PASS
- pytest PASS
- package coverage >=95%
- PR remains draft/unmerged until final acceptance
