# Autonomous Multimodal Editor — Implementation & Acceptance Matrix

This file is the auditable companion to
`plans/autonomous-multimodal-editor-contract.md`.

Status vocabulary:

- `PASS`: implementation and required evidence both exist.
- `IMPLEMENTED_PENDING_LIVE`: production implementation exists, but current live acceptance is outstanding.
- `BLOCKED_EXTERNAL`: implementation is ready but an external service/account condition prevents verification.
- `PENDING`: required work/evidence is not complete.

No `PASS` may be inferred from historical V7/V8 acceptance when the contract requires evidence from the current autonomous multimodal architecture.

## Authoritative branch

- Repository: `TheBayoumi/clipper`
- Branch: `feat/word-reveal-face-tracking`
- PR: `#2`
- PR policy: remain draft/unmerged until every final DoD gate passes and actual MP4 review is accepted.

## Phase matrix

| Phase | Requirement | Implementation evidence | Verification evidence | Status |
|---|---|---|---|---|
| A | Explicit target videos only | `CampaignBrief`, `brief.py`, `youtube.py`, tracked Double Coverage campaign | brief/source-rights tests | PASS |
| A | No implicit production channel discovery | production source resolution uses explicit target; authorized channel is rights evidence | source/brief tests | PASS |
| A | Campaign config contains policy rather than clip quotas | quota-free tracked campaign schema; legacy internal values retained only for paid-cache compatibility | brief/model tests | PASS |
| B | Output count derives from independent quality moments | `yield_policy.py`, `quality_batch.py`, pipeline dynamic-yield path | 0/1/2/7 and no-quota-fill regressions | PASS |
| B | Zero worthwhile moments is valid | dynamic manifest state and empty QualityMoment behavior | zero-yield tests | PASS |
| B | No unrelated reserve promoted to fill target count | one accepted result maximum per concept; reserves same-concept only | yield/pipeline tests | PASS |
| C | Canonical multimodal timeline | `multimodal_timeline.py` | multimodal timeline tests | PASS |
| C | Evidence-derived modality profile | `modality_profile.py` | modality-profile tests | PASS |
| C | Visually dependent sources fail closed without sufficient vision | `assert_required_modalities_available`; quality-batch visual policy gate | fail-closed coverage tests | PASS |
| D | SemanticCore separated from NarrativeEnvelope | `story_graph.py` | story-graph tests | PASS |
| D | Final window cannot amputate setup/payoff | `NarrativeEnvelope.require_contains`; QualityMoment containment guards | regression tests | PASS |
| E | Deterministic campaign-feasible windows | `window_solver.py` | duration/policy/property tests | PASS |
| E | Previous 1–11 second EditPlan failure structurally prevented | legal windows generated before quality ranking; whole envelope containment | exact short-core/complete-envelope tests | PASS |
| F | Model ranks legal alternatives instead of inventing final timestamps | `autonomous_quality_planner.py`; `quality_windows` chooses supplied IDs | planner tests | PASS |
| F | Structured JSON schemas for all active editorial task families | `editorial_prompt.py`; Outlines worker | local schema tests; Modal smoke pending | IMPLEMENTED_PENDING_LIVE |
| G | Adaptive source-first visual strategy | `visual_strategy.py` | visual-strategy tests | PASS |
| H | Generated-media subsystem hard policy gate | `generated_media.py` | generated-media policy tests | PASS |
| H | Synthetic visuals never invoked for Double Coverage | campaign policy `forbid` + pre-provider gate | zero-provider-call regression | PASS |
| I | Content-addressed durable stage DAG | `dag.py`, `stage_contracts.py` | DAG/replay tests | PASS |
| I | Downstream change does not rerun upstream paid work | dependency/output fingerprints | interrupted/resume regression | PASS |
| I | Source-policy vision resumes at completed observation granularity | content-addressed per-frame checkpoints + explicit durable commit hook | interruption/resume and fully-cached zero-inference regressions | IMPLEMENTED_PENDING_LIVE |
| I | Editorial/vision repeated inference uses persistent model-worker lifecycle | Modal class workers with enter-time loading; no ordinary-batch checkpoint reload | local lifecycle contract tests; live lifecycle IDs/load counts required | IMPLEMENTED_PENDING_LIVE |
| I | Vision capacity is runtime-derived rather than a fixed production batch/VRAM threshold | learned good/bad capacity with adaptive recovery | forced-capacity regression + live per-device VRAM evidence required | IMPLEMENTED_PENDING_LIVE |
| J | Dynamic acceptance derives expectations from evidence | `acceptance.py` | dynamic acceptance tests | PASS |
| K | Two-person podcast | synthetic multimodal acceptance corpus | `test_phase_k_cross_content_validation.py` | PENDING CI |
| K | Single-person talking head | synthetic multimodal acceptance corpus | `test_phase_k_cross_content_validation.py` | PENDING CI |
| K | Screen tutorial | synthetic multimodal acceptance corpus | `test_phase_k_cross_content_validation.py` | PENDING CI |
| K | Gameplay | synthetic multimodal acceptance corpus | `test_phase_k_cross_content_validation.py` | PENDING CI |
| K | Sports/action | synthetic multimodal acceptance corpus | `test_phase_k_cross_content_validation.py` | PENDING CI |
| K | Visual demonstration | synthetic multimodal acceptance corpus | `test_phase_k_cross_content_validation.py` | PENDING CI |
| K | Low-speech source | synthetic multimodal acceptance corpus | `test_phase_k_cross_content_validation.py` | PENDING CI |
| K | Sponsor/advertisement region | source-hazard exclusion fixture | `test_phase_k_cross_content_validation.py` | PENDING CI |
| K | Logo/branding hazard | source-branding exclusion fixture | `test_phase_k_cross_content_validation.py` | PENDING CI |
| K | No worthwhile moments | zero-quality-yield fixture | `test_phase_k_cross_content_validation.py` | PENDING CI |

## Final Definition of Done

| Final gate | Evidence | Status |
|---|---|---|
| Explicit-target source execution | implementation + tests | PASS |
| No implicit production discovery | implementation + tests | PASS |
| No fixed clip-count requirement | implementation + tests | PASS |
| Quality-derived clip yield | implementation + tests | PASS |
| No weak filler clips to meet a quota | implementation + tests | PASS |
| Multimodal timeline | implementation + tests | PASS |
| Speech-dominant handling | implementation + tests | PASS |
| Visually dependent handling | implementation + tests | PASS |
| SemanticCore/NarrativeEnvelope separation | implementation + tests | PASS |
| Deterministic duration-feasible windows | implementation + tests | PASS |
| JSON-schema constrained editorial generation | local implementation PASS; deployed 11-family smoke required | IMPLEMENTED_PENDING_LIVE |
| No production domain lexical vocabulary | autonomous quality-planner path | PASS |
| Boundary and payoff closure | structured boundary + containment gates | PASS |
| Campaign policy enforcement | structured campaign/source-hazard gates | PASS |
| Source hazard exclusion | classifier + deterministic forbidden spans | PASS |
| Content-addressed per-stage cache | DAG/stage contracts | PASS |
| Interrupted-run resume | DAG regression | PASS |
| No unrelated-stage recomputation | DAG regression | PASS |
| Compute/cost accounting | telemetry implemented; current production measurements required | IMPLEMENTED_PENDING_LIVE |
| Warm editorial/vision worker reuse | class lifecycle implemented; exact live worker lifecycle/model-load evidence required | IMPLEMENTED_PENDING_LIVE |
| Runtime-derived vision capacity and OOM recovery | adaptive capacity implementation/tests; live VRAM evidence required | IMPLEMENTED_PENDING_LIVE |
| Durable per-observation visual resume without replay | implementation + interruption/cache regressions; live interrupted/resume evidence required | IMPLEMENTED_PENDING_LIVE |
| Per-device vision VRAM telemetry | worker/provider telemetry implemented; live evidence required | IMPLEMENTED_PENDING_LIVE |
| Adaptive visual strategy | implementation + tests | PASS |
| Generated-media policy gating | implementation + tests | PASS |
| Double Coverage synthetic-media prohibition | tracked policy + provider non-invocation test | PASS |
| True 9:16, captions, active-speaker/action framing | historical regressions preserved; current production MP4 proof required | IMPLEMENTED_PENDING_LIVE |
| Technical QC | implementation/tests; current production artifacts required | IMPLEMENTED_PENDING_LIVE |
| Multimodal final QC | implementation/tests; current production artifacts required | IMPLEMENTED_PENDING_LIVE |
| 0-quality-moment source handled correctly | tests | PASS |
| N-quality-moment source yields N clips | tests | PASS |
| Ruff | exact software-quality run `32503724109` | PASS |
| Ruff format | exact software-quality run `32503724109` | PASS |
| strict mypy | exact software-quality run `32503724109` | PASS |
| pytest | 558 passing on software-quality head | PASS |
| package coverage >=95% | 95.06% on software-quality head | PASS |
| Cross-content Phase K corpus | new tests awaiting latest-head CI | PENDING |
| Both current Modal apps deployed | deploy attempt `32504325285` rejected before deploy because workspace exceeded spend limit | BLOCKED_EXTERNAL |
| 11-family deployed schema smoke | cannot execute until Modal deployment allowed | BLOCKED_EXTERNAL |
| Current Double Coverage planning-only acceptance | requires deployed current workers; preserve `/artifacts/_cache`, no fresh inference | BLOCKED_EXTERNAL |
| Current Double Coverage rendered acceptance | requires planning pass and Modal spend availability | BLOCKED_EXTERNAL |
| Actual current MP4 review | requires current rendered artifacts | BLOCKED_EXTERNAL |
| Current live cost/timing/cache-reuse evidence | requires current production execution | BLOCKED_EXTERNAL |
| PR remains draft/unmerged | PR #2 | PASS |

## External blocker

Modal deployment currently fails before worker deployment with:

`Workspace ... has exceeded its spend limit`

Evidence: GitHub Actions deploy run `32504325285`, step `Deploy exact-HEAD open-model workers`.

This is not a code failure. Do not retry paid/deploy work repeatedly until the workspace spend limit is raised/reset. Once available, create/update only `acceptance/modal-deploy-request.json` to trigger the exact-head deployment workflow.

## Remaining acceptance sequence

1. Keep CI green on the final code head and close Phase K fixture acceptance.
2. Raise/reset the Modal workspace spend limit.
3. Trigger exact-head Modal deployment using the explicit marker.
4. Require both apps + 11-family schema smoke + HF access + pipeline endpoint resolution to PASS.
5. Run Double Coverage planning-only with persistent cache and **without** fresh inference.
6. Verify cache reuse, quality-derived legal windows, source hazards, boundary closure, and policy gates.
7. Run rendering only after planning passes.
8. Verify every generated MP4 with technical QC and open-VLM final review.
9. Materialize/download the actual MP4 artifacts and perform manual playback/frame review.
10. Record actual timing, GPU usage, cost, cache hits, model revisions and source/output hashes.
11. Update this matrix with exact run IDs/artifact hashes and only then consider the contract satisfied.
