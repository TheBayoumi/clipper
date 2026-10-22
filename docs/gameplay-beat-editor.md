# Clipper gameplay beat editor

For every independently verified hostile payoff, Clipper searches both a normal continuous 10–12 second clip and a source-native two- or three-beat montage. The montage retains verified combat setup and terminal payoff, hard-cuts away uninteresting source time, and produces an actual 10–12 second edit. It does not fabricate evidence, reorder source chronology, cross verified source cuts, include player death or cut protected Finishing Move choreography. MW4 activates and calibrates this Clipper-owned capability through `semantic_editor.beat_editing`.

All retained segments must contain directly verified hostile interaction and lie inside individual source shot boundaries. The skipped gap is bounded; source-original timing, FFV1/NUT staging and delivery QA remain owned by Clipper. Composite opening/ending/retention metrics are measured on retained frames, not the removed footage. Coverage and de-duplication are based on actual source segments.

The analysis manifest now includes `analysis_compute`: normal gameplay features, refinement, semantic verification and editing run on CPU with FFmpeg/NumPy. Qwen3-VL runs on Modal GPUs only if the separately qualified `neural_visual_events` capability is enabled. MW4 now enables source-bound GPU scouting by default, with immutable Git-SHA identity and fail-closed GPU requirements; earlier single-frame GPU smoke passes do not certify full-source neural understanding. Source-wide six-reel neural, independent A/V gold and final same-SHA render acceptance are still mandatory.

Dynamic Clipper beat search now varies total target duration, payoff length,
setup lead and terminal payoff tail across a bounded half-second grid. If a
third independent verified combat interaction occurs between setup and outcome,
it can serve as an escalation beat. Every candidate still goes through normal
source-shot, stringout-cut, death and Finishing Move integrity, independent
anchor coverage and conflict allocation. Both 2- and 3-beat legal proposals
are retained for allocator choice; no fixed clip cardinality is introduced.
