# Book pipeline — Phase 1 CPU-only domain integration

The canonical book stages are now in `src/clipper/books/`:

- `contracts.py`: frozen typed stage inputs/outputs and existing stage-contract fingerprints
- `selection.py`: unchanged source-backed catalog selection, history/diversity and legacy JSON layouts
- `scripting.py`: pinned Qwen3 request, schema validation, repair, sentence-only editing, cache/recovery
- `orchestrator.py`: DagStore-backed selection and scripting; later stages registered as NOT_PORTED
- `cli.py`: `clipper books validate`, `plan`, and `run`

The new entrypoints have explicit boundaries:

```sh
clipper books validate --catalog niches/book-editorial-catalog.json --config niches/book-lessons.yaml
clipper books plan --book-id atomic-habits --lesson-id two-minute-rule --output artifacts/books/selection.json --niche-config-output artifacts/books/niche.json --storyboard-output artifacts/books/outline.json
clipper books run --book-id atomic-habits --lesson-id two-minute-rule --script-brief niches/atomic-habits-episode-02.json
```

`books plan` reproduces the legacy selection CLI's JSON serialization, selection
fingerprint, niche config and storyboard outline. `books run` processes selection
and an owner-supplied recorded script without loading weights, then **fails
closed** with nonzero status at `NOT_PORTED: books.narration`. Without
`--script-brief`, the scripting stage uses its existing pinned, explicit
Qwen3 inference path; do not use that mode without authorization.

Book DAG stage records and content-addressed outputs use `clipper.dag.DagStore`
rather than a separate manifest or resume mechanism. Stage output records are
independent of the legacy Qwen cache; the established script cache key and
payload JSON remain unchanged. New CLI artifacts are local drafts, never
publishable evidence.

Compatibility boundaries:
- `scripts/book_episode_selection.py` delegates to the selection package, retaining its CLI
- `scripts/niche_video.py` delegates script drafting/validation/recovery but retains
  archived TTS, prototype visual rendering and the original niche CLI locally;
  it is therefore **not yet a purely thin wrapper**
- Narration, direction, visuals, assembly and review are registered contracts,
  NOT production implementations. No producer is launched for these stages
- This phase does not modify workflow YAMLs, W1/W3, human-review trust policy,
  creative gates, pinned endpoint identity, or existing deployment paths
- Tests use synthetic fakes or a committed script fixture; there was no live
  Modal execution, Qwen/Kokoro weight load, video generation or creative review

Production acceptance still requires a fresh independent adversarial review and
correlated runtime evidence when those later stages are implemented.
