# Book Pipeline Phase 2 — Book-bound storyboard resolution

The book catalog declares one filename per book under
`books[].storyboard_template`. A valid declaration has the exact form
`<book-id>-approved-storyboard.json`; path traversal, unspecified and
cross-book substitution are rejected. The same directory as
`--catalog` contains the corresponding file. No fallback exists.

The canonical `clipper books plan` and `clipper books run` commands
first make the source-backed catalog selection, resolve its own template,
validate the production-storyboard v2 contract and book identity, and
record immutable storyboard provenance inside the selection DagStore
stage output. That provenance contains book ID, the chosen path, source
SHA-256 **of the original file bytes**, and schema. The selection
stage identity includes that provenance, so a changed template forces
a new stage key even when lesson selection remains unchanged.

The existing `--storyboard-template <path>` CLI flag remains an explicit
override. Overrides are still validated for the selected book and
planning-only approval state; an override for another book is never
accepted. In-memory `BookOrchestrator.plan(catalog, board)` callers
retain their API, with the provided object fingerprint recorded as
in-memory provenance. The standalone legacy
`book_episode_selection.py` and `selection.to_storyboard_outline`
remain byte-compatible by design; the book-bound enforcement lives at
the new canonical orchestrator and CLI boundary, not in that legacy
helper. The legacy lesson fingerprint, niche config and outline schema
remain unchanged.

## Plannable books

- `atomic-habits` — original committed template, unchanged
- `deep-work` — new book-specific planning template
- `psychology-of-money` — new book-specific planning template

The new files reuse the existing narrative roles, 30–60-second,
9:16, 1080×1920 and human-review requirements. These are
**specification-only templates**, not visual assets or evidence of
human approval of finished footage. Both new templates remain blocked
pending an approved rigged character, creative direction and human
review. They deliberately do not claim another book's storyboard images
or visual-reference hashes.

`clipper books validate` checks all catalog-bound templates in addition
to the existing book configuration; it fails if any declared storyboard
is missing or invalid. Planning and run fail before any selection-stage
PASS if the selected template is missing, mismatched, malformed or
publication-eligible.

All later production stages are unchanged:
selection, scripting and recorded-WAV narration are implemented;
direction, visuals, assembly and review remain `NOT_PORTED`.
The pipeline still fails closed at direction after verified narration.

No production workflows, W1/W3 decisions, reviewer-identity policy,
creative review gates, VOICE_REVISION limitation, endpoint pins,
Kokoro cache rules, publishing controls or coverage floors were changed.
No inference, provider/model downloads, GPU or paid execution, deployment
or publishing occurred. Production acceptance requires a fresh
independent review and separately authorized correlated runtime evidence.
