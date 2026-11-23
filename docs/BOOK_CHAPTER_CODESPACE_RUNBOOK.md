# Private Atomic Habits chapters 1 and 7 — single Codespace checkpoint

**Preparation only.** This runbook does not authorize GitHub Actions or any
assistant to start a Codespace, invoke weights, download a book, or upload text.
Owner must independently review the changed code first. **The Codespace does not
exist yet.** No part of this workflow publishes or renders images/video.

## Before creation — hard stop checks

- Preserve the **account-wide $0 paid-overage block** in GitHub billing and
  confirm unused included core-hours. Four cores for two hours consumes up to
  **eight core-hours**; included allowance is not proof of completion. Do not
  choose another host, machine size, account, or repository.
- Owner creates **one** temporary **private** Codespace on
  `TheBayoumi/clipper`, branch `review/unified-tts-cleanup`, machine **4
  cores / 16 GiB RAM / 32 GiB storage**. Actual runtime/CPU availability must
  be confirmed there. Start the clock at **Codespace creation**, not first
  inference. Stop it before creation + 2 hours, including setup, review and
  any cleanup.
- The launcher reads the live Codespace creation timestamp and machine shape
  through `gh api user/codespaces/$CODESPACE_NAME`; if this fails (including
  token permission), **stop**, do not use an unverified hand-entered timestamp.
  It enforces a common hard GNU `timeout` ending at creation + 115 minutes,
  reserving **five minutes** for private evidence cleanup and stopping the
  Codespace. This does not make any CPU call finish in time.
- Place **only the permitted original source and artifacts in a private
  directory outside the worktree**, e.g. `/tmp/clipper-chapter-private`, with
  permissions 0700/0600. Do not commit, print, paste, attach to this chat,
  upload through CI, or copy raw book text into repository files. The command
  uses one `HF_HOME` under this directory and the existing pinned Qwen
  revision. A single snapshot is reused across both scopes; public tokenizer
  metadata is checked before chapter weights are loaded. Kokoro's `v1.0`
  remains a **label**, not a verified immutable commit.

## Source prerequisite and exact scope inspection

This integration accepts an **existing immutable candidate analysis reference**
from the real acquisition+faithful ingest pipeline, together with the original
private raw EPUB/PDF and ingest artifacts. An arbitrary pasted chapter or short
Episode is **not** an acceptable replacement. If the parent candidate/edition
cannot be verified, stop; do not fabricate an analysis ref or mine outside the
two selected scopes. Existing offline candidate ingest may be performed
privately before this checkpoint; it needs the original pinned acquisition
artifact and registry. The host driver does not fetch a substitute book.

In the private Codespace (commands run only **after** owner approval):

```sh
export PRIVATE_CHAPTER_ROOT=/tmp/clipper-chapter-private
export PRIVATE_CHAPTER_MANIFEST="$PRIVATE_CHAPTER_ROOT/two-chapters.json"
mkdir -p "$PRIVATE_CHAPTER_ROOT"
chmod 700 "$PRIVATE_CHAPTER_ROOT"
# Setup dependencies on this same CPU host, counted in its two-hour budget.
python -m pip install -e ".[niche-models,book-constrained,book-pdf]"
# Inspect the original candidate's chapter/page section IDs privately.
python - <<'PY' > "$PRIVATE_CHAPTER_ROOT/canonical-sections.json"
import json, os
from pathlib import Path
from clipper.books.orchestrator import BookOrchestrator
root = Path(os.environ["PRIVATE_CHAPTER_ROOT"])
# Operator supplies the exact previously ingested immutable candidate ref.
ref = root / "book-candidates/analysis-refs/atomic-habits/EXACT_INGEST_KEY.json"
_, book = BookOrchestrator(root).load_candidate_for_analysis(ref)
print(json.dumps([{"section_id":s.chapter_id,"heading":s.heading}
                  for s in book.sections], ensure_ascii=False))
PY
chmod 600 "$PRIVATE_CHAPTER_ROOT/canonical-sections.json"
```

The example `EXACT_INGEST_KEY` **must** be replaced with the actual private
ref name; no file should be invented. If PDF sections are pages, include
**all** pages belonging to each chapter, and relevant captions/footnotes/
appendices in canonical source reading order. Do not assume a heading named
“Chapter 1” covers all its illustrations or notes. Select **two disjoint
explicit ordered lists** with no other chapter sections.

Create `$PRIVATE_CHAPTER_MANIFEST` in the private directory (0600):

```json
{
  "schema": "clipper-two-private-chapter-codespace-v1",
  "candidate_ref": "/tmp/clipper-chapter-private/book-candidates/analysis-refs/atomic-habits/EXACT_INGEST_KEY.json",
  "chapter_1": {
    "section_ids": ["EXACT_CANONICAL_CHAPTER_1_SECTION_IDS_IN_ORDER"],
    "scope_sha256": null,
    "review_file": null
  },
  "chapter_7": {
    "section_ids": ["EXACT_CANONICAL_CHAPTER_7_SECTION_IDS_IN_ORDER"],
    "scope_sha256": null,
    "review_file": null
  },
  "plan": {
    "schema": "clipper-chapter-preproduction-run-v2",
    "brief_cpu": true,
    "adaptation_cpu": true,
    "script_cpu": true,
    "direction_cpu": true,
    "kokoro_cpu": true,
    "max_scenes": 8,
    "max_words": 1000,
    "max_beats": 20,
    "max_shots_per_scene": 3,
    "max_new_tokens": 8192,
    "max_attempts_per_stage": 2,
    "max_runtime_seconds": 900,
    "voice": "af_heart"
  }
}
```

Inspect *without inference*:

```sh
bash scripts/private_chapter_codespace.sh inspect
```

Privately review `book-candidates/private-host/scope-inspection-<SHA256>.json`
(where the SHA-256 is returned by `inspect`): all selected work units, source bytes, warnings, image/figure/caption/footnote
gaps, unselected sections, exclusions, page warnings and recovery status.
The rough source expectations are ~4,372 whitespace words for chapter 1 and
~1,567 for chapter 7; they are **not** trusted exact hashes or permission to
trim. Unexpectedly low/high counts or unresolved linked notes require an
explicit owner disposition. If the original edition's complete chapter
scope cannot be established, **stop**. Chapter boundaries must never be
silently narrowed. Lock each exact returned `scope_sha256` into its own
manifest field **only after private source inspection**; this is a scope
identity lock, **not** editorial approval.

Source ledger size is bounded to **163,840 bytes** (four times the pinned
40,960-token model context), with at most **64 original work units/gaps per
chapter**; actual eligibility requires an independent tokenizer-EXACT
preflight of the full chat template **plus the full output reservation**
before any chapter call. UTF-8 bytes are **not** used as a proxy for actual
token count on this host. If a full chapter is outside these limits, stop and
report it; do not shorten, silently trim, or switch to Episode mode.

## Phase 1 — real scoped mining and brief, then STOP

```sh
bash scripts/private_chapter_codespace.sh brief
```

Only the two locked parent-canonical scopes are mined, sequentially and
independently, with the existing constrained Qwen3-1.7B miner, one shared
pinned snapshot, and its raw/private completion evidence. Each chapter runs
real ChapterBrief on its **entire scoped original source ledger**, including
zero-candidate units and explicit skipped markers. A failed/incomplete scoped
mining phase is recorded as such and never treated as complete or silently
restarted. No adaptation, script, narration, or direction call can happen
during `brief` because `review_file=null`.

A completed brief writes under `book-candidates/private-host/`:

- `chapter-1-review-packet.json`, `chapter-7-review-packet.json`: generated
  brief, original exact source units and gaps, quoted spans, full coverage,
  and a **deliberately invalid** `REVIEW_REQUIRED` review template.
- `chapter-1-brief-record.json`, `chapter-7-brief-record.json`: pinned
  scope/progress hash and exact generated brief identity.
- Source/attempt proofs remain in the existing private chapter, mining and
  output directories. **Review packet is not a human decision.**

## Human source/omission gate and Phase 2 — resume

Independently inspect the **full original chapter** beside each private
packet. A human supplies two separate explicit review JSON files matching
`clipper-chapter-human-beat-review-v2`: one decision and substantive reason
for every original unit/gap and every brief beat. `support_attested`,
`reject`, `accept_coverage`, qualifications and omission dispositions
must reflect actual inspection. **Never auto-accept template rows.** If
support cannot be attested, reject and stop; do not manufacture semantics.

Set the two distinct `review_file` fields in the private manifest to these
files. Then, **only if the same Codespace still has time**:

```sh
bash scripts/private_chapter_codespace.sh resume
```

Each chapter re-verifies original scope, complete mining, exact raw/validated
brief, all prior attempt records, and its independently attested review.
Only then does it run the existing adaptation, linked chapter script,
measured Kokoro WAV narration/timing and source-bound text shot/direction
continuity. No image, video, renderer, upload or publishing. The final
`CHAPTER_PREPRODUCTION_READY_UNREVIEWED` remains **editorially
unapproved**, with `publish_eligible=false`.

If setup, source mapping, mining, review, narration or any retry cannot
finish within the one host window, leave private evidence and report the
last completed phase only. **Never create another host, restart a failed
irreversible attempt, extend the deadline or turn on paid overage.**
Stop the Codespace before the **original** creation+2h deadline; handle
private artifacts out of band through an owner-controlled private route,
not this chat or the repository.
