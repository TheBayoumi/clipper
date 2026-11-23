# Native selected-chapter private preproduction — Atomic Habits chapters 1 and 7

**Checkpoint preparation, not authorization to provision a Codespace.** The
owner must independently review the changed code first. No private book text
is provided to chat, committed to Git, or emitted in public CI. No images,
video, deployment or publishing. Only selected chapters **1 and 7** may be
transferred to the one future private host. Book-wide scheduling can call the
same per-chapter API later, but has no new execution authority now.

## Two separate security and provenance boundaries

1. **Exporter machine, retained-source private directory.** The owner
   already holds an exact original candidate reference, full original
   EPUB/PDF, verified acquisition, immutable original ingest/dossier/DAG
   evidence. This material stays entirely **on that source machine**.
   `python -m clipper.books.selected_scope` calls the existing strict
   `load_candidate_for_analysis()`, verifies raw bytes and parent source
   before selecting any chapter. It creates a canonical, content-addressed
   bundle containing **only selected original chapter reading blocks,
   original work-unit text/anchors/ordinals, and selected linked reading
   blocks/unrecovered markers**. It carries hashes of the original parent
   edition, acquisition, source, raw and ingest identities, not original
   full-book bytes, other chapter text, or fake BookText/candidate handles.
2. **Isolated CPU Codespace.** Only the two selected bundle JSON files are
   moved, by an owner-controlled private transfer outside Git. The host
   importer checks exact canonical bytes, filename SHA-256, parent hash
   consistency, original unit ordering and exact re-projection of *every*
   included reading block into its selected units and skipped gap markers.
   **The host does not reverify the absent original whole-book EPUB or DAG**.
   Bundle integrity is not a signed external attestation: changing an entire
   private bundle along with its filename is not prevented cryptographically.
   Parent verification was performed solely on the exporter; this distinction
   is preserved in every report, with `parent_full_book_complete=false`.
   Treat source authorization as **unverified**.

If a caption, image, chart, table, footnote, endnote or appendix is **outside**
the chapter's main canonical section, the exporter operator must explicitly
include its original linked section ID via `--linked-sections` in source
reading order. Missing/unrecovered image text appears as a **gap marker**,
not authenticated prose. The current faithful model does not establish an
exhaustive EPUB hyperlink-target graph or prove all externally linked notes
have been selected: `linked_notes_figures_completeness=UNVERIFIED_REQUIRES_SOURCE_REVIEW`.
Inspect that gap against the retained original before locking. Do not claim
a complete chapter when linked notes or caption scope remains uncertain, and
do not use synthetic replacement passages.

## Source-side export (BEFORE creating the Codespace)

These commands run only on the existing private retained-source environment,
**not** in chat, public GitHub Actions, or the new Codespace. Replace exact
paths/IDs with the actual verified private retained-source evidence:

```bash
export PRIVATE_SOURCE_ROOT=/absolute/private/retained-book-artifacts
export ORIGINAL_CANDIDATE_REF="$PRIVATE_SOURCE_ROOT/book-candidates/analysis-refs/atomic-habits/EXACT_INGEST_KEY.json"

# First inspect original canonical source section IDs and any linked note IDs
# against the original retained EPUB/PDF. DO NOT guess IDs or narrow chapters.
python -m clipper.books.selected_scope \
  --artifact-root "$PRIVATE_SOURCE_ROOT" \
  --candidate-ref "$ORIGINAL_CANDIDATE_REF" \
  --chapter-number 1 \
  --chapter-sections ORIGINAL_CHAPTER_1_IDS_IN_CANONICAL_ORDER \
  --linked-sections ORIGINAL_CHAPTER_1_LINKED_NOTES_IF_ANY

python -m clipper.books.selected_scope \
  --artifact-root "$PRIVATE_SOURCE_ROOT" \
  --candidate-ref "$ORIGINAL_CANDIDATE_REF" \
  --chapter-number 7 \
  --chapter-sections ORIGINAL_CHAPTER_7_IDS_IN_CANONICAL_ORDER \
  --linked-sections ORIGINAL_CHAPTER_7_LINKED_NOTES_IF_ANY
```

Each command prints a digest and private path, never book text. If no linked
sections exist, omit `--linked-sections` entirely. The command must succeed
only after original full-parent validation; failure is not permission to
construct new candidate handles or rewrite a two-chapter fake whole EPUB.

Inspect both private exported files' selected section IDs, reading-block
anchors, work-unit counts, gaps and complete byte lengths. Expected rough
words: chapter 1 ~4,372, chapter 7 ~1,567; those are *not* permission to
truncate or replace source. Retain original ordinals, source order, image
markers and meaningful omissions. Only copy the **two exact content-addressed
`book-candidates/selected-scopes/<SHA256>.json`** files to the future
Codespace's matching private directory. No other source material is needed
there. The host will reject bundles whose name or bytes disagree.

## Before host creation — strict cost and deadline controls

- Keep GitHub's existing account-wide **$0 paid-overage block** unchanged,
  independently confirm enough unspent included hours. One private Codespace
  only, repository `TheBayoumi/clipper`, branch
  `review/unified-tts-cleanup`, **4 cores / 16 GiB / 32 GiB**, no other
  machine, account, repository or branch. Maximum **eight included core-hours**.
- **Two hours total starting at Codespace creation**, including setup, source
  inspection, both chapter source-driven argument/audit/reconciliation briefs, independent human reviews, any
  resumes, failure handling and cleanup. A machine-time cap is **not**
  assurance that either chapter completes.
- **Mandatory external enforcement:** `scripts/private_chapter_external_supervisor.sh`
  runs on the owner's **separate computer**, not inside the Codespace, and
  begins immediately after creation **before** installing dependencies. It
  waits for the exact REST instance to reach Available within an immutable
  **creation +10-minute bootstrap deadline**, then performs bounded SSH
  readiness retries and public-key transfer before issuing its first signed
  heartbeat. Startup and retries also reserve at least 60 seconds before
  creation +115 minutes; provisioning never resets the original clock.
  Unexpected/terminal REST state, lost REST access, replacement numeric ID,
  exhausted startup budget, or hung CLI fails closed into identity-checked
  cleanup. No dependencies, private book transfer, model execution or setup
  are permitted before the authenticated signed heartbeat is available.
  After bootstrap it continuously signs and privately transfers
  45-second-verifiable heartbeats (owner signing private key stays outside
  host) and forcibly stops/deletes
  that exact Codespace at creation +115 minutes, independently checking REST
  stop/deletion. If restarted with the exact approved REST instance between
  minute 115 and minute 120, it enters **cleanup-only**, with no setup or
  heartbeat; at/after minute 120 it cannot claim bounded automated cleanup.
  Never restart the clock, approve a different numeric ID or create another host. This covers **dependency installation, waiting for human review,
  all invocations and cleanup**. Do not start setup without a functioning
  external supervisor; if external connectivity, heartbeat or REST deletion
  verification fails, **stop and verify/delete the Codespace manually**.
- The in-host wrapper's GNU `timeout` **only** guards the Python invocation;
  it is **not** proof of Codespace stop or deletion. Both the CLI and direct
  `chapter_host.execute()` require GitHub REST-backed ID/shape/creation time,
  exact approved checkout SHA and a fresh signed external heartbeat before
  any providers. A branch name, environment variable, Python timer or
  unsigned self-generated local file is insufficient. Every invocation uses
  the same actual Codespace creation timestamp; no reset or extra host.
- The external signed heartbeat verifies *recent external monitoring*, not a
  mathematical guarantee that future network access will survive. The owner
  remains responsible for confirming REST stop/deletion and the unchanged
  account-wide $0 paid-overage block. No host usage is approved by this
  runbook until actual independent Codex post-fix review clears the SHA.

## Private host manifest — only selected evidence

All paths are **absolute and private outside the Git worktree**; `umask 077`.
After Codex review, the owner must first separately confirm the exact
approved commit SHA and create the one authorized host; this document
**does not create that host**. Generate a fresh private RSA key on the
owner's external machine (for example
`openssl genpkey -algorithm RSA -pkeyopt rsa_keygen_bits:2048 -out owner-supervisor.key`
and `openssl pkey -in owner-supervisor.key -pubout -out owner-supervisor.pub`).
Keep `owner-supervisor.key` **outside** the Codespace forever. On the owner
machine, start and keep running:

```bash
export CODESPACE_NAME=EXACT_EXISTING_APPROVED_NAME
export PRIVATE_CHAPTER_APPROVED_SHA=EXACT_POST_CODEX_REVIEWED_40_HEX_SHA
export PRIVATE_CHAPTER_SUPERVISOR_PRIVATE_KEY=/absolute/owner-supervisor.key
export PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY=/absolute/owner-supervisor.pub
export PRIVATE_CHAPTER_ROOT=/tmp/clipper-chapter-private
# Mandatory, owner-side absolute PRIVATE directory outside any Git worktree.
export PRIVATE_CHAPTER_RECOVERY_DIR=/absolute/owner-private-chapter-recovery
bash scripts/private_chapter_external_supervisor.sh
```

**This process must be active BEFORE any host setup**; it sends only
heartbeat, public key and signature, never private source text. Stop/delete
confirmation must be observed after the run. In the Codespace, export the
matching **same** exact-approved SHA and fingerprint of the public key:

```sh
export PRIVATE_CHAPTER_APPROVED_SHA=EXACT_POST_CODEX_REVIEWED_40_HEX_SHA
export PRIVATE_CHAPTER_SUPERVISOR_PUBLIC_KEY_SHA256=OWNER_PUBKEY_SHA256
```

Do **not** use the SHA from this document or a branch name as implicit
approval. Independently fingerprint the public key on the owner machine
(`sha256sum owner-supervisor.pub`). If the signer heartbeat is not fresh,
host preflight fails; do not bypass or fall back to an in-host watchdog.

After the external supervisor is active and the two bundles are transferred,
use `/tmp/clipper-chapter-private` as an example private root:

```sh
export PRIVATE_CHAPTER_ROOT=/tmp/clipper-chapter-private
export PRIVATE_CHAPTER_MANIFEST="$PRIVATE_CHAPTER_ROOT/two-chapters.json"
mkdir -p "$PRIVATE_CHAPTER_ROOT/book-candidates/selected-scopes"
chmod 700 "$PRIVATE_CHAPTER_ROOT"
# No dependencies or weights may be installed until this stdlib preflight
# authenticates REST instance, exact SHA, and external signed heartbeat.
python -m clipper.books.chapter_lifecycle \
  --artifact-root "$PRIVATE_CHAPTER_ROOT" \
  --codespace-created-at EXACT_GITHUB_REST_CREATED_AT
python -m pip install -e ".[niche-models,book-constrained]"
```

The `two-chapters.json` manifest (mode 0600) must use exact imported digests:

```json
{
  "schema": "clipper-two-private-chapter-codespace-v2",
  "chapter_1": {
    "bundle_path": "/tmp/clipper-chapter-private/book-candidates/selected-scopes/CHAPTER_1_SHA256.json",
    "bundle_sha256": "CHAPTER_1_SHA256",
    "scope_sha256": null,
    "review_file": null
  },
  "chapter_7": {
    "bundle_path": "/tmp/clipper-chapter-private/book-candidates/selected-scopes/CHAPTER_7_SHA256.json",
    "bundle_sha256": "CHAPTER_7_SHA256",
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

**Phase inspect — model-free** (single shared host clock already running):

```sh
bash scripts/private_chapter_codespace.sh inspect
```

Privately inspect the content-addressed
`book-candidates/private-host/scope-inspection-<SHA256>.json` report, which
exposes chapter-specific original work counts and exact selected
anchors, linked IDs, figure/unrecovered markers and source gaps. It flags
parent bytes unverified on host and linked-note completeness not attested.
Resolve those questions using the retained original on the exporter machine;
do not copy full source onto the host. Lock each exact `scope_sha256`
reported for its own selected bundle in the private manifest. Neither the
scope lock nor a tokenizer-valid prompt constitutes human semantic approval.

**Phase hybrid brief — selected-source argument, local audit and reconciliation, sequential per chapter:**

```sh
bash scripts/private_chapter_codespace.sh hybrid-brief
```

Both bundle identities and scopes are verified **before the first producer call**.
The quality-first hybrid route **does not call the per-unit miner**: it freezes
every original selected source unit and unresolved gap, drafts the chapter-level
argument, audits disjoint contiguous source windows, and reconciles an
explicit unit-level keep/compress/omit decision and source-anchored beats.
Source-ledger/input/output caps remain finite; exact rendered token-ID hashes,
effective chat-template and tokenizer-config identities, and the full reserved
output receive public-only pinned-tokenizer checks **before** chapter attempts.
The very token tensor consumed by real CPU generation must match that
preflight. Raw failures, two-attempt owner claims and replay guards remain.
No truncation, short Episode fallback, paid fallback, images or video.
Kokoro does not run during the hybrid brief.

The complete selected chapter brief and full original-unit/gap review packet
are private `book-candidates/private-host/chapter-1-review-packet.json`
and `chapter-7-review-packet.json`. Their `REVIEW_REQUIRED` templates are
deliberately invalid; **no automated acceptance**. Every unit KEEP/COMPRESS/
OMIT and each beat requires a substantive independent source decision.
Missing/uncertain notes, source qualification and omissions must be treated
explicitly. Each chapter has a separate review and brief identity.

**Phase hybrid resume — only after explicit two independent source-level human reviews:**

Set each `review_file` to its own full reviewed JSON, with no placeholder
values. Continue only if the **same** Codespace has time:

```sh
bash scripts/private_chapter_codespace.sh hybrid-resume
```

The original selected bundle, scope, complete direct-source ledger, the
argument/local-audit/reconciled brief raw replay, producer/tokenizer identity
and human review are verified again, followed by adaptation, connected
script, shot/direction continuity and Kokoro narration with measured WAV timing. CPU model phases release caches
sequentially. Two chapter artifacts remain independent, and even completed
text preproduction has `publish_eligible=false`.

The wrapper still accepts legacy `brief` and `resume` modes for their
separately reviewed legacy path, but **they are not the authorized
quality-first hybrid commands** for this checkpoint. An `inspect` report
is not a semantic sign-off; a `hybrid-brief` result is not authority
to run `hybrid-resume` until BOTH independent private human reviews exist.

If one chapter fails or time expires, report **its actual last completed
phase and private evidence**; never claim the other chapter succeeded,
restart irreversible attempts, extend the host window, automatically infer
approval, or create another host. No private text, raw output or WAV is
transferred to this chat or published.

### Owner-only recovery before deletion (same 120-minute limit)

The mandatory `PRIVATE_CHAPTER_RECOVERY_DIR` must exist or be safely
creatable on the owner's **separate** machine outside Git, with owner-only
permissions. The external supervisor tries to retrieve the exact chapter
output inventory before STOP/DELETE, within a separate bounded **16-second
retrieval allowance inside the original cleanup reserve**. The remote
`chapter_host recover` command builds a checksum-bound TAR containing
only scoped, verified private output files: validated text/replay evidence,
the human-review checkpoint, chapter source ledger and final result when
available, per-chapter narration reports, and every WAV linked by the
verified report. The source EPUB, nonselected chapters, images and videos
are not part of this output archive.

On success the owner receives
`$PRIVATE_CHAPTER_RECOVERY_DIR/$CODESPACE_NAME/chapter-recovery.tar`,
`safe-recovery-status.json`, and `retrieval-verification.json`. A separate
`host-cleanup-verification.json` records REST-confirmed host stop and deletion
without inferring anything about file recovery. The
owner independently hashes the TAR and every allowlisted member against the
private output manifest **before** marking retrieval verified. If the
download/hash check fails, the archive must **not** be treated as recovered;
the status records `retrieval_status=INCOMPLETE`, with a best-effort safe
phase-only status. This never vetoes or delays a separate, identity-checked
STOP/DELETE operation. **Within the final seconds of the original 120-minute
deadline, DELETE precedes STOP and polling**, after a fresh numeric REST
identity check; waiting for a hung graceful STOP must not consume the DELETE
slot. A verified DELETE with no STOP attestation is reported distinctly,
and an expired deadline is never misreported as a rejected host identity.
Report retrieval completeness and REST host deletion **separately**: recovered
files are not proof of deletion, and successful deletion is not proof that
any files survived.

The private TAR indexes each physical WAV or other identical file **once**,
while retaining both chapter-number references when both attested narration
reports refer to the same content hash. Per-chapter inventories explicitly
distinguish `VERIFIED`, `INCOMPLETE`, and `NOT_STARTED`. A missing or
corrupt chapter-7 artifact must not discard independently verified chapter-1
outputs; a failed chapter is never described as recovered. The owner verifies
the TAR hash, each member hash, each chapter-reference mapping and its
individual retrieval status before treating even partial output as saved.
Text-stage recovery first checks the exact winning claim and any prior
rejected raw/failure history with the normal bounded stage replay verifier;
validated/raw files alone never make a stage recoverable.

The safe status only exposes stage names, source-independent SHA identities,
and `runtime_journal_status` versus `retrieval_status`; it never contains
source passages, review prose, narration text or WAV bytes. The TAR and any
other private outputs stay strictly on the owner machine. A process killed
during adaptation, connected script, narration or direction should report
the actual last journaled subphase and earlier verified artifact identities.
A timestamp, heartbeat, output manifest or green synthetic CI does not
prove successful live chapter execution.
