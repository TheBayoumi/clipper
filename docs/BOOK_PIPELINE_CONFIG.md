# Configurable chapter pipeline: offline planning v1

The CLI remains a planning boundary and neither authorizes nor starts inference,
downloads weights, creates Modal resources, or publishes. The existing hybrid
evaluator also accepts a bounded configured text policy through its Python API.
Legacy evaluator calls and Codespace protections remain unchanged.

## Operator commands

Generate a script target configuration using the SHA-256 of the selected
chapter bundle (not the parent EPUB):

```sh
python -c 'import json,sys; from clipper.books.pipeline_plan import template; print(json.dumps(template(sys.argv[1], "script"), indent=2))' "$BUNDLE_SHA256" > "$PRIVATE/config.json"
clipper books pipeline-plan --config "$PRIVATE/config.json" --artifact-root "$PRIVATE"
```

The configuration declares every stage explicitly. Edit `enabled`, `semantic`,
and `input` independently. Targets are source, brief, script, narration,
direction, preproduction, visuals, video, render and publish. The last four
are unsupported book bridges, not executable targets. The direction entry
models estimated timing; audio-timed direction integration is not yet represented.

A source descriptor can be prepared using the actual public planner API:

```python
from pathlib import Path
import hashlib
import json

from clipper.books.pipeline_plan import (
    DESCRIPTOR_SCHEMA,
    registry,
    stage_input_identity,
    template,
)

root = Path("/absolute/private/artifacts")
relative = "selected-chapter.json"
raw = (root / relative).read_bytes()
digest = hashlib.sha256(raw).hexdigest()
config = template(digest, "script")
config["stages"]["source"]["enabled"] = False
config["stages"]["source"]["input"] = {
    "schema": DESCRIPTOR_SCHEMA,
    "identity": stage_input_identity("source", registry()["source"], digest, {}, ()),
    "output_sha256": digest,
    "path": relative,
}
(root / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
```

Use owner-private files and a directory outside the repository. Never commit the
bundle or config containing private references. The CLI validates actual selected
bundle bytes using strict selected-scope import in a temporary directory. It does
not write into the supplied artifact root.

The report contains ordered stage decisions, identities where actual dependency
output hashes are available, semantic lineage, configuration hash, and reasons.
It contains no source prose or paths. Exit 0 means the target has a resolved
*plan*, exit 1 means blocked, exit 2 means invalid configuration. All reports
are DRY_RUN_ONLY, never execution approval.

## Reuse and semantics

A disabled dependency cannot run. A matching descriptor alone is insufficient:
a trusted application validator must read and independently validate the actual
artifact and its provenance against the supplied dependency bytes. Without one,
the planner reports unverified input. No configuration boolean can grant trust.
The CLI currently wires only the selected-source validator; integration of
existing Qwen raw replay, human review and other stage validators is outstanding.
The Python resolver accepts trusted validators from application code for future
integration. They must be offline and independently verify provenance; test
fixtures are not production validators.

Stage identity uses StageContract and stage_identity. It binds source digest,
stage schema, provider, pinned model/revision, effective stage-local limits,
dependency output hashes and transitive semantic lineage. Target selection and
unrelated stage configuration are excluded. Pending dependencies have no
fabricated output hash or final cache key. Existing legacy cache keys are not
migrated automatically. Explicit compatibility validation is required.

Human review is an independent gate, not a model invocation. A disabled review
does not become an approved review. A verified review remains human attestation,
not independent semantic certification.

## Implementation inventory

| Stage | Existing implementation | Planner integration |
| --- | --- | --- |
| source | strict selected-scope import | offline validator |
| argument, audits, reconcile | hybrid Qwen | planning only |
| human review | source/beat validation | gate, replay adapter pending |
| adaptation, script | hybrid Qwen | planning only |
| narration | Kokoro | planning only |
| direction | chapter shot validation | planning only, estimated timing |
| visuals, video, render, publish | unported chapter bridge | unsupported |

Existing-video clipping and legacy Episode orchestration are separate and are
not made configurable by this change. Provider selection is pinned in the
registry; alternate providers and host/resource authority are not yet wired.
Full execution integration, private checkpoint IO and the Modal adapter remain
separate follow-up work. No runtime or cost qualification is implied.

## Bounded configured hybrid evaluator API

The existing `chapter.evaluate_hybrid_chapter` accepts optional
`pipeline_config=template(bundle_sha256, "brief")` or the "script" target.
Pass the existing canonical selected bundle and section manifest paths, the
existing explicit ChapterPlan CPU permissions, and explicit producer bindings.
Configuration never grants CPU permissions. The CLI remains dry-run-only.

For a reviewed script resume, retain the exact returned brief identity and
the human review file. Use the script target, disable argument, audits and
reconcile for native cache-only replay, and pass expected_brief_identity and
review_file. Human review remains mandatory even if its stage is disabled.
Only the replay-only prefix followed by enabled work is supported; a disabled
stage after a potentially new stage is rejected before any producer call.
A missing or invalid native cache in the replay prefix fails closed.

Configured source identity is versioned separately from legacy source identity.
Each configured request binds its stage semantic lineage and relevant limits.
Target selection, CPU permission flags and unrelated voice settings do not
enter those configured text keys. The existing raw validator, attempt
reservation, EOS and provenance checks remain authoritative. No legacy cache
migration or external text descriptor import is performed. Canonical native
cache replay is not the same as the CLI descriptor verifier.

Only brief and script targets are supported in the evaluator integration.
Other targets fail closed; narration and direction do not run. The script
result remains unreviewed and nonpublishable. The local API is not a private
host authorization mechanism or a Modal route.

## Configured execution evidence

Each successful configured brief, human-review pause or script invocation retains
a content-addressed private execution receipt. It binds the exact configuration,
semantic lineage, stage decisions, source and artifact identities. This receipt
is separate from the stable script content artifact: replay can change invocation
evidence from RAN to REUSED without changing the script identity.

The brief target reports BRIEF_READY_UNREVIEWED and review REQUIRED_NEXT. It is
not human-approved. The source stage always strictly imports the caller-supplied
immutable selected bundle and never acquires source data. The native evaluator currently requires source validation enabled. Disabling it
is rejected before any import or producer call, rather than silently ignoring the
flag. The offline planner can validate a disabled source from an explicit input
descriptor, but the native evaluator does not yet implement that descriptor bridge. Producer callbacks on disabled
text stages are provenance bindings for exact native cache verification, never
permission to invoke them. Human review disabled means reuse a matching actual
attestation, never waive the review requirement.

## Native replay CLI

`clipper books pipeline-replay --config PRIVATE/config.json --selected-source
PRIVATE/chapter.json --section-manifest PRIVATE/sections.json --artifact-root PRIVATE`
consumes the configuration in the existing hybrid evaluator. Disable all compute
stages in this replay-only command; source validation stays enabled. For a script
replay, also pass `--review-file` and `--expected-brief-identity`. Only retained
actual-Qwen native artifacts with matching provenance can pass; injected test
artifacts do not become actual-model output. Missing caches fail closed without
inference, model downloads or network tokenizer fallback.

This is an actual configuration-consuming cache replay entrypoint, not permission
to generate missing work. Real inference still needs the separately verified
private execution host and resource approval. No Modal launch or private transfer
bridge is implemented by this command.


## Private phase packet boundary (not a launch command)

The `clipper.books.private_phase` module defines a bounded binary packet,
reviewed phase manifest, restore/collect functions and the worker consumer
`execute_private_phase`. The packet carries only explicit selected source,
section manifest, configuration and optional review bytes. The separate
checkpoint contains allowlisted native evidence, not a mounted repository.
The worker uses the configured hybrid evaluator, not a synthetic launcher.

The manifest binds exact commit, workspace, configuration, source, section,
phase, checkpoint, review and brief identities plus owner and resource bounds.
Four CPU cores and 16 GiB are candidate limits, not measured performance.
The timeout is a finite platform phase bound, not a whole-campaign deadline.

The intended caller is an authenticated private SDK client. This commit does
not supply that client, authenticate credentials, independently establish
workspace identity, reserve a durable dispatch claim, create an App/Function,
or verify teardown. The manifest approval field is a required declaration,
not authenticated authority by itself. Do not call the worker as a standalone
launch route. The future launcher must establish these prerequisites before
dispatch and bind the actual App/Call identity to durable owner evidence.
The existing Actions secret must not receive private book bytes or be exported.

The caller can use `pack`, `preflight` and `persist_checkpoint` offline.
Checkpoint output must be durably saved and verified before remote teardown.
An owner-only Drive backup is a separate caller operation, not an in-container
backup. The worker returns a private review packet at the brief boundary and
native script evidence after review; a failed phase may return partial evidence
with a no-retry status. Platform termination can still prevent output return.
No automatic retry or recovery of an ambiguous remote invocation is authorized.


## Private SDK launcher (unqualified, no execution approval)

The launcher module is `clipper.books.private_modal`. Its default command
is offline preflight:

```sh
python -m clipper.books.private_modal \
    --manifest "$PRIVATE/manifest.json" --packet "$PRIVATE/input.bin" \
    --approved-manifest-sha256 "$REVIEWED_MANIFEST_SHA256" \
    --checkout "$REVIEWED_CHECKOUT" --journal-root "$PRIVATE/journal" \
    --output "$PRIVATE/checkpoint.bin"
```

A script phase also takes `--checkpoint` for the exact previous brief archive.
The reviewed digest is the canonical content_fingerprint of the manifest.
Only a separately approved invocation may add `--execute`. The launcher uses
the pinned 1.5.5 SDK, explicit securely injected environment credentials and
live workspace validation before constructing an App. Do not pass credentials
on command lines or use public Actions for private bytes. An exclusive durable
phase directory prevents a second dispatch in the same journal root.

The Function is ephemeral, without a public endpoint, GPU or volume. It uses
the public pinned repository checkout, no ambient source mount, a single
container and explicit CPU/memory request/limit pairs. The SDK returns private
bytes; checkpoint persistence and reread precede App context exit. Positive
cleanup requires exact stopped state and the string task count "0"; otherwise
the result is blocked, even if checkpoint bytes were saved.

This implementation has not been run or qualified. Image dependency resolution,
SDK serialization, image-build bounds, caller-side hard interruption handling,
durable recovery of an unknown App/Call, complete checkpoint validation and
credential availability remain qualification blockers. No live authorization is
implied by the code or manifest. Never automatically retry an ambiguous phase.


### Failed phase retention

The worker now guards archive finalization as well as evaluator execution.
A failure returns a fixed no-retry envelope with the manifest and owner
identities, safe artifact digests where known, and either a marked partial
archive or explicit OUTPUT_NOT_RETAINED. Partial archives record omitted
member counts and whether inventory was truncated; they are not valid
automatic resume inputs. The caller journals the sanitized response before
attempting to save bytes, including when no packet exists. A failed brief
cannot automatically resume. A human expert may inspect retained valid
brief evidence after a review-packet failure; a full rerun requires separate
owner approval and reconciliation of the original dispatch.

The image build now requests repository-compatible gdown and yt-dlp
versions and imports the actual worker module before staging pinned model
material. This build remains unqualified. Cleanup observations use bounded
backoff and one exact-ID stop, and client handles are closed on exit.
The private journal and output directories must be owner-only directories.


### Live authorization hold and CPU image qualification

The public launcher currently rejects every execute request before authentication
or resource creation. The internal dispatch implementation remains available
for offline lifecycle qualification, not as an alternate operational entrypoint.
A future reviewed change must explicitly lift the hold only after the owner
approves the exact phase resource estimate/cap and effective paid-overage zero
is verified with the merchant. Neither the manifest approval string nor
included credits establishes a zero-cost guarantee. No billing API is assumed.

The CPU image recipe follows the retained public CPU smoke workflow pins:
torch 2.6.0+cpu from the PyTorch CPU wheel index, transformers 4.57.3,
huggingface_hub 0.36.2 and lm-format-enforcer 0.11.3. The repository-compatible
base dependencies are additionally pinned. The public workflow is not proof
that this complete image builds or that a real chapter fits the candidate
four-core/16-GiB limits. Image construction fetches public code, dependencies
and pinned public model material; private invocation has network blocked.
The remote workspace constant represents caller-side authenticated workspace
verification, not independent remote workspace attestation. SDK transport
does not by itself establish any provider at-rest privacy guarantee.

Cleanup uses the same Python interpreter and pinned Modal package as the
launcher. The get timeout includes 120 seconds for platform startup and a
separate 180-second result/transport margin beyond the phase timeout.
The exclusive dispatch claim is permanent on failure or uncertainty; any
manual operator reset requires explicit new approval and reconciliation of
the original exact App/Call, never automatic retry.

Cleanup uses a 45-second monotonic observation/stop budget after workspace
verification, with bounded CLI calls and increasing backoff. A timed-out exact
stop is recorded as ambiguous; the same App is reobserved without resubmitting
the stop. Missing or malformed exact-App evidence remains unverified.
