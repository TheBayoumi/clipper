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
