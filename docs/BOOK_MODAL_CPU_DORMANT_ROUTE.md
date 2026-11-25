# Scoped Modal public-synthetic CPU rehearsal — staged, not activated

## Authority and boundaries

A narrowly scoped owner request calls for one **model-free public-synthetic
dry run before any model or chapter processing**, using existing included
credits only, with **paid overage not authorized**. The explicit provenance
references stored in a temporary approval are `2026-11-25T11:09:05+02:00` and
`2026-11-25T15:00:33+02:00`. They are stable operational labels, **not** independent
authentication, account-spend enforcement, or a guarantee that an included
credit balance remains. No fabricated positive hard-dollar cap is recorded.

The only expected workspace is `mahmoud-bayoumi-mm`. The only live credential
route is the existing unsuffixed Actions `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET` pair, with `MODAL_PROFILE` absent. Do not create new
tokens, Google credentials, secret/volume resources, or broader connection
checks. The approved operation is
`PUBLIC_SYNTHETIC_MODEL_FREE_BOOK_REHEARSAL` only—never private books,
model inference, deployment, publishing, or paid overage.

## Dormant one-shot workflow (no activation file committed)

`.github/workflows/modal-public-synthetic.yml` responds only to a push on
`review/unified-tts-cleanup` adding
`acceptance/modal-public-synthetic/request.json`. It does **not**
respond to its own source changes, pull requests, schedules, or
workflow_dispatch. The activation request must be a **single commit
adding exactly that one file**, with no corresponding file in its parent.
The later request must contain exactly:

```json
{
  "schema": "clipper-modal-public-synthetic-request-v1",
  "request_id": "public-synthetic1502",
  "expected_workspace": "mahmoud-bayoumi-mm",
  "approved_stage_sha": "<exact reviewed staging HEAD>"
}
```

The gate rejects an unexpected parent or file delta. It is evaluated
before dependency installation or secrets. Approval is generated **at
runtime** outside the checkout using the verified activation HEAD as
`source_sha`, so the repository contains no self-referential approval
manifest. The generated approval has exact schema
`clipper-modal-public-synthetic-approval-v2`, intent, workspace,
environment, Actions credential route, <=1-hour UTC expiration,
`included_credit_only=true`, `paid_overage_authorized=false`,
the two provenance references, execution permission, and a separate
lifecycle receipt. SHA-256 is computed over the identical approval bytes
which the launcher later parses. This is a narrow record of approval, **not
cryptographic authentication**.

Python is 3.11 and Modal SDK is exactly 1.5.5; the workflow has one
5-minute job, installs only the SDK without credentials, and scopes
existing tokens to the execution step. It uses `timeout` with TERM
then a 10-second KILL escalation, once only. No raw worker logs, private
content, environment dumps, account details, or uploads are emitted.

## Existing limited launcher path

The default CLI remains an inert JSON plan. The bounded parent revalidates
clean full-SHA local checkout, exact approval bytes/digest, strict field set,
unexpired authority, selected Actions token pair and lifecycle receipt.
It creates a temporary journal and single-use worker capability.
The internal worker rechecks all input constraints before SDK access,
performs the client-bound workspace lookup **within** the bounded
control-plane window, requires an exact workspace match and verifies
approval expiry **after** identity but before any App/Image construction.

The historical one-shot attempt reached remote fixture output packaging and
failed on SDK 1.5.5 BlobCreate with AuthError/HTTP 401. It established an
oversized serialized return that crossed the 8,192-byte inline threshold;
the provider's exact authorization reason and any causal effect of the
security flags are **unproven**. The public fixture's offline human-review
pause and separately labeled synthetic continuation are independently
validated, but the live end-to-end rehearsal is **not** validated.

The offline correction now validates the **full synthetic report inside the
remote Function**, then returns only a fixed-schema compact DTO before the
Modal SDK can serialize the Function result. The original 97-entry path-keyed
SHA-256 inventory remains internal to the Function; its canonical sorted,
compact JSON is validated with the existing 512-entry/64-KiB limits and
represented outside the Function only by its SHA-256 digest and count.
The DTO includes exact source SHA/pin, status, both fixed track statuses,
false publishing permission, empty guard attempts and strict zero model/
network/publication counts. The worker, parent and CLI independently
validate the same exact DTO keys and types; no arbitrary nested fields,
artifact paths, source excerpts, raw logs or secrets may be published.

The bounded worker journals a safe exception class, last known phase,
numeric HTTP status when available, and bounded synthetic child exit code.
These are preserved alongside exact App ID and independently observed cleanup
state on failure, without exposing exception messages or raw stderr.
An offline protocol-4 pickle regression checks below 8,192 bytes; **only a
separate isolated Modal SDK 1.5.5 serializer probe can validate the actual
SDK wire encoding**, after public fixture generation to preserve the guard.
No runtime attempt, provider authentication or resource creation is
authorized by this code correction.

The existing Function configuration remains exactly:
`cpu=(1,1)`, `memory=(512,512)`, `gpu=None`,
`block_network=True`, `restrict_modal_access=True`,
`include_source=False`, `single_use_containers=True`,
`max_containers=1`, `min_containers=0`,
`buffer_containers=0`, `retries=0`,
`timeout=30`, `startup_timeout=60`, and
`scaledown_window=2`. The image uses an anonymous public source
checkout, pinned to the exact source SHA. The accepted synthetic
rehearsal maintains a real human-review pause and a separately labeled
synthetic test continuation, with model/network/publish call guards.
It does **not** touch private chapters or production drivers.

The control-plane parent allows 120 seconds, including source preparation,
identity, App/context, call and stop/list reconciliation. If the worker
stalls, it kills the worker process group and attempts bounded cleanup.
App IDs are considered actionable only with `ap-` plus 22 ASCII
alphanumeric characters, including recovered journal entries. App
creation during context entry may occur before its ID is known; such a
case is **UNKNOWN/BLOCKED**, not proof of absence. Cancel occurs inside
the App context before a possibly blocking exit. KeyboardInterrupt
retains interrupted evidence and attempts bounded reconciliation.

Pinned Modal 1.5.5 provides `modal app list --json --env ENV` and
`modal app stop -y --env ENV EXACT_APP_ID`. The parent lists first:
a positive exact-ID `stopped` row with zero tasks skips stop.
Otherwise it stops only that validated exact ID and verifies with a new
bounded list. Errors, missing IDs, malformed rows and timeouts are
retained, never converted into success. A clean terminal observation
is not proof of universal resource deletion or account billing closure.

The currently staged workflow gate does not independently pin the reviewed
functional base: a later request could echo its arbitrary parent. **Do not
activate the existing workflow.** After this functional commit is reviewed,
stage a separate workflow-only commit whose gate pins the literal reviewed
functional commit, checks that the intervening commit changes only its
workflow file, and checks that the activation commit adds only its request
file. No activation request is included in this correction.

## Current disposition

**OFFLINE CORRECTION ONLY — NOT ACTIVATED.** Historical run
45627938078 failed; the exact App was subsequently confirmed stopped
with zero tasks (45628442604), and all temporary trial/reconcile/diagnostic
workflows plus the activation request were removed. Retained security flags
include restricted Modal access, blocked networking, no source mounting and
the unchanged single-container/zero-retry limits. The earlier BlobCreate
AuthError does not prove which authorization rule was responsible.

The unchanged CI 95% gate, exact-SHA Ruff/tests, independent code review
and a separately authorized future activation are required. Existing credit
assertions do not guarantee provider billing limits. No paid overage is
authorized.
