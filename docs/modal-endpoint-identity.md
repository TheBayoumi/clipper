# Modal editorial endpoint identity: read-only research

Source: https://modal.com/docs/cli/latest/endpoint (consulted 21 November 2026).

The documented `modal endpoint create --model MODEL` option selects the base
Hugging Face model architecture. The CLI **does not document a revision-pinning
option for the base `--model`**. The similarly named `--custom-hf-revision`
is explicitly for `--custom-hf-repo`, not for the base model; it is therefore
not passed as if it pinned the editorial contract.

Modal documents `modal endpoint list --json` and
`modal endpoint info NAME --json` as read-only metadata commands. Its public
CLI reference does not specify the JSON field names or guarantee the deployed
base-model commit revision is exposed. All parsing tests are **synthetic
fixtures**, not captured workspace data; live compatibility remains UNVERIFIED.

The adapter only returns URLs when both pinned model ID and model revision match.
On missing or unfamiliar identity metadata it lists **field names only**, never
metadata values, and refuses reuse. If creation has already happened, it records
an actionable unverified-state error and never deletes or modifies the endpoint.

A future owner-authorized read-only check of actual listing shape is:

```sh
modal endpoint list --json
```

No Modal command was executed or endpoint created during this code change.
