# Book-shot creative review contract

The creative review reads `required_on_screen_text` exclusively from the
content-bound episode plan. A `book_intro` shot must declare a list of
distinct non-blank strings or explicitly `[]` when no lettering is allowed.
For other shot types, omission means no text is required. The episode plan
builders derive intro title and author from source-backed book metadata.

## Existing approval compatibility

An approval binds the complete shot, plan fingerprint and text policy.
Adding an explicit intro declaration changes the shot and plan fingerprints,
**even if the printed title and author match the former implicit default**.
Old undeclared intro plans now fail closed. Existing approval records are not
rewritten, rehashed, migrated or silently reapproved. Any migration or re-review
is a separate owner decision. No inference, render, deployment or publishing
action is performed by this change.

## Creative review trust boundary

The gate checks that the review record is bound to the **exact video hash**,
**shot fingerprint**, **criterion versions**, and **observation record hash**,
and that a recorded human-approval entry exists with per-criterion decisions.
The `reviewer` field is a non-blank string; it is **not authentication**.
The file does not prove which person wrote or approved the record.

This is suitable for a **single-operator workflow** in which the repository
owner controls the filesystem and approval JSON. It does **not** defend against
a person who has write access to the review files: that person could replace
the approval entry and recompute its unkeyed hashes.

### OPEN OWNER DECISIONS — not implemented

- **A. Keep the single-operator trust boundary:** no additional infrastructure
  or operational cost; no protection against another filesystem writer.
- **B. Reviewer allowlist in committed configuration:** constrains accepted
  reviewer labels and adds configuration/maintenance work, but a writer can
  still impersonate an allowed name unless identity is checked independently.
- **C. Cryptographically signed approvals with an owner-controlled key:**
  supports verifying approval origin if the signing key is protected, at the
  cost of key management, rotation, recovery, and review tooling.

No allowlist, signing key, environment variable, schema field, or new identity
policy has been added. No inference, deployment or publishing was performed.
