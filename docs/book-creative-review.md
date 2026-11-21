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
