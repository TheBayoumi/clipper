from __future__ import annotations

from pathlib import Path


path = Path("scripts/tmp_codex_findings_repair.py")
text = path.read_text(encoding="utf-8")

# Align the watchdog literal with its real indentation.
text = text.replace(
    '            "Modal spy did not observe a closed pipeline editorial-call set "\n'
    '            "before the production terminal barrier"',
    '                "Modal spy did not observe a closed pipeline editorial-call set "\n'
    '                "before the production terminal barrier"',
)

# Align run_modal_pipeline with the current explicit-target helper contract.
text = text.replace(
    '    ensure_modal_runtime()\n    candidates = _explicit_candidates(brief)\n',
    '    ensure_modal_runtime()\n\n    candidates = _explicit_candidates(brief_path)\n',
)
text = text.replace(
    'campaign brief has no explicit authorized targets',
    'campaign contains no explicit authorized targets',
)

# Current runner computes remaining budget before materializing source payloads.
old = '''    """    source_payloads = [
        {
            "evidence": evidence,
            "video_id": candidate.video_id,
            "channel_id": candidate.channel_id,
            "canonical_url": candidate.url,
        }
        for candidate, evidence in zip(candidates, sources, strict=True)
    ]
    remaining_gpu_seconds, remaining_estimated_usd = budget.remaining_budgets()
""",'''
new = '''    """    remaining_gpu_seconds, remaining_estimated_usd = budget.remaining_budgets()
    source_payloads = [
        {
            "evidence": evidence,
            "video_id": candidate.video_id,
            "channel_id": candidate.channel_id,
            "canonical_url": candidate.url,
        }
        for candidate, evidence in zip(candidates, sources, strict=True)
    ]
""",'''
if old not in text:
    raise RuntimeError("temporary repair script source-payload pattern drifted")
text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")
