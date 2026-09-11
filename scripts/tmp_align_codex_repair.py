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
    '_explicit_candidates(brief)',
    '_explicit_candidates(brief_path)',
)
text = text.replace(
    'campaign brief has no explicit authorized targets',
    'campaign contains no explicit authorized targets',
)

# Make the generated vision-cancellation branch explicit instead of swallowing
# the terminal cancellation exception with `pass`.
text = text.replace(
    '''            except Exception:
                pass
            self._instance_handle = None
            raise ModalRemoteError(''',
    '''            except Exception:
                self._instance_handle = None
            else:
                self._instance_handle = None
            raise ModalRemoteError(''',
)

# The resume regression fixture must use a real hexadecimal SHA-256 value.
text = text.replace(
    '"source_hashes": {"v1": "s" * 64},',
    '"source_hashes": {"v1": "b" * 64},',
)
text = text.replace(
    '"sha256": "s" * 64,',
    '"sha256": "b" * 64,',
)

# Replace the repair-script source-payload mutation with a pattern that matches
# the current runner without moving the existing budget-exhaustion guard.
start_marker = '''replace_once(
    "src/clipper/modal_execution.py",
    """    source_payloads = ['''
end_marker = '''replace_once(
    "src/clipper/modal_execution.py",
    """        "resume_from_run_id": resume_from_run_id,'''
start = text.find(start_marker)
end = text.find(end_marker, start + 1)
if start < 0 or end < 0:
    raise RuntimeError("temporary repair script source-payload mutation markers drifted")
replacement = '''replace_once(
    "src/clipper/modal_execution.py",
    """    source_payloads = [
        {
            "evidence": evidence,
            "video_id": candidate.video_id,
            "channel_id": candidate.channel_id,
            "canonical_url": candidate.url,
        }
        for candidate, evidence in zip(candidates, sources, strict=True)
    ]
""",
    """    source_payloads = [
        {
            "evidence": evidence,
            "video_id": candidate.video_id,
            "channel_id": candidate.channel_id,
            "canonical_url": candidate.url,
        }
        for candidate, evidence in zip(candidates, sources, strict=True)
    ]
    if resume_provenance is not None:
        expected_hashes = {
            str(key): str(value).lower()
            for key, value in dict(resume_provenance["source_hashes"]).items()
        }
        actual_hashes = {
            candidate.video_id: str(evidence.get("sha256") or "").lower()
            for candidate, evidence in zip(candidates, sources, strict=True)
        }
        if actual_hashes != expected_hashes:
            raise RuntimeError(
                "resume provenance source hashes do not match acquired source masters"
            )
""",
)
'''
text = text[:start] + replacement + text[end:]

path.write_text(text, encoding="utf-8")
