#!/usr/bin/env python3
"""Fail closed unless Codex has reviewed the exact PR head with no unresolved P0-P2."""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from typing import Any
from urllib.parse import urlparse

CODEX_LOGIN = "chatgpt-codex-connector"
BLOCKING_SEVERITY_RE = re.compile(r"\bP[012]\b")
REVIEWED_COMMIT_RE = re.compile(r"Reviewed commit:\*\*\s*`([0-9a-f]{7,40})`", re.IGNORECASE)


def _validate_github_api_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "api.github.com":
        raise RuntimeError(f"refusing non-GitHub API URL: {url!r}")


def _request_json(url: str, token: str, *, payload: dict[str, Any] | None = None) -> Any:
    _validate_github_api_url(url)
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data)
    request.add_header("Accept", "application/vnd.github+json")
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("X-GitHub-Api-Version", "2022-11-28")
    if payload is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return json.load(response)


def _repo_api(repo: str, path: str) -> str:
    return f"https://api.github.com/repos/{repo}/{path.lstrip('/')}"


def _review_matches_head(review: dict[str, Any], head_sha: str) -> bool:
    if str((review.get("user") or {}).get("login") or "") != CODEX_LOGIN:
        return False
    commit_id = str(review.get("commit_id") or "")
    if commit_id:
        return commit_id == head_sha
    match = REVIEWED_COMMIT_RE.search(str(review.get("body") or ""))
    return bool(match and head_sha.startswith(match.group(1).lower()))


def _is_blocking_codex_thread(
    thread: dict[str, Any],
    *,
    allow_runtime_evidence: bool = False,
) -> bool:
    if bool(thread.get("isResolved")):
        return False
    nodes = ((thread.get("comments") or {}).get("nodes") or [])
    blocking_bodies: list[str] = []
    for comment in nodes:
        if str((comment.get("author") or {}).get("login") or "") != CODEX_LOGIN:
            continue
        body = str(comment.get("body") or "")
        if BLOCKING_SEVERITY_RE.search(body):
            blocking_bodies.append(body)
    if not blocking_bodies:
        return False
    if allow_runtime_evidence and all(
        "NEEDS_RUNTIME_EVIDENCE" in body for body in blocking_bodies
    ):
        return False
    return True


def _is_deferred_runtime_evidence_thread(thread: dict[str, Any]) -> bool:
    return not _is_blocking_codex_thread(
        thread,
        allow_runtime_evidence=True,
    ) and _is_blocking_codex_thread(thread)


def _clean_reaction_matches_head(
    comment: dict[str, Any], reactions: list[dict[str, Any]], head_sha: str
) -> bool:
    body = str(comment.get("body") or "")
    if "@codex review" not in body.lower() or head_sha not in body.lower():
        return False
    return any(
        str((reaction.get("user") or {}).get("login") or "") == CODEX_LOGIN
        and str(reaction.get("content") or "") == "+1"
        for reaction in reactions
    )


def _load_event_pr_context() -> tuple[int | None, str | None]:
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path:
        return None, None
    with open(event_path, encoding="utf-8") as handle:
        event = json.load(handle)
    pull_request = event.get("pull_request") or {}
    number = event.get("number") or pull_request.get("number")
    raw_head = (pull_request.get("head") or {}).get("sha")
    pr_number = number if isinstance(number, int) and number > 0 else None
    head_sha = str(raw_head or "").lower() or None
    return pr_number, head_sha


def _select_pr_number_for_head(pulls: list[dict[str, Any]], head_sha: str) -> int:
    matches = []
    for pull in pulls:
        if str(((pull.get("head") or {}).get("sha")) or "").lower() != head_sha:
            continue
        number = pull.get("number")
        if isinstance(number, int) and number > 0:
            matches.append(number)
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(
            "exact-head Codex gate requires exactly one pull request: "
            f"head={head_sha} matches={unique}"
        )
    return unique[0]


def _resolve_pr_number(
    repo: str,
    token: str,
    *,
    head_sha: str,
    explicit_pr_number: int | None,
    event_pr_number: int | None,
) -> int:
    if explicit_pr_number is not None:
        if explicit_pr_number <= 0:
            raise ValueError("--pr-number must be positive")
        return explicit_pr_number
    if event_pr_number is not None:
        return event_pr_number
    pulls = _request_json(
        _repo_api(repo, f"commits/{head_sha}/pulls?per_page=100"),
        token,
    )
    if not isinstance(pulls, list):
        raise RuntimeError("GitHub commit-pulls lookup did not return an array")
    return _select_pr_number_for_head(pulls, head_sha)


def _fetch_unresolved_threads(repo: str, pr_number: int, token: str) -> list[dict[str, Any]]:
    owner, name = repo.split("/", 1)
    query = """
    query($owner:String!, $name:String!, $number:Int!, $cursor:String) {
      repository(owner:$owner, name:$name) {
        pullRequest(number:$number) {
          reviewThreads(first:100, after:$cursor) {
            pageInfo { hasNextPage endCursor }
            nodes {
              isResolved
              comments(first:100) { nodes { body author { login } } }
            }
          }
        }
      }
    }
    """
    cursor: str | None = None
    threads: list[dict[str, Any]] = []
    while True:
        payload = {
            "query": query,
            "variables": {
                "owner": owner,
                "name": name,
                "number": pr_number,
                "cursor": cursor,
            },
        }
        result = _request_json("https://api.github.com/graphql", token, payload=payload)
        if result.get("errors"):
            raise RuntimeError(f"GitHub GraphQL review-thread query failed: {result['errors']}")
        connection = result["data"]["repository"]["pullRequest"]["reviewThreads"]
        threads.extend(connection.get("nodes") or [])
        page_info = connection["pageInfo"]
        if not page_info.get("hasNextPage"):
            return threads
        cursor = str(page_info.get("endCursor") or "")
        if not cursor:
            raise RuntimeError("GitHub review-thread pagination returned no cursor")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head-sha")
    parser.add_argument("--pr-number", type=int)
    parser.add_argument(
        "--allow-runtime-evidence",
        action="store_true",
        help=(
            "Static preflight only: allow unresolved P0-P2 threads when every blocking Codex "
            "comment in the thread is explicitly marked NEEDS_RUNTIME_EVIDENCE."
        ),
    )
    args = parser.parse_args(argv)

    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not repo or "/" not in repo:
        raise RuntimeError("GITHUB_REPOSITORY must be owner/repo")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required")

    event_pr_number, event_head_sha = _load_event_pr_context()
    requested_head_sha = str(args.head_sha or event_head_sha or "").lower()
    if not re.fullmatch(r"[0-9a-f]{40}", requested_head_sha):
        raise RuntimeError(
            "Codex review gate requires an exact 40-character head SHA via "
            "--head-sha or pull-request event"
        )
    pr_number = _resolve_pr_number(
        repo,
        token,
        head_sha=requested_head_sha,
        explicit_pr_number=args.pr_number,
        event_pr_number=event_pr_number,
    )

    pr = _request_json(_repo_api(repo, f"pulls/{pr_number}"), token)
    head_sha = str(((pr.get("head") or {}).get("sha")) or "").lower()
    if head_sha != requested_head_sha:
        raise RuntimeError(
            "pull-request head moved away from the reviewed exact head: "
            f"requested={requested_head_sha} actual={head_sha}"
        )

    reviews = _request_json(
        _repo_api(repo, f"pulls/{pr_number}/reviews?per_page=100"),
        token,
    )
    has_exact_review = any(_review_matches_head(review, head_sha) for review in reviews)

    if not has_exact_review:
        comments = _request_json(
            _repo_api(repo, f"issues/{pr_number}/comments?per_page=100"),
            token,
        )
        for comment in comments:
            body = str(comment.get("body") or "")
            if "@codex review" not in body.lower() or head_sha not in body.lower():
                continue
            reactions_url = str(comment.get("reactions", {}).get("url") or "")
            if not reactions_url:
                comment_id = int(comment["id"])
                reactions_url = _repo_api(
                    repo,
                    f"issues/comments/{comment_id}/reactions?per_page=100",
                )
            reactions = _request_json(reactions_url, token)
            if _clean_reaction_matches_head(comment, reactions, head_sha):
                has_exact_review = True
                break

    if not has_exact_review:
        print(
            f"::error::Codex has not completed a review of exact PR head {head_sha}. "
            "Comment '@codex review' with this exact SHA and rerun this failed gate "
            "after Codex responds."
        )
        return 1

    threads = _fetch_unresolved_threads(repo, pr_number, token)
    blocking = [
        thread
        for thread in threads
        if _is_blocking_codex_thread(
            thread,
            allow_runtime_evidence=args.allow_runtime_evidence,
        )
    ]
    deferred = [
        thread for thread in threads if _is_deferred_runtime_evidence_thread(thread)
    ]
    if blocking:
        mode = "static preflight" if args.allow_runtime_evidence else "final"
        print(
            f"::error::{len(blocking)} unresolved Codex P0/P1/P2 review thread(s) "
            f"remain on PR #{pr_number} for {mode} gate."
        )
        return 1

    if args.allow_runtime_evidence:
        print(
            f"Codex static preflight PASS: exact head {head_sha}; "
            f"deferred_runtime_threads={len(deferred)}; no unresolved static P0/P1/P2."
        )
    else:
        print(
            f"Codex review gate PASS: exact head {head_sha}; "
            "no unresolved Codex P0/P1/P2 threads."
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"::error::Codex review gate failed closed: {exc}")
        raise SystemExit(1) from exc
