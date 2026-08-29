#!/usr/bin/env python3
"""Fail closed unless Codex has reviewed the exact PR head with no unresolved P0-P2."""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
from typing import Any

CODEX_LOGIN = "chatgpt-codex-connector"
BLOCKING_SEVERITY_RE = re.compile(r"\bP[012]\b")
REVIEWED_COMMIT_RE = re.compile(r"Reviewed commit:\*\*\s*`([0-9a-f]{7,40})`", re.IGNORECASE)


def _request_json(url: str, token: str, *, payload: dict[str, Any] | None = None) -> Any:
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


def _is_blocking_codex_thread(thread: dict[str, Any]) -> bool:
    if bool(thread.get("isResolved")):
        return False
    nodes = ((thread.get("comments") or {}).get("nodes") or [])
    for comment in nodes:
        if str((comment.get("author") or {}).get("login") or "") != CODEX_LOGIN:
            continue
        if BLOCKING_SEVERITY_RE.search(str(comment.get("body") or "")):
            return True
    return False


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


def _load_event_pr_number() -> int:
    event_path = os.environ.get("GITHUB_EVENT_PATH", "")
    if not event_path:
        raise RuntimeError("GITHUB_EVENT_PATH is required")
    with open(event_path, encoding="utf-8") as handle:
        event = json.load(handle)
    number = event.get("number") or (event.get("pull_request") or {}).get("number")
    if not isinstance(number, int) or number <= 0:
        raise RuntimeError("Codex review gate requires a pull-request event")
    return number


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
            "variables": {"owner": owner, "name": name, "number": pr_number, "cursor": cursor},
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


def main() -> int:
    repo = os.environ.get("GITHUB_REPOSITORY", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    if not repo or "/" not in repo:
        raise RuntimeError("GITHUB_REPOSITORY must be owner/repo")
    if not token:
        raise RuntimeError("GITHUB_TOKEN is required")
    pr_number = _load_event_pr_number()

    pr = _request_json(_repo_api(repo, f"pulls/{pr_number}"), token)
    head_sha = str(((pr.get("head") or {}).get("sha")) or "").lower()
    if not re.fullmatch(r"[0-9a-f]{40}", head_sha):
        raise RuntimeError(f"PR head SHA is invalid: {head_sha!r}")

    reviews = _request_json(_repo_api(repo, f"pulls/{pr_number}/reviews?per_page=100"), token)
    has_exact_review = any(_review_matches_head(review, head_sha) for review in reviews)

    if not has_exact_review:
        comments = _request_json(_repo_api(repo, f"issues/{pr_number}/comments?per_page=100"), token)
        for comment in comments:
            body = str(comment.get("body") or "")
            if "@codex review" not in body.lower() or head_sha not in body.lower():
                continue
            reactions_url = str(comment.get("reactions", {}).get("url") or "")
            if not reactions_url:
                comment_id = int(comment["id"])
                reactions_url = _repo_api(repo, f"issues/comments/{comment_id}/reactions?per_page=100")
            reactions = _request_json(reactions_url, token)
            if _clean_reaction_matches_head(comment, reactions, head_sha):
                has_exact_review = True
                break

    if not has_exact_review:
        print(
            f"::error::Codex has not completed a review of exact PR head {head_sha}. "
            f"Comment '@codex review' with this exact SHA and rerun this failed gate after Codex responds."
        )
        return 1

    threads = _fetch_unresolved_threads(repo, pr_number, token)
    blocking = [thread for thread in threads if _is_blocking_codex_thread(thread)]
    if blocking:
        print(
            f"::error::{len(blocking)} unresolved Codex P0/P1/P2 review thread(s) remain on PR #{pr_number}."
        )
        return 1

    print(f"Codex review gate PASS: exact head {head_sha}; no unresolved Codex P0/P1/P2 threads.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"::error::Codex review gate failed closed: {exc}")
        raise SystemExit(1) from exc
