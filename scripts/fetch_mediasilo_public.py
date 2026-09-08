from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse, urlunparse


MIN_MEDIA_BYTES = 1_000_000
MEDIA_SUFFIXES = {".mp4", ".mov", ".m4v", ".webm"}


def _redact_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _media_files(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.suffix.lower() in MEDIA_SUFFIXES
        and path.stat().st_size >= MIN_MEDIA_BYTES
    ]


def _run(command: list[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _try_ytdlp(url: str, output_dir: Path) -> Path | None:
    if shutil.which("yt-dlp") is None:
        return None

    template = str(output_dir / "source.%(ext)s")
    result = _run(
        [
            "yt-dlp",
            "--no-playlist",
            "--restrict-filenames",
            "--write-info-json",
            "--no-write-comments",
            "--no-write-thumbnail",
            "--output",
            template,
            url,
        ]
    )
    (output_dir / "yt-dlp.log").write_text(result.stdout, encoding="utf-8")
    files = _media_files(output_dir)
    return max(files, key=lambda path: path.stat().st_size) if files else None


def _cookie_header(cookies: list[dict[str, object]]) -> str:
    pairs: list[str] = []
    for cookie in cookies:
        name = cookie.get("name")
        value = cookie.get("value")
        if isinstance(name, str) and isinstance(value, str):
            pairs.append(f"{name}={value}")
    return "; ".join(pairs)


def _download_with_curl(
    url: str,
    output_path: Path,
    *,
    referer: str,
    user_agent: str,
    cookie_header: str,
) -> bool:
    command = ["curl", "--fail", "--location", "--retry", "3", "--output", str(output_path)]
    command.extend(["--header", f"Referer: {referer}"])
    command.extend(["--user-agent", user_agent])
    if cookie_header:
        command.extend(["--header", f"Cookie: {cookie_header}"])
    command.append(url)
    result = _run(command)
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        return False
    return output_path.exists() and output_path.stat().st_size >= MIN_MEDIA_BYTES


def _download_hls(
    url: str,
    output_path: Path,
    *,
    referer: str,
    user_agent: str,
    cookie_header: str,
) -> bool:
    header_lines = [f"Referer: {referer}", f"User-Agent: {user_agent}"]
    if cookie_header:
        header_lines.append(f"Cookie: {cookie_header}")
    headers = "\r\n".join(header_lines) + "\r\n"
    result = _run(
        [
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "warning",
            "-headers",
            headers,
            "-i",
            url,
            "-map",
            "0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            "-y",
            str(output_path),
        ]
    )
    (output_path.parent / "ffmpeg.log").write_text(result.stdout, encoding="utf-8")
    if result.returncode != 0:
        output_path.unlink(missing_ok=True)
        return False
    return output_path.exists() and output_path.stat().st_size >= MIN_MEDIA_BYTES


def _try_playwright(url: str, output_dir: Path) -> Path | None:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
    from playwright.sync_api import sync_playwright

    observed: list[dict[str, object]] = []
    candidate_urls: set[str] = set()
    user_agent = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/140.0.0.0 Safari/537.36"
    )

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True, user_agent=user_agent)
        page = context.new_page()

        def record_response(response: object) -> None:
            response_url = getattr(response, "url", "")
            if not isinstance(response_url, str):
                return
            try:
                headers = getattr(response, "headers", {})
                content_type = headers.get("content-type", "") if isinstance(headers, dict) else ""
                status = int(getattr(response, "status", 0))
            except Exception:
                content_type = ""
                status = 0
            observed.append(
                {
                    "url": _redact_url(response_url),
                    "status": status,
                    "content_type": content_type,
                }
            )
            lowered = response_url.lower()
            if (
                ".m3u8" in lowered
                or ".mp4" in lowered
                or "video/mp4" in str(content_type).lower()
                or "application/vnd.apple.mpegurl" in str(content_type).lower()
            ):
                candidate_urls.add(response_url)

        page.on("response", record_response)
        page.goto(url, wait_until="domcontentloaded", timeout=60_000)
        page.wait_for_timeout(6_000)

        for label in ("Accept", "Accept All", "I Agree", "Got it"):
            locator = page.get_by_role("button", name=re.compile(f"^{re.escape(label)}$", re.I))
            if locator.count():
                try:
                    locator.first.click(timeout=2_000)
                except Exception:
                    pass

        try:
            page.locator("video").evaluate_all(
                "els => els.forEach(v => {v.muted = true; v.play().catch(() => {});})"
            )
        except Exception:
            pass

        for role_name in ("Play", "play"):
            locator = page.get_by_role("button", name=re.compile(role_name, re.I))
            if locator.count():
                try:
                    locator.first.click(timeout=2_000)
                    break
                except Exception:
                    pass

        page.wait_for_timeout(5_000)
        page.screenshot(path=str(output_dir / "page.png"), full_page=True)
        (output_dir / "page.html").write_text(page.content(), encoding="utf-8")

        resource_urls = page.evaluate(
            "() => performance.getEntriesByType('resource').map(entry => entry.name)"
        )
        if isinstance(resource_urls, list):
            for resource_url in resource_urls:
                if not isinstance(resource_url, str):
                    continue
                lowered = resource_url.lower()
                if ".m3u8" in lowered or ".mp4" in lowered:
                    candidate_urls.add(resource_url)

        video_urls = page.locator("video").evaluate_all(
            "els => els.flatMap(v => [v.currentSrc, v.src, ...Array.from(v.querySelectorAll('source')).map(s => s.src)]).filter(Boolean)"
        )
        if isinstance(video_urls, list):
            candidate_urls.update(item for item in video_urls if isinstance(item, str))

        download_locators = [
            page.get_by_role("button", name=re.compile("download", re.I)),
            page.get_by_role("link", name=re.compile("download", re.I)),
            page.locator("[aria-label*='download' i]"),
            page.locator("[title*='download' i]"),
        ]
        for locator in download_locators:
            count = min(locator.count(), 8)
            for index in range(count):
                try:
                    with page.expect_download(timeout=8_000) as download_info:
                        locator.nth(index).click(timeout=3_000)
                    download = download_info.value
                    suggested = Path(download.suggested_filename).name or "mediasilo-download.bin"
                    destination = output_dir / suggested
                    download.save_as(str(destination))
                    if destination.stat().st_size >= MIN_MEDIA_BYTES:
                        browser.close()
                        return destination
                    destination.unlink(missing_ok=True)
                except PlaywrightTimeoutError:
                    continue
                except Exception:
                    continue

        cookies = context.cookies()
        cookie_header = _cookie_header(cookies)
        (output_dir / "network-redacted.json").write_text(
            json.dumps(observed, indent=2), encoding="utf-8"
        )

        ranked = sorted(
            candidate_urls,
            key=lambda item: (
                0 if ".mp4" in item.lower() else 1,
                0 if ".m3u8" in item.lower() else 1,
                len(item),
            ),
        )
        for index, candidate in enumerate(ranked):
            if candidate.startswith("blob:"):
                continue
            if ".m3u8" in candidate.lower():
                destination = output_dir / f"source-hls-{index}.mp4"
                if _download_hls(
                    candidate,
                    destination,
                    referer=url,
                    user_agent=user_agent,
                    cookie_header=cookie_header,
                ):
                    browser.close()
                    return destination
            else:
                suffix = Path(urlparse(candidate).path).suffix.lower()
                if suffix not in MEDIA_SUFFIXES:
                    suffix = ".mp4"
                destination = output_dir / f"source-direct-{index}{suffix}"
                if _download_with_curl(
                    candidate,
                    destination,
                    referer=url,
                    user_agent=user_agent,
                    cookie_header=cookie_header,
                ):
                    browser.close()
                    return destination

        browser.close()
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch authorized public MediaSilo review media.")
    parser.add_argument("url")
    parser.add_argument("--output-dir", type=Path, default=Path("mediasilo-output"))
    args = parser.parse_args()

    parsed = urlparse(args.url)
    if parsed.scheme != "https" or not parsed.netloc.endswith("mediasilo.com"):
        parser.error("url must be an HTTPS mediasilo.com review URL")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    media = _try_ytdlp(args.url, output_dir)
    if media is None:
        media = _try_playwright(args.url, output_dir)

    if media is None:
        print("No downloadable media was materialized; inspect the uploaded diagnostics.", file=sys.stderr)
        return 2

    final_path = output_dir / f"campaign-source{media.suffix.lower()}"
    if media != final_path:
        media.replace(final_path)

    metadata = {
        "source_review_url": args.url,
        "downloaded_file": final_path.name,
        "bytes": final_path.stat().st_size,
    }
    (output_dir / "result.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
