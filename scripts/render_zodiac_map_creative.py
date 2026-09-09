from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import urllib.request
from pathlib import Path

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/140.0.0.0 Safari/537.36"
)
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
WIDTH = 1080
HEIGHT = 1920
FPS = 60


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def load_assets(path: Path) -> list[dict[str, object]]:
    bodies = json.loads(path.read_text())
    assets: list[dict[str, object]] = []
    seen: set[str] = set()
    for body in bodies.values():
        if not isinstance(body, list):
            continue
        for item in body:
            if not isinstance(item, dict):
                continue
            asset_id = item.get("id")
            if not isinstance(asset_id, str) or asset_id in seen:
                continue
            seen.add(asset_id)
            assets.append(item)
    return assets


def derivative_url(item: dict[str, object], kind: str = "source") -> str:
    for derivative in item.get("derivatives") or []:
        if not isinstance(derivative, dict) or derivative.get("type") != kind:
            continue
        url = derivative.get("downloadUrlWithAssetTitle") or derivative.get("url")
        if isinstance(url, str) and url.startswith("https://"):
            return url
    raise RuntimeError(f"No {kind} URL for {item.get('fileName') or item.get('title')}")


def pick_asset(path: Path, asset_type: str, needle: str) -> tuple[str, str]:
    for item in load_assets(path):
        name = str(item.get("fileName") or item.get("title") or "")
        if item.get("type") == asset_type and needle.lower() in name.lower():
            return name, derivative_url(item, "source")
    raise RuntimeError(f"Could not find {asset_type} containing {needle!r} in {path}")


def download(url: str, destination: Path) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": UA})
    digest = hashlib.sha256()
    size = 0
    with urllib.request.urlopen(request, timeout=240) as response, destination.open("wb") as output:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            output.write(block)
            digest.update(block)
            size += len(block)
    if size < 100_000:
        raise RuntimeError(f"Suspiciously small download: {destination} ({size} bytes)")
    return {"bytes": size, "sha256": digest.hexdigest()}


def video_filters(text: str | None = None, font_size: int = 56, y: str = "180") -> str:
    base = f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2,fps={FPS},format=yuv420p"
    if not text:
        return base
    escaped = text.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return (
        base
        + f",drawtext=fontfile='{FONT}':text='{escaped}':fontcolor=white:fontsize={font_size}:"
          f"borderw=4:bordercolor=black@0.95:box=1:boxcolor=black@0.30:boxborderw=14:"
          f"x=(w-text_w)/2:y={y}"
    )


def render_video_segment(
    source: Path,
    start: float,
    duration: float,
    destination: Path,
    text: str | None = None,
    font_size: int = 56,
    y: str = "180",
) -> None:
    vf = video_filters(text=text, font_size=font_size, y=y)
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "0:a:0", "-vf", vf,
        "-af", "aresample=48000,aformat=channel_layouts=stereo",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-profile:v", "high", "-level", "4.2",
        "-r", str(FPS), "-fps_mode", "cfr", "-g", "120", "-keyint_min", "60", "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(destination),
    ])


def render_tac_segment(tac: Path, audio_source: Path, destination: Path) -> None:
    # Preserve the full horizontal Tac Map over a full-frame blurred fill, then add a slight digital push.
    # Audio is the official flythrough clip audio; no music/SFX is added externally.
    filter_complex = (
        f"[0:v]split=2[bg0][fg0];"
        f"[bg0]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},gblur=sigma=28,eq=brightness=-0.18[bg];"
        f"[fg0]scale=1020:-1[fg];"
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2[comp];"
        f"[comp]zoompan=z='min(zoom+0.0015,1.045)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"drawtext=fontfile='{FONT}':text='NEW WARZONE MAP?':fontcolor=white:fontsize=64:"
        f"borderw=4:bordercolor=black@0.95:box=1:boxcolor=black@0.30:boxborderw=16:"
        f"x=(w-text_w)/2:y=180,format=yuv420p[v]"
    )
    run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(tac), "-ss", "0.000", "-i", str(audio_source), "-t", "0.600",
        "-filter_complex", filter_complex, "-map", "[v]", "-map", "1:a:0",
        "-af", "aresample=48000,aformat=channel_layouts=stereo",
        "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-profile:v", "high", "-level", "4.2",
        "-r", str(FPS), "-fps_mode", "cfr", "-g", "120", "-keyint_min", "60", "-sc_threshold", "0",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(destination),
    ])


def concat_segments(segments: list[Path], destination: Path) -> None:
    manifest = destination.with_suffix(".concat.txt")
    manifest.write_text("".join(f"file '{segment.resolve()}'\n" for segment in segments))
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
        "-c", "copy", "-movflags", "+faststart", str(destination),
    ])


def probe(path: Path) -> dict[str, object]:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,sample_rate,channels",
            "-of", "json", str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-bodies", type=Path, required=True)
    parser.add_argument("--gameplay-bodies", type=Path, required=True)
    parser.add_argument("--tac-bodies", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("zodiac-map-render"))
    parser.add_argument("--output", type=Path, default=Path("Zodiac_NEW_WARZONE_MAP_FINAL.mp4"))
    args = parser.parse_args()

    work = args.work_dir.resolve()
    assets = work / "assets"
    segments_dir = work / "segments"
    assets.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    map_name, map_url = pick_asset(args.map_bodies, "video", "9x16")
    gameplay_name, gameplay_url = pick_asset(args.gameplay_bodies, "video", "9x16")
    tac_name, tac_url = pick_asset(args.tac_bodies, "image", "Tac_Map")

    map_file = assets / "map_flythrough_9x16.mp4"
    gameplay_file = assets / "gameplay_9x16.mp4"
    tac_file = assets / "tac_map.jpg"

    downloads = {
        "map": {"name": map_name, **download(map_url, map_file)},
        "gameplay": {"name": gameplay_name, **download(gameplay_url, gameplay_file)},
        "tac": {"name": tac_name, **download(tac_url, tac_file)},
    }

    # Creative intent: a distinct map-first test. Hard cuts only. Official source audio only.
    timeline = [
        {"id": "S0", "kind": "tac", "source": tac_name, "source_start": None, "duration": 0.60, "final": [0.00, 0.60], "text": "NEW WARZONE MAP?"},
        {"id": "S1", "kind": "map", "source": map_name, "source_start": 0.60, "duration": 1.00, "final": [0.60, 1.60], "text": "ZODIAC - FREE BETA WEEKEND 2"},
        {"id": "S2", "kind": "map", "source": map_name, "source_start": 24.15, "duration": 1.40, "final": [1.60, 3.00], "text": None},
        {"id": "S3", "kind": "gameplay", "source": gameplay_name, "source_start": 11.50, "duration": 1.60, "final": [3.00, 4.60], "text": None},
        {"id": "S4a", "kind": "map", "source": map_name, "source_start": 31.00, "duration": 0.70, "final": [4.60, 5.30], "text": None},
        {"id": "S4b", "kind": "map", "source": map_name, "source_start": 33.55, "duration": 0.70, "final": [5.30, 6.00], "text": None},
        {"id": "S4c", "kind": "map", "source": map_name, "source_start": 35.05, "duration": 0.70, "final": [6.00, 6.70], "text": None},
        {"id": "S5", "kind": "gameplay", "source": gameplay_name, "source_start": 14.50, "duration": 2.20, "final": [6.70, 8.90], "text": None},
        {"id": "S6", "kind": "cta", "source": gameplay_name, "source_start": 20.90, "duration": 1.70, "final": [8.90, 10.60], "text": None},
    ]

    rendered: list[Path] = []
    for item in timeline:
        destination = segments_dir / f"{item['id']}.mp4"
        if item["kind"] == "tac":
            render_tac_segment(tac_file, map_file, destination)
        else:
            source = map_file if item["kind"] == "map" else gameplay_file
            render_video_segment(
                source,
                float(item["source_start"]),
                float(item["duration"]),
                destination,
                text=item.get("text"),
                font_size=48 if item.get("text") else 56,
                y="1450" if item.get("text") else "180",
            )
        rendered.append(destination)

    output = args.output.resolve()
    concat_segments(rendered, output)

    final_probe = probe(output)
    report = {
        "creative": "NEW WARZONE MAP?",
        "timeline": timeline,
        "downloads": downloads,
        "probe": final_probe,
        "requirements": {
            "official_assets_only": True,
            "added_music": False,
            "added_sfx": False,
            "ai_voice": False,
            "target_resolution": "1080x1920",
            "target_fps": 60,
            "target_duration_seconds": 10.6,
            "cta_seconds": 1.7,
        },
    }
    (work / "render-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
