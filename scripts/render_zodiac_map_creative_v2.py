from __future__ import annotations

import argparse
import hashlib
import json
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
VIDEO_BITRATE = "250M"  # Matches CapCut screenshot: 250000 Kbps CBR.


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
    candidates: list[str] = []
    for item in load_assets(path):
        name = str(item.get("fileName") or item.get("title") or "")
        if item.get("type") == asset_type:
            candidates.append(name)
            if needle.lower() in name.lower():
                return name, derivative_url(item, "source")
    raise RuntimeError(
        f"Could not find {asset_type} containing {needle!r} in {path}; candidates={candidates}"
    )


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


def x264_args() -> list[str]:
    return [
        "-c:v", "libx264",
        "-preset", "slow",
        "-profile:v", "high",
        "-level:v", "5.2",
        "-b:v", VIDEO_BITRATE,
        "-minrate", VIDEO_BITRATE,
        "-maxrate", VIDEO_BITRATE,
        "-bufsize", "500M",
        "-x264-params", "nal-hrd=cbr:force-cfr=1",
        "-r", str(FPS),
        "-fps_mode", "cfr",
        "-g", "120",
        "-keyint_min", "60",
        "-sc_threshold", "0",
        "-pix_fmt", "yuv420p",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-color_range", "tv",
    ]


def audio_args() -> list[str]:
    return ["-c:a", "aac", "-b:a", "320k", "-ar", "48000", "-ac", "2"]


def video_filters(text: str | None = None, font_size: int = 56, y: str = "180") -> str:
    base = (
        f"scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=decrease,"
        f"pad={WIDTH}:{HEIGHT}:(ow-iw)/2:(oh-ih)/2:black,"
        f"fps={FPS},format=yuv420p"
    )
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
    run([
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}",
        "-map", "0:v:0", "-map", "0:a:0", "-vf", video_filters(text, font_size, y),
        "-af", "aresample=48000,aformat=channel_layouts=stereo",
        *x264_args(), *audio_args(), "-movflags", "+faststart", str(destination),
    ])


def render_vertical_art_segment(art: Path, audio_source: Path, destination: Path) -> None:
    # Native 9:16 campaign artwork: no blurred side-fill, no horizontal-map letterboxing.
    filter_complex = (
        f"[0:v]scale={WIDTH}:{HEIGHT}:force_original_aspect_ratio=increase,crop={WIDTH}:{HEIGHT},"
        f"zoompan=z='min(zoom+0.0012,1.035)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
        f"d=1:s={WIDTH}x{HEIGHT}:fps={FPS},"
        f"drawtext=fontfile='{FONT}':text='NEW WARZONE MAP?':fontcolor=white:fontsize=64:"
        f"borderw=4:bordercolor=black@0.95:box=1:boxcolor=black@0.30:boxborderw=16:"
        f"x=(w-text_w)/2:y=180,format=yuv420p[v]"
    )
    run([
        "ffmpeg", "-y", "-loop", "1", "-i", str(art), "-ss", "0", "-i", str(audio_source), "-t", "0.600",
        "-filter_complex", filter_complex, "-map", "[v]", "-map", "1:a:0",
        "-af", "aresample=48000,aformat=channel_layouts=stereo",
        *x264_args(), *audio_args(), "-movflags", "+faststart", str(destination),
    ])


def concat_video_only(segments: list[Path], destination: Path) -> None:
    manifest = destination.with_suffix(".concat.txt")
    manifest.write_text("".join(f"file '{segment.resolve()}'\n" for segment in segments))
    run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(manifest),
        "-map", "0:v:0", "-c:v", "copy", "-an", "-movflags", "+faststart", str(destination),
    ])


def render_continuous_audio(
    map_file: Path,
    gameplay_file: Path,
    timeline: list[dict[str, object]],
    destination: Path,
) -> None:
    # Build ONE final AAC track instead of concatenating independently encoded AAC chunks.
    # The CTA keeps the gameplay audio rolling from 16.70 s so the visual jump at 8.90 s
    # does not create the audible 4.2 s discontinuity that existed in the previous render.
    filters: list[str] = []
    labels: list[str] = []
    for index, item in enumerate(timeline):
        source_kind = str(item.get("audio_kind") or ("map" if item["kind"] in ("art", "map") else "gameplay"))
        input_index = 0 if source_kind == "map" else 1
        audio_start = float(item.get("audio_start", item.get("source_start") or 0.0))
        duration = float(item["duration"])
        label = f"a{index}"
        filters.append(
            f"[{input_index}:a:0]atrim=start={audio_start:.3f}:duration={duration:.3f},"
            f"asetpts=PTS-STARTPTS,aresample=48000,aformat=channel_layouts=stereo[{label}]"
        )
        labels.append(f"[{label}]")

    total = sum(float(item["duration"]) for item in timeline)
    filters.append(
        "".join(labels)
        + f"concat=n={len(labels)}:v=0:a=1,"
          f"afade=t=out:st={max(0.0, total - 0.18):.3f}:d=0.18[aout]"
    )
    run([
        "ffmpeg", "-y", "-i", str(map_file), "-i", str(gameplay_file),
        "-filter_complex", ";".join(filters), "-map", "[aout]",
        *audio_args(), str(destination),
    ])


def mux_video_and_audio(video_only: Path, audio_track: Path, destination: Path) -> None:
    run([
        "ffmpeg", "-y", "-i", str(video_only), "-i", str(audio_track),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
        "-shortest", "-movflags", "+faststart", str(destination),
    ])


def probe(path: Path) -> dict[str, object]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries",
        "format=duration,size,bit_rate:stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,bit_rate,sample_rate,channels,color_space,color_transfer,color_primaries,color_range",
        "-of", "json", str(path),
    ], text=True)
    return json.loads(raw)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-bodies", type=Path, required=True)
    parser.add_argument("--gameplay-bodies", type=Path, required=True)
    parser.add_argument("--thumbs-bodies", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("zodiac-map-render-v2"))
    parser.add_argument("--output", type=Path, default=Path("Zodiac_NEW_WARZONE_MAP_9x16_250M_FINAL.mp4"))
    args = parser.parse_args()

    work = args.work_dir.resolve()
    assets = work / "assets"
    segments_dir = work / "segments"
    assets.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    map_name, map_url = pick_asset(args.map_bodies, "video", "9x16")
    gameplay_name, gameplay_url = pick_asset(args.gameplay_bodies, "video", "9x16")
    art_name, art_url = pick_asset(args.thumbs_bodies, "image", "zodiac_wz_1080x1920png")

    map_file = assets / "map_flythrough_9x16.mp4"
    gameplay_file = assets / "gameplay_9x16.mp4"
    art_file = assets / "zodiac_wz_1080x1920.png"

    downloads = {
        "map": {"name": map_name, **download(map_url, map_file)},
        "gameplay": {"name": gameplay_name, **download(gameplay_url, gameplay_file)},
        "opening_art": {"name": art_name, **download(art_url, art_file)},
    }

    timeline = [
        {"id": "S0", "kind": "art", "source": art_name, "source_start": None, "audio_kind": "map", "audio_start": 0.00, "duration": 0.60, "final": [0.00, 0.60], "text": "NEW WARZONE MAP?"},
        {"id": "S1", "kind": "map", "source": map_name, "source_start": 0.60, "audio_kind": "map", "audio_start": 0.60, "duration": 1.00, "final": [0.60, 1.60], "text": "ZODIAC - FREE BETA WEEKEND 2"},
        {"id": "S2", "kind": "map", "source": map_name, "source_start": 24.15, "audio_kind": "map", "audio_start": 24.15, "duration": 1.40, "final": [1.60, 3.00], "text": None},
        {"id": "S3", "kind": "gameplay", "source": gameplay_name, "source_start": 11.50, "audio_kind": "gameplay", "audio_start": 11.50, "duration": 1.60, "final": [3.00, 4.60], "text": None},
        {"id": "S4a", "kind": "map", "source": map_name, "source_start": 31.00, "audio_kind": "map", "audio_start": 31.00, "duration": 0.70, "final": [4.60, 5.30], "text": None},
        {"id": "S4b", "kind": "map", "source": map_name, "source_start": 33.55, "audio_kind": "map", "audio_start": 33.55, "duration": 0.70, "final": [5.30, 6.00], "text": None},
        {"id": "S4c", "kind": "map", "source": map_name, "source_start": 35.05, "audio_kind": "map", "audio_start": 35.05, "duration": 0.70, "final": [6.00, 6.70], "text": None},
        {"id": "S5", "kind": "gameplay", "source": gameplay_name, "source_start": 14.50, "audio_kind": "gameplay", "audio_start": 14.50, "duration": 2.20, "final": [6.70, 8.90], "text": None},
        {"id": "S6", "kind": "cta", "source": gameplay_name, "source_start": 20.90, "audio_kind": "gameplay", "audio_start": 16.70, "duration": 1.70, "final": [8.90, 10.60], "text": None},
    ]

    rendered: list[Path] = []
    for item in timeline:
        destination = segments_dir / f"{item['id']}.mp4"
        if item["kind"] == "art":
            render_vertical_art_segment(art_file, map_file, destination)
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

    video_only = work / "final-video-only.mp4"
    audio_track = work / "final-continuous-audio.m4a"
    concat_video_only(rendered, video_only)
    render_continuous_audio(map_file, gameplay_file, timeline, audio_track)

    output = args.output.resolve()
    mux_video_and_audio(video_only, audio_track, output)
    final_probe = probe(output)

    report = {
        "creative": "NEW WARZONE MAP? - native 9x16 art",
        "timeline": timeline,
        "downloads": downloads,
        "probe": final_probe,
        "audio_fix": {
            "single_final_aac_encode": True,
            "cta_visual_source_start": 20.90,
            "cta_audio_source_start": 16.70,
            "cta_audio_continues_from_previous_gameplay": True,
            "final_fade_seconds": 0.18,
        },
        "requirements": {
            "official_assets_only": True,
            "native_9x16_opening_art": True,
            "added_music": False,
            "added_sfx": False,
            "ai_voice": False,
            "target_resolution": "1080x1920",
            "target_fps": 60,
            "target_video_bitrate_kbps": 250000,
            "rate_control": "CBR",
            "codec": "H.264",
            "container": "MP4",
            "color_space": "Rec.709 SDR",
            "target_duration_seconds": 10.6,
            "cta_seconds": 1.7,
        },
    }
    (work / "render-report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
