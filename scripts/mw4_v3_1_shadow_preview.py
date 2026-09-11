from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import mw4_fullframe_retention_v3_1 as renderer
import mw4_semantic_gameplay_v3_1 as semantic


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-key', required=True)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding='utf-8'))
    timeline = semantic.analyze_source(args.source, config)
    plans = semantic.build_plans(timeline, config, config.get('excluded_windows', {}).get(args.source_key, []))
    selected = semantic.select_plans(plans, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sheets = args.output_dir / 'contact_sheets'

    report = {
        'source_key': args.source_key,
        'semantic_engine': 'deterministic-gameplay-v3.1',
        'editorial_planner': 'semantic-editor-v3.1',
        'shot_count': len(timeline.shots),
        'consolidated_event_count': len(timeline.consolidated_events),
        'engagement_count': len(timeline.engagements),
        'finishing_move_like_count': len(timeline.finishing_moves),
        'finishing_moves': [asdict(item) for item in timeline.finishing_moves],
        'selected': [asdict(item) for item in selected],
        'preview_note': 'Review proxies only. Final production remains 1920x1080 60000/1001 at 250 Mbps CBR.',
    }
    (args.output_dir / f'{args.source_key}_shadow.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    for index, plan in enumerate(selected, 1):
        graph, duration = renderer.build_filter(plan, config)
        target = args.output_dir / f'V31_SHADOW_{args.source_key}_{index:02d}_{plan.story_type}.mp4'
        run([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(args.source),
            '-filter_complex', graph, '-map', '[outv]', '-map', '[aout]',
            '-c:v', 'libx264', '-preset', 'veryfast', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
            '-b:v', '8M', '-maxrate', '10M', '-bufsize', '20M',
            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
            '-movflags', '+faststart', '-t', f'{duration:.3f}', str(target),
        ])
        renderer.create_contact_sheets(target, sheets)

    print(json.dumps({
        'source': args.source_key,
        'selected': len(selected),
        'stories': [plan.story_type for plan in selected],
        'finishing_move_like': len(timeline.finishing_moves),
    }))


if __name__ == '__main__':
    main()
