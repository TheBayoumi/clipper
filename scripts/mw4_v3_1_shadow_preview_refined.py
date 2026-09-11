from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import mw4_fullframe_retention_v3_1 as renderer
import mw4_semantic_gameplay_v3_1_refined as semantic

# Reuse the same FFmpeg edit/effect graph as production while routing semantic
# decisions through the refined V3.1 planner.
renderer.semantic = semantic


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
    excluded = config.get('excluded_windows', {}).get(args.source_key, [])
    timeline = semantic.analyze_source(args.source, config)
    diagnostics = semantic.diagnose_source(timeline, config, excluded)
    plans = semantic.build_plans(timeline, config, excluded)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f'{args.source_key}_shadow_refined.json'
    report = {
        'source_key': args.source_key,
        'semantic_engine': 'deterministic-gameplay-v3.1-refined',
        'editorial_planner': 'semantic-editor-v3.1-refined',
        'candidate_mode': 'engagement_driven_natural_boundaries',
        'diagnostics': diagnostics,
        'candidate_count_after_semantic_gates': len(plans),
        'selected': [],
        'preview_note': (
            'Review proxies only. Final production remains 1920x1080 '
            '60000/1001 at 250 Mbps CBR.'
        ),
    }

    minimum = int(config.get('minimum_count_per_source', 2))
    if len(plans) < minimum:
        report['failure'] = (
            f'Only {len(plans)} refined semantic candidates passed; minimum is {minimum}. '
            'Quality gates were not lowered.'
        )
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        raise RuntimeError(report['failure'])

    selected = semantic.select_plans(plans, config)
    report['selected'] = [asdict(item) for item in selected]
    report['finishing_move_like_count'] = len(timeline.finishing_moves)
    report['finishing_moves'] = [asdict(item) for item in timeline.finishing_moves]
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    sheets = args.output_dir / 'contact_sheets'
    for index, plan in enumerate(selected, 1):
        graph, duration = renderer.build_filter(plan, config)
        target = args.output_dir / f'V31_REFINED_{args.source_key}_{index:02d}_{plan.story_type}.mp4'
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
        'diagnostic_pass_count': diagnostics['counts']['pass'],
    }))


if __name__ == '__main__':
    main()
