from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

import mw4_fullframe_retention_v3_1 as renderer
import mw4_semantic_gameplay_v3_1_final as semantic

# Shadow and production now use the SAME semantic planner and the SAME FFmpeg
# edit/effect graph. Only the preview encoder bitrate/preset is cheaper.
renderer.semantic = semantic


def run(command: list[str], *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, check=True, text=True, capture_output=capture)


def _probe_preview(path: Path) -> dict[str, Any]:
    result = run([
        'ffprobe', '-v', 'error',
        '-show_entries',
        'stream=codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,sample_rate,channels:format=duration',
        '-of', 'json', str(path),
    ], capture=True)
    return json.loads(result.stdout)


def _validate_preview(path: Path) -> dict[str, Any]:
    data = _probe_preview(path)
    video = next(item for item in data['streams'] if item['codec_type'] == 'video')
    audio = next(item for item in data['streams'] if item['codec_type'] == 'audio')
    duration = float(data['format']['duration'])
    checks = {
        'duration_10_to_20s': 10.0 <= duration <= 20.0,
        'full_frame_1920x1080': (video['width'], video['height']) == (1920, 1080),
        'codec_h264': video['codec_name'] == 'h264',
        'r_frame_rate_60000_1001': video['r_frame_rate'] == '60000/1001',
        'avg_frame_rate_60000_1001': video['avg_frame_rate'] == '60000/1001',
        'audio_48khz': audio['sample_rate'] == '48000',
        'audio_stereo': audio['channels'] == 2,
    }
    run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'])
    if not all(checks.values()):
        raise RuntimeError(f'Preview technical QA failed for {path.name}: {checks}')
    return {'checks': checks, 'probe': data}


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
    diagnostics = semantic.diagnose_source(timeline, config, excluded, args.source_key)
    plans = semantic.build_plans_for_source(timeline, config, excluded, args.source_key)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / f'{args.source_key}_shadow_refined.json'
    summary_path = args.output_dir / f'{args.source_key}_shadow_summary.json'

    report: dict[str, Any] = {
        'source_key': args.source_key,
        'semantic_engine': 'deterministic-gameplay-v3.1-final',
        'editorial_planner': 'semantic-editor-v3.1-final',
        'candidate_mode': 'engagement_driven_hardened_source_integrity',
        'diagnostics': diagnostics,
        'candidate_count_after_semantic_gates': len(plans),
        'selected': [],
        'rendered_previews': [],
        'finishing_move_like_count': len(timeline.finishing_moves),
        'verified_finishing_move_count': len(timeline.finishing_moves),
        'finishing_moves': [asdict(item) for item in timeline.finishing_moves],
        'verified_source_cut_windows': diagnostics.get('verified_source_cut_windows', []),
        'unplanned_source_cuts': [],
        'unplanned_source_cut_count': 0,
        'preview_note': (
            'Actual review MP4s using the exact production semantic decisions and FFmpeg effect graph. '
            'Only encoding is cheaper. Final production remains 1920x1080 60000/1001 at 250 Mbps CBR.'
        ),
    }

    minimum = int(config.get('minimum_count_per_source', 2))
    if len(plans) < minimum:
        report['failure'] = (
            f'Only {len(plans)} hardened V3.1 semantic candidates passed; minimum is {minimum}. '
            'Quality gates were not lowered to fill quota.'
        )
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        summary_path.write_text(json.dumps({
            'source_key': args.source_key,
            'selected_count': 0,
            'verified_finishing_move_count': len(timeline.finishing_moves),
            'selected_finishing_move_count': 0,
            'unplanned_source_cut_count': 0,
            'failure': report['failure'],
        }, indent=2), encoding='utf-8')
        raise RuntimeError(report['failure'])

    selected = semantic.select_plans(plans, config)
    report['selected'] = [asdict(item) for item in selected]

    violations: list[str] = []
    for index, plan in enumerate(selected, 1):
        for item in semantic.plan_integrity_violations(plan, timeline, config, args.source_key):
            violations.append(f'clip {index}: {item}')
    report['unplanned_source_cuts'] = violations
    report['unplanned_source_cut_count'] = len(violations)

    selected_finishers = [plan for plan in selected if plan.story_type == 'finishing_move_open']
    if timeline.finishing_moves and not selected_finishers:
        report['failure'] = 'A visually verified Finishing Move exists but no finishing_move_open plan was selected.'
    elif violations:
        report['failure'] = 'Selected plans violate the hardened source-integrity contract.'

    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    if report.get('failure'):
        summary_path.write_text(json.dumps({
            'source_key': args.source_key,
            'selected_count': len(selected),
            'stories': [plan.story_type for plan in selected],
            'verified_finishing_move_count': len(timeline.finishing_moves),
            'selected_finishing_move_count': len(selected_finishers),
            'unplanned_source_cut_count': len(violations),
            'failure': report['failure'],
        }, indent=2), encoding='utf-8')
        raise RuntimeError(report['failure'])

    sheets = args.output_dir / 'contact_sheets'
    rendered: list[dict[str, Any]] = []
    for index, plan in enumerate(selected, 1):
        graph, duration = renderer.build_filter(plan, config)
        target = args.output_dir / f'V31_FINAL_{args.source_key}_{index:02d}_{plan.story_type}.mp4'
        run([
            'ffmpeg', '-y', '-hide_banner', '-loglevel', 'error', '-i', str(args.source),
            '-filter_complex', graph, '-map', '[outv]', '-map', '[aout]',
            '-c:v', 'libx264', '-preset', 'veryfast', '-profile:v', 'high', '-pix_fmt', 'yuv420p',
            '-b:v', '8M', '-maxrate', '10M', '-bufsize', '20M',
            '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
            '-movflags', '+faststart', '-t', f'{duration:.3f}', str(target),
        ])
        qa = _validate_preview(target)
        renderer.create_contact_sheets(target, sheets)
        rendered.append({
            'file': target.name,
            'story_type': plan.story_type,
            'effect_profile': plan.effect_profile,
            'output_duration': plan.output_duration,
            'qa': qa,
        })

    report['rendered_previews'] = rendered
    report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')

    summary = {
        'source_key': args.source_key,
        'selected_count': len(selected),
        'stories': [plan.story_type for plan in selected],
        'verified_finishing_move_count': len(timeline.finishing_moves),
        'selected_finishing_move_count': len(selected_finishers),
        'unplanned_source_cut_count': len(violations),
        'preview_qa_passed': all(all(item['qa']['checks'].values()) for item in rendered),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')

    print(json.dumps({
        'source': args.source_key,
        'selected': len(selected),
        'stories': [plan.story_type for plan in selected],
        'verified_finishing_moves': len(timeline.finishing_moves),
        'selected_finishing_moves': len(selected_finishers),
        'unplanned_source_cuts': len(violations),
        'diagnostic_pass_count': diagnostics['counts']['pass'],
    }))


if __name__ == '__main__':
    main()
