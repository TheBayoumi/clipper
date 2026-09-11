from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import mw4_gameplay_batch as batch


def run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build full-window visual QA sheets for MW4 selections")
    parser.add_argument("--source-key", choices=("r1", "batch2", "week2"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    candidates = batch.discover_candidates(args.source, args.source_key, config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index, candidate in enumerate(candidates, start=1):
        stem = f"{args.source_key}_{index:02d}_{candidate.effect_profile}"
        pattern = args.output_dir / f"{stem}_sheet_%02d.jpg"
        run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                f"{candidate.start:.3f}",
                "-i",
                str(args.source),
                "-t",
                f"{candidate.duration:.3f}",
                "-vf",
                "fps=4,scale=320:-1,tile=6x5:padding=2:margin=2",
                "-frames:v",
                "2",
                str(pattern),
            ]
        )
        record = asdict(candidate)
        record["index"] = index
        records.append(record)

    (args.output_dir / f"{args.source_key}_candidates.json").write_text(
        json.dumps(records, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
