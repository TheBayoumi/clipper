"""Run the deployed vision worker against production-shaped synthetic frames."""

from __future__ import annotations

import argparse
import base64
import json
from typing import Any

import cv2
import modal
import numpy as np


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", default="clipper-open-editor")
    parser.add_argument("--class-name", default="VisionModel")
    parser.add_argument("--frames", type=int, default=35)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser


def _frame(index: int, *, width: int, height: int) -> str:
    color = ((index * 37) % 255, (index * 71) % 255, (index * 113) % 255)
    image = np.full((height, width, 3), color, dtype=np.uint8)
    cv2.putText(
        image,
        f"synthetic frame {index:02d}",
        (max(20, width // 12), max(60, height // 2)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(1.0, width / 1000),
        (255, 255, 255),
        max(2, width // 500),
        cv2.LINE_AA,
    )
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise RuntimeError("failed to encode synthetic vision frame")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def main() -> int:
    args = _parser().parse_args()
    if args.frames < 1 or args.width < 64 or args.height < 64:
        raise ValueError("frames must be positive and dimensions must be at least 64 pixels")
    frames = [_frame(index, width=args.width, height=args.height) for index in range(args.frames)]
    payload: dict[str, Any] = {
        "task": "visual_timeline_scout",
        "frames_base64": frames,
        "context": {
            "video_id": "synthetic-smoke",
            "source_hash": "synthetic",
            "frame_timestamps": [float(index * 90) for index in range(args.frames)],
            "instruction": "Describe only visible synthetic evidence.",
        },
    }
    worker = modal.Cls.from_name(args.app, args.class_name)()
    worker.ready.remote()
    result = worker.inspect.remote(payload)
    if not isinstance(result, dict):
        raise RuntimeError(f"vision worker returned {type(result).__name__}, expected object")
    if result.get("error"):
        raise RuntimeError(json.dumps(result["error"], ensure_ascii=False))
    print(
        json.dumps(
            {
                "frames": args.frames,
                "dimensions": [args.width, args.height],
                "value": result.get("value"),
                "usage": result.get("usage"),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
