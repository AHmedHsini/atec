"""Run a video through the scout without requiring a pose file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from veil.mission import run
from veil.reports import write_json
from veil.source import VideoFrameSource


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a video into the VEIL pipeline")
    parser.add_argument("video")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--speed", type=float, default=0.25)
    parser.add_argument("--pose-source", default="auto", choices=("auto", "constant", "vo", "file"))
    parser.add_argument("--loss", type=float, default=0.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    source = VideoFrameSource(args.video, speed=args.speed, pose_source=args.pose_source, camera_path=args.camera)
    report = run(
        live=False,
        loss=args.loss,
        hold=0,
        frames=1,
        dt=0.1,
        blackout=(1e9, 1e9 + 1),
        scenario="ingest",
        seed=1,
        source=source,
        processes=False,
        write_report=False,
    )
    report["bad_frames"] = source.bad_frames
    report["pose_warning"] = source.pose_warning
    report["pose_mode"] = source.pose_mode
    out = Path(args.out) if args.out else Path("artifacts") / "external" / "ingest.json"
    write_json(out, report)
    print(json.dumps({"out": str(out), "pose_warning": source.pose_warning, "hazards_delivered": report["hazards_delivered"], "bad_frames": source.bad_frames}, indent=2))


if __name__ == "__main__":
    main()
