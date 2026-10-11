"""Run the VEIL scout: camera, agreement gate, lossy link, dashboard."""

import argparse
import os

from veil.mission import run
from veil.source import VideoFrameSource


def main() -> None:
    parser = argparse.ArgumentParser(description="ATEC'26 VEIL mission")
    parser.add_argument("--loss", type=float, default=0.3, help="packet loss while the link is up")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--hold", type=float, default=-1, help="seconds to keep the dashboard up; -1 waits")
    parser.add_argument("--source", default=None, help="PNG directory or mp4. A missing pose file uses constant velocity.")
    parser.add_argument("--speed", type=float, default=0.25, help="forward speed m/s when no pose file exists")
    parser.add_argument("--pose-source", default="auto", choices=("auto", "constant", "vo", "file"))
    parser.add_argument("--camera", default=None, help="camera.json override")
    parser.add_argument("--panorama", action="store_true", help="turn through the Apollo 11 360 strip")
    parser.add_argument("--polar", action="store_true", help="use the PolarDB network and dataset/polar/demo photos")
    parser.add_argument("--in-process", action="store_true", help="keep onboard, radio, and ground in one process")
    args = parser.parse_args()
    source = None
    blackout = (4.0, 12.0)
    if args.panorama:
        os.environ["VEIL_MODEL"] = "polar"
        from pathlib import Path

        from veil.panorama import PanoramaSource

        source = PanoramaSource(Path(__file__).resolve().parent.parent / "dataset" / "panorama" / "apollo11.jpg")
        blackout = (1.0e9, 1.0e9)
    elif args.polar:
        os.environ["VEIL_MODEL"] = "polar"
        if not args.source:
            from pathlib import Path

            args.source = str(Path(__file__).resolve().parent.parent / "dataset" / "polar" / "demo")
    if source is None and args.source:
        source = VideoFrameSource(args.source, speed=args.speed, pose_source=args.pose_source, camera_path=args.camera)
    if source is not None and getattr(source, "camera", None) is not None and getattr(source.camera, "pov", "") == "surface":
        blackout = (1.0e9, 1.0e9)
    report = run(
        live=True,
        realtime=True,
        loss=args.loss,
        port=args.port,
        hold=args.hold,
        frames=130,
        dt=0.2,
        blackout=blackout,
        scenario="live",
        source=source,
        processes=not args.in_process,
        secure=True,
    )
    print(report)


if __name__ == "__main__":
    main()
