"""Run the VEIL scout: camera, agreement gate, lossy link, dashboard."""

import argparse

from veil.mission import run
from veil.source import VideoFrameSource


def main() -> None:
    parser = argparse.ArgumentParser(description="ATEC'26 VEIL mission")
    parser.add_argument("--loss", type=float, default=0.3, help="packet loss while the link is up")
    parser.add_argument("--port", type=int, default=8877)
    parser.add_argument("--hold", type=float, default=-1, help="seconds to keep the dashboard up; -1 waits")
    parser.add_argument("--source", default=None, help="PNG directory or mp4 plus a pose CSV")
    parser.add_argument("--in-process", action="store_true", help="keep onboard, radio, and ground in one process")
    args = parser.parse_args()
    source = VideoFrameSource(args.source) if args.source else None
    report = run(
        live=True,
        realtime=True,
        loss=args.loss,
        port=args.port,
        hold=args.hold,
        frames=130,
        dt=0.2,
        blackout=(4.0, 12.0),
        scenario="live",
        source=source,
        processes=not args.in_process,
    )
    print(report)


if __name__ == "__main__":
    main()
