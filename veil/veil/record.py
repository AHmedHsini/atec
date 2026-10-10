"""Render the simulated drive to PNG frames, a pose CSV, and an optional mp4."""

from __future__ import annotations

import argparse

from veil.source import record_drive


def main() -> None:
    parser = argparse.ArgumentParser(description="Record the VEIL drive")
    parser.add_argument("--out", required=True, help="directory for PNG frames and poses.csv")
    parser.add_argument("--frames", type=int, default=140)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=26)
    parser.add_argument("--video", action="store_true", help="also write drive.mp4 and drive_poses.csv")
    args = parser.parse_args()
    print(record_drive(args.out, args.frames, args.step, args.dt, args.seed, args.video))


if __name__ == "__main__":
    main()
