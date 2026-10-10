"""Camera frames for the onboard computer.

Perception reads this interface and does not care whether the picture
was painted this step or loaded from a file.
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from veil.sim import render_pose, scout_pose

POSE_COLUMNS = ("frame", "t_s", "x_m", "y_m", "yaw_rad")


class FrameSource:
    def frames(self):
        raise NotImplementedError

    def close(self) -> None:
        return None


class SimFrameSource(FrameSource):
    def __init__(self, frames: int, step_m: float, dt: float, seed: int):
        self.frame_count = frames
        self.step_m = step_m
        self.dt = dt
        self.seed = seed

    def frames(self):
        rng = np.random.default_rng(self.seed)
        for frame in range(self.frame_count):
            scout = scout_pose(frame, self.step_m)
            yield {
                "frame": frame,
                "t_s": frame * self.dt,
                "x_m": scout[0],
                "y_m": scout[1],
                "yaw_rad": 0.0,
                "image": render_pose(scout, rng),
            }


class VideoFrameSource(FrameSource):
    """A PNG directory with poses.csv, or an mp4 beside a poses CSV."""

    def __init__(self, path: str):
        self.path = Path(path)
        self._capture = None

    def frames(self):
        if self.path.is_dir():
            yield from self._from_directory()
            return
        yield from self._from_video()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def _from_directory(self):
        rows = _read_poses(self.path / "poses.csv")
        for row in rows:
            frame = int(row["frame"])
            image_path = self.path / f"{frame:04d}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(image_path)
            yield _sample(row, image)

    def _from_video(self):
        csv_path = self.path.with_name(self.path.stem + "_poses.csv")
        rows = _read_poses(csv_path)
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            raise FileNotFoundError(self.path)
        for row in rows:
            ok, image = self._capture.read()
            if not ok:
                break
            if image.ndim == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            yield _sample(row, image)
        self.close()


def record_drive(
    directory: str,
    frames: int = 140,
    step_m: float = 0.05,
    dt: float = 0.1,
    seed: int = 26,
    write_video: bool = False,
) -> dict:
    """Write a PNG sequence and poses.csv. Optionally also an mp4."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    source = SimFrameSource(frames, step_m, dt, seed)
    writer = None
    video_path = out / "drive.mp4"
    if write_video:
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240), False)
    with (out / "poses.csv").open("w", newline="", encoding="utf-8") as handle:
        csv_writer = csv.DictWriter(handle, fieldnames=list(POSE_COLUMNS))
        csv_writer.writeheader()
        written = 0
        for sample in source.frames():
            gray = sample["image"]
            cv2.imwrite(str(out / f"{sample['frame']:04d}.png"), gray)
            csv_writer.writerow({name: sample[name] for name in POSE_COLUMNS})
            if writer is not None:
                writer.write(gray)
            written += 1
    if writer is not None:
        writer.release()
        if video_path.exists():
            side = out / "drive_poses.csv"
            side.write_text((out / "poses.csv").read_text(encoding="utf-8"), encoding="utf-8")
    return {"frames": written, "directory": str(out), "video": str(video_path) if write_video else None}


def _read_poses(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sample(row: dict, image: np.ndarray) -> dict:
    return {
        "frame": int(row["frame"]),
        "t_s": float(row["t_s"]),
        "x_m": float(row["x_m"]),
        "y_m": float(row["y_m"]),
        "yaw_rad": float(row["yaw_rad"]),
        "image": image,
    }
