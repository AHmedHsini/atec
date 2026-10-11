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
    """PNG directory or mp4. A missing pose file uses constant velocity."""

    def __init__(self, path: str, speed: float = 0.25, pose_source: str = "auto", camera_path: str | None = None):
        from veil.camera import load_camera, use_camera

        self.path = Path(path)
        self.speed = float(speed)
        self.pose_source = pose_source
        self.camera_path = camera_path
        self.camera = use_camera(load_camera(camera_path))
        self._capture = None
        self.bad_frames = 0
        self.pose_warning = ""
        self.pose_mode = "file"

    def frames(self):
        try:
            if self.path.is_dir():
                yield from self._from_directory()
                return
            yield from self._from_video()
        except Exception as exc:
            self.bad_frames += 1
            self.pose_warning = f"ingest stopped: {exc}"
            return

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        from veil.camera import use_camera

        use_camera(None)

    def _from_directory(self):
        pose_path = self.path / "poses.csv"
        rows = _read_poses(pose_path) if pose_path.exists() else None
        if rows is None:
            self.pose_mode = "constant-velocity"
            self.pose_warning = "pose source: assumed constant velocity"
            names = sorted(self.path.glob("*.png"))
            for index, image_path in enumerate(names):
                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    self.bad_frames += 1
                    continue
                t_s = index * 0.1
                yield _sample(_motion_row(index, t_s, self.speed), self._fit(image))
            return
        for row in rows:
            frame = int(row["frame"])
            image_path = self.path / f"{frame:04d}.png"
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                self.bad_frames += 1
                continue
            yield _sample(row, self._fit(image))

    def _from_video(self):
        csv_path = self.path.with_name(self.path.stem + "_poses.csv")
        use_file = self.pose_source != "vo" and self.pose_source != "constant" and csv_path.exists()
        self._capture = cv2.VideoCapture(str(self.path))
        if not self._capture.isOpened():
            self.bad_frames += 1
            self.pose_warning = "unreadable video"
            return
        fps = float(self._capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if use_file and self.pose_source == "auto":
            rows = _read_poses(csv_path)
            self.pose_mode = "file"
            for row in rows:
                image = self._read_frame()
                if image is None:
                    self.bad_frames += 1
                    continue
                yield _sample(row, image)
            self.close()
            return
        self.pose_mode = "visual-odometry" if self.pose_source == "vo" else "constant-velocity"
        self.pose_warning = "pose source: assumed constant velocity" if self.pose_mode != "visual-odometry" else "pose source: visual odometry"
        previous = None
        y_m = 0.0
        index = 0
        last_t = 0.0
        while True:
            msec = float(self._capture.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            image = self._read_frame()
            if image is None:
                break
            if msec > 0:
                t_s = msec / 1000.0
            elif fps > 1:
                t_s = index / fps
                if index == 0:
                    self.pose_warning += "; timestamps missing, used frame rate"
            else:
                t_s = index * 0.1
                if index == 0:
                    self.pose_warning += "; timestamps missing, used 0.1 s steps"
            dt = max(1e-3, t_s - last_t) if index else 0.0
            if self.pose_mode == "visual-odometry" and previous is not None and dt > 0:
                speed = _flow_speed(previous, image, self.camera)
                if speed is None:
                    speed = self.speed
                    if "flow failed" not in self.pose_warning:
                        self.pose_warning += "; flow failed, used constant speed"
                y_m += speed * dt
            else:
                y_m = self.speed * t_s
            previous = image
            yield _sample(_motion_row(index, t_s, self.speed, y_m), image)
            last_t = t_s
            index += 1
        self.close()

    def _read_frame(self):
        if self._capture is None:
            return None
        try:
            ok, image = self._capture.read()
        except cv2.error:
            self.bad_frames += 1
            return None
        if not ok or image is None:
            return None
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.size == 0:
            self.bad_frames += 1
            return None
        return self._fit(image)

    def _fit(self, image: np.ndarray) -> np.ndarray:
        from veil.camera import HEIGHT, WIDTH, use_camera

        if image.shape[1] != self.camera.width or image.shape[0] != self.camera.height:
            self.camera = self.camera.scaled_to(int(image.shape[1]), int(image.shape[0]))
        fitted = image
        if fitted.shape[1] != WIDTH or fitted.shape[0] != HEIGHT:
            self.camera = self.camera.scaled_to(WIDTH, HEIGHT)
            fitted = cv2.resize(fitted, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        use_camera(self.camera)
        return fitted


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


def _motion_row(index: int, t_s: float, speed: float, y_m: float | None = None) -> dict:
    return {
        "frame": index,
        "t_s": t_s,
        "x_m": 0.0,
        "y_m": speed * t_s if y_m is None else y_m,
        "yaw_rad": 0.0,
    }


def _flow_speed(previous: np.ndarray, image: np.ndarray, camera) -> float | None:
    """Median ground-plane speed from sparse flow on the lower half of the frame."""
    from veil.camera import pixel_to_ground

    h, w = image.shape[:2]
    y0 = h // 2
    points = []
    for v in range(y0 + 8, h - 8, 16):
        for u in range(16, w - 16, 24):
            points.append([u, v])
    if len(points) < 8:
        return None
    prev_pts = np.asarray(points, np.float32).reshape(-1, 1, 2)
    nxt, status, _err = cv2.calcOpticalFlowPyrLK(previous, image, prev_pts, None)
    if nxt is None or status is None:
        return None
    speeds = []
    for src, dst, ok in zip(prev_pts.reshape(-1, 2), nxt.reshape(-1, 2), status.reshape(-1)):
        if not ok:
            continue
        a = pixel_to_ground(float(src[0]), float(src[1]), (0.0, 0.0))
        b = pixel_to_ground(float(dst[0]), float(dst[1]), (0.0, 0.0))
        if a is None or b is None:
            continue
        speeds.append(abs(b[1] - a[1]))
    if len(speeds) < 5:
        return None
    return float(np.median(speeds))


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
