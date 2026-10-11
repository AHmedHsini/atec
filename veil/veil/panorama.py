"""A 360-degree surface strip. Turning the heading cuts a camera frame out of it."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from veil.camera import CameraSpec, use_camera

HFOV_DEG = 50.0
MAST_M = 1.6


def load_strip(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.ndim != 2 or image.size == 0:
        raise ValueError(f"unreadable panorama: {path}")
    return image


def _pitch_deg(image: np.ndarray, fy: float, cy: float) -> float:
    height, width = image.shape
    rows = []
    for column in range(0, width, max(1, width // 80)):
        bright = np.where(image[:, column] > 28)[0]
        if len(bright):
            rows.append(int(bright[0]))
    if len(rows) < 8:
        return 0.0
    horizon = float(np.median(rows))
    return math.degrees(math.atan((cy - horizon) / fy))


def crop_view(image: np.ndarray, yaw_deg: float) -> tuple[np.ndarray, CameraSpec]:
    """Cut one forward view. The strip's width is a full circle."""
    height, width = image.shape
    crop_w = max(64, int(round(width * HFOV_DEG / 360.0)))
    center = (float(yaw_deg) % 360.0) / 360.0 * width
    start = int(round(center - crop_w / 2.0)) % width
    if start + crop_w <= width:
        crop = image[:, start : start + crop_w]
    else:
        wrapped = np.concatenate((image[:, start:], image[:, : (start + crop_w) % width]), axis=1)
        crop = wrapped[:, :crop_w]
    fx = (crop_w / 2.0) / math.tan(math.radians(HFOV_DEG / 2.0))
    fy = fx
    spec = CameraSpec(
        crop_w,
        height,
        fx,
        fy,
        crop_w / 2.0,
        height / 2.0,
        MAST_M,
        _pitch_deg(image, fy, height / 2.0),
        pov="surface",
    )
    return crop, spec


class PanoramaSource:
    """Live frames from one strip. The dashboard sets yaw_deg while this runs."""

    def __init__(self, path: str | Path, dt: float = 0.2):
        self.path = Path(path)
        self.image = load_strip(self.path)
        self.dt = float(dt)
        self.yaw_deg = 0.0
        self.pose_warning = "Apollo 11 panorama. Heading turns the crop. USGS-style strip, file from Wikimedia."
        self.pose_mode = "panorama"
        self.camera = crop_view(self.image, 0.0)[1]

    def frames(self):
        frame = 0
        while True:
            crop, spec = crop_view(self.image, self.yaw_deg)
            self.camera = use_camera(spec)
            yield {
                "frame": frame,
                "t_s": frame * self.dt,
                "x_m": 0.0,
                "y_m": 0.0,
                "yaw_rad": math.radians(float(self.yaw_deg) % 360.0),
                "image": crop,
            }
            frame += 1

    def close(self) -> None:
        use_camera(None)
