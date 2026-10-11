"""Pinhole mast camera and ground-plane intersection.

The scout camera looks forward and is pitched down. A pixel becomes a ray
from the mast. The ray is intersected with the local ground plane z = 0,
which is the landing-site vertical. One view gives a hypothesis. A second
view from a new position has to land on the same point before that
hypothesis is trusted.
"""

from __future__ import annotations

import math

import numpy as np

WIDTH = 320
HEIGHT = 240
FX = 300.0
FY = 300.0
CX = WIDTH / 2.0
CY = HEIGHT / 2.0
CAM_H = 0.55
PITCH = math.radians(18.0)


class CameraSpec:
    """Intrinsics and mast pose. Defaults match the flight painter."""

    def __init__(
        self,
        width: int = WIDTH,
        height: int = HEIGHT,
        fx: float = FX,
        fy: float = FY,
        cx: float = CX,
        cy: float = CY,
        mast_height_m: float = CAM_H,
        pitch_deg: float = 18.0,
        distortion: list | None = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.mast_height_m = float(mast_height_m)
        self.pitch_deg = float(pitch_deg)
        self.distortion = list(distortion or [])

    def scaled_to(self, width: int, height: int) -> "CameraSpec":
        if width <= 0 or height <= 0 or self.width <= 0 or self.height <= 0:
            return self
        sx = width / self.width
        sy = height / self.height
        return CameraSpec(
            width,
            height,
            self.fx * sx,
            self.fy * sy,
            self.cx * sx,
            self.cy * sy,
            self.mast_height_m,
            self.pitch_deg,
            self.distortion,
        )


def load_camera(path: str | None = None) -> CameraSpec:
    spec = CameraSpec()
    if not path:
        return spec
    import json
    from pathlib import Path

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("camera config must be a JSON object")
    known = {
        "width": spec.width,
        "height": spec.height,
        "fx": spec.fx,
        "fy": spec.fy,
        "cx": spec.cx,
        "cy": spec.cy,
        "mast_height_m": spec.mast_height_m,
        "pitch_deg": spec.pitch_deg,
        "distortion": spec.distortion,
    }
    known.update({key: raw[key] for key in known if key in raw})
    return CameraSpec(**known)


_active = CameraSpec()


def use_camera(spec: CameraSpec | None = None) -> CameraSpec:
    global _active
    _active = spec or CameraSpec()
    return _active


def active_camera() -> CameraSpec:
    return _active


def _axes(yaw: float, pitch: float = PITCH):
    forward = np.array([math.sin(yaw), math.cos(yaw), 0.0])
    right = np.array([math.cos(yaw), -math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    cp, sp = math.cos(pitch), math.sin(pitch)
    cam_forward = cp * forward - sp * up
    cam_down = -sp * forward - cp * up
    return right, cam_down, cam_forward


def project(points: np.ndarray, scout_xy: tuple[float, float], yaw: float = 0.0):
    """Project Nx3 world points to pixels. Returns u, v, z_cam."""
    rel = np.asarray(points, dtype=float).copy()
    rel[:, 0] -= scout_xy[0]
    rel[:, 1] -= scout_xy[1]
    rel[:, 2] -= CAM_H
    right, down, fwd = _axes(yaw)
    x_cam = rel @ right
    y_cam = rel @ down
    z_cam = rel @ fwd
    with np.errstate(divide="ignore", invalid="ignore"):
        u = FX * x_cam / z_cam + CX
        v = FY * y_cam / z_cam + CY
    return u, v, z_cam


def pixel_to_ground(
    u: float,
    v: float,
    scout_xy: tuple[float, float],
    yaw: float = 0.0,
    pitch: float | None = None,
    mast_h: float | None = None,
    cx: float | None = None,
    cy: float | None = None,
    fx: float | None = None,
    fy: float | None = None,
):
    """Intersect the pixel ray with z = 0. None if the ray misses the ground.

    Returns x, y, flat range from the scout. Optional overrides are for
    perturbation studies. The flight call uses the active camera spec,
    which defaults to the module constants.
    """
    if not math.isfinite(u) or not math.isfinite(v) or not math.isfinite(yaw):
        return None
    if not math.isfinite(scout_xy[0]) or not math.isfinite(scout_xy[1]):
        return None
    spec = _active
    fx = spec.fx if fx is None else fx
    fy = spec.fy if fy is None else fy
    cx = spec.cx if cx is None else cx
    cy = spec.cy if cy is None else cy
    mast = spec.mast_height_m if mast_h is None else mast_h
    pitch_rad = math.radians(spec.pitch_deg) if pitch is None else pitch
    if fx == 0 or fy == 0 or not math.isfinite(mast):
        return None
    x_cam = (u - cx) / fx
    y_cam = (v - cy) / fy
    right, down, fwd = _axes(yaw, pitch_rad)
    direction = x_cam * right + y_cam * down + fwd
    if direction[2] >= -1e-8:
        return None
    t = -mast / direction[2]
    if t <= 0:
        return None
    x = scout_xy[0] + t * direction[0]
    y = scout_xy[1] + t * direction[1]
    flat = math.hypot(x - scout_xy[0], y - scout_xy[1])
    return x, y, flat


def camera_depth(x: float, y: float, scout_xy: tuple[float, float], yaw: float = 0.0, mast_h: float | None = None, pitch: float | None = None) -> float:
    """Depth along the optical axis of a ground point, in meters."""
    spec = _active
    mast = spec.mast_height_m if mast_h is None else mast_h
    pitch_rad = math.radians(spec.pitch_deg) if pitch is None else pitch
    rel = np.array([x - scout_xy[0], y - scout_xy[1], -mast], dtype=float)
    _right, _down, fwd = _axes(yaw, pitch_rad)
    return float(rel @ fwd)


def geometric_sigma(
    u: float,
    v: float,
    scout_xy: tuple[float, float],
    yaw: float = 0.0,
    samples: int = 8,
    pixel_sigma: float = 1.0,
    pitch_sigma_deg: float = 1.0,
    height_sigma_m: float = 0.02,
    seed: int = 0,
) -> float:
    """95th percentile ground scatter from pixel, pitch, and mast noise."""
    rng = np.random.default_rng(seed)
    spec = _active
    base_pitch = math.radians(spec.pitch_deg)
    hits = []
    for _ in range(samples):
        hit = pixel_to_ground(
            u + float(rng.normal(0, pixel_sigma)),
            v + float(rng.normal(0, pixel_sigma)),
            scout_xy,
            yaw,
            pitch=base_pitch + math.radians(float(rng.normal(0, pitch_sigma_deg))),
            mast_h=spec.mast_height_m + float(rng.normal(0, height_sigma_m)),
        )
        if hit is not None:
            hits.append((hit[0], hit[1]))
    if len(hits) < 4:
        return 0.05
    cloud = np.asarray(hits, dtype=float)
    offset = np.linalg.norm(cloud - cloud.mean(axis=0), axis=1)
    return float(max(0.05, np.percentile(offset, 95)))


def radius_m(distance_m: float, radius_px: float, depth_m: float, fx: float | None = None) -> tuple[float, float]:
    """Old flat-range radius and the pinhole radius. Caller keeps the better one."""
    focal = _active.fx if fx is None else fx
    old = distance_m * radius_px / focal if focal else 0.0
    new = depth_m * radius_px / focal if focal and depth_m > 0 else old
    return old, new


def view_polygon(scout_xy: tuple[float, float], yaw: float = 0.0, max_range: float = 8.0):
    """Ground patch the camera actually sees, same rays as the picture."""
    far_v = 30
    for row in range(HEIGHT - 2, 25, -3):
        hit = pixel_to_ground(CX, row, scout_xy, yaw)
        if hit and hit[2] >= max_range:
            far_v = row
            break
    corners = (
        pixel_to_ground(1, HEIGHT - 2, scout_xy, yaw),
        pixel_to_ground(WIDTH - 2, HEIGHT - 2, scout_xy, yaw),
        pixel_to_ground(WIDTH - 2, far_v, scout_xy, yaw),
        pixel_to_ground(1, far_v, scout_xy, yaw),
    )
    if any(point is None for point in corners):
        return []
    return [{"x": float(round(float(point[0]), 2)), "y": float(round(float(point[1]), 2))} for point in corners]
