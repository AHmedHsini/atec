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


def pixel_to_ground(u: float, v: float, scout_xy: tuple[float, float], yaw: float = 0.0):
    """Intersect the pixel ray with z = 0. None if the ray misses the ground."""
    if not math.isfinite(u) or not math.isfinite(v) or not math.isfinite(yaw):
        return None
    if not math.isfinite(scout_xy[0]) or not math.isfinite(scout_xy[1]):
        return None
    x_cam = (u - CX) / FX
    y_cam = (v - CY) / FY
    right, down, fwd = _axes(yaw)
    direction = x_cam * right + y_cam * down + fwd
    if direction[2] >= -1e-8:
        return None
    t = -CAM_H / direction[2]
    if t <= 0:
        return None
    x = scout_xy[0] + t * direction[0]
    y = scout_xy[1] + t * direction[1]
    flat = math.hypot(x - scout_xy[0], y - scout_xy[1])
    return x, y, flat


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
