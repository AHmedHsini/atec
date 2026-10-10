"""Pinhole camera on a flat crater floor. Owner: Ahmed.

Frame contract, used to defend the pixel-to-ground transform:

- World / landing site: X forward, Y right, Z up. Origin is Site Alpha. Ground is z = 0.
- Rover pose: (rover_x, rover_y, yaw). Yaw 0 faces +X. Positive yaw turns toward +Y.
- Camera sits height_m above the rover and pitches down by pitch_deg.
- Optical frame: x right, y down, z forward (OpenCV). No lens distortion.

Assumptions a jury can challenge, and that we state on purpose:
the floor is flat, height and pitch are known, and the camera is a pinhole.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraModel:
    fx: float = 480.0
    fy: float = 480.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480
    height_m: float = 1.2
    pitch_deg: float = 30.0

    def project(
        self,
        x: float,
        y: float,
        z: float,
        rover_x: float = 0.0,
        rover_y: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> tuple[float, float] | None:
        cam = self._to_camera(x, y, z, rover_x, rover_y, yaw_deg)
        if cam is None:
            return None
        cx, cy, cz = cam
        if cz <= 1e-4:
            return None
        u = self.fx * cx / cz + self.cx
        v = self.fy * cy / cz + self.cy
        return u, v

    def pixel_to_ground(
        self,
        u: float,
        v: float,
        rover_x: float = 0.0,
        rover_y: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> tuple[float, float] | None:
        """Intersect the pixel ray with the ground plane z = 0."""
        dx = (u - self.cx) / self.fx
        dy = (v - self.cy) / self.fy
        dz = 1.0
        rx, ry, rz = self._camera_to_rover(dx, dy, dz)
        yaw = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        wx = cos_y * rx - sin_y * ry
        wy = sin_y * rx + cos_y * ry
        wz = rz
        if wz >= -1e-6:
            return None
        t = -self.height_m / wz
        if t <= 0:
            return None
        return rover_x + t * wx, rover_y + t * wy

    def _to_camera(
        self,
        x: float,
        y: float,
        z: float,
        rover_x: float,
        rover_y: float,
        yaw_deg: float,
    ) -> tuple[float, float, float] | None:
        yaw = math.radians(yaw_deg)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        dx = x - rover_x
        dy = y - rover_y
        # Inverse yaw: world vector into rover frame (Y right, X forward).
        rx = cos_y * dx + sin_y * dy
        ry = -sin_y * dx + cos_y * dy
        rz = z - self.height_m
        return self._rover_to_camera(rx, ry, rz)

    def _rover_to_camera(self, rx: float, ry: float, rz: float) -> tuple[float, float, float]:
        theta = math.radians(self.pitch_deg)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        # Camera axes expressed in the rover frame. Pitch is positive downward.
        cam_x = ry
        cam_y = -sin_t * rx - cos_t * rz
        cam_z = cos_t * rx - sin_t * rz
        return cam_x, cam_y, cam_z

    def optical_z(
        self,
        x: float,
        y: float,
        z: float,
        rover_x: float = 0.0,
        rover_y: float = 0.0,
        yaw_deg: float = 0.0,
    ) -> float | None:
        cam = self._to_camera(x, y, z, rover_x, rover_y, yaw_deg)
        if cam is None:
            return None
        return cam[2]

    def _camera_to_rover(self, cx: float, cy: float, cz: float) -> tuple[float, float, float]:
        theta = math.radians(self.pitch_deg)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        # Columns are the camera axes in the rover frame.
        rx = cos_t * cz - sin_t * cy
        ry = cx
        rz = -sin_t * cz - cos_t * cy
        return rx, ry, rz
