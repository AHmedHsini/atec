"""Synthetic crater floor used as the onboard camera feed. Owner: Ahmed.

Hazards are fixed in the landing-site frame. The renderer and the localizer
share CameraModel, so localization error is measured against known positions.
Replace this feed with real frames without changing the rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from crater.scout.camera import CameraModel

ROCK = "rock"
CRATER = "crater"
ICE = "ice"

# BGR swatches the baseline detector knows. A learned model should not need these.
ROCK_BGR = (30, 40, 110)
CRATER_RIM_BGR = (175, 170, 160)
CRATER_CORE_BGR = (100, 104, 108)
ICE_BGR = (250, 245, 235)


@dataclass(frozen=True)
class Hazard:
    kind: str
    x: float
    y: float
    radius_m: float


def default_hazards() -> list[Hazard]:
    # Placed so a scout near the lander sees the whole marker, not a clipped edge.
    return [
        Hazard(ROCK, 5.2, 1.15, 0.40),
        Hazard(ROCK, 6.6, -1.35, 0.34),
        Hazard(CRATER, 7.4, 0.15, 0.48),
        Hazard(ICE, 8.6, -0.55, 0.28),
    ]


@dataclass
class VisibleHazard:
    hazard: Hazard
    u: float
    v: float
    radius_px: float


class CraterScene:
    def __init__(self, camera: CameraModel | None = None, hazards: list[Hazard] | None = None, seed: int = 4):
        self.camera = camera or CameraModel()
        self.hazards = hazards if hazards is not None else default_hazards()
        self._rng = np.random.default_rng(seed)
        self._regolith = self._make_regolith()

    def _make_regolith(self) -> np.ndarray:
        cam = self.camera
        noise = self._rng.normal(108, 10, (cam.height, cam.width, 3))
        return np.clip(noise, 0, 255).astype(np.uint8)

    def visible(
        self,
        rover_x: float,
        rover_y: float,
        yaw_deg: float,
    ) -> list[VisibleHazard]:
        found: list[VisibleHazard] = []
        cam = self.camera
        for hazard in self.hazards:
            pix = cam.project(hazard.x, hazard.y, 0.0, rover_x, rover_y, yaw_deg)
            if pix is None:
                continue
            u, v = pix
            depth = self.camera.optical_z(hazard.x, hazard.y, 0.0, rover_x, rover_y, yaw_deg)
            if depth is None or depth < 0.4:
                continue
            radius_px = hazard.radius_m * cam.fx / depth
            if radius_px < 5:
                continue
            if not (
                radius_px + 2 <= u < cam.width - radius_px - 2
                and radius_px + 2 <= v < cam.height - radius_px - 2
            ):
                continue
            found.append(VisibleHazard(hazard, u, v, radius_px))
        return found

    def render(self, rover_x: float, rover_y: float, yaw_deg: float) -> np.ndarray:
        frame = self._regolith.copy()
        for item in self.visible(rover_x, rover_y, yaw_deg):
            center = (int(round(item.u)), int(round(item.v)))
            radius = max(4, int(round(item.radius_px)))
            kind = item.hazard.kind
            if kind == ROCK:
                cv2.circle(frame, center, radius, ROCK_BGR, -1, lineType=cv2.LINE_AA)
            elif kind == ICE:
                cv2.circle(frame, center, radius, ICE_BGR, -1, lineType=cv2.LINE_AA)
            elif kind == CRATER:
                cv2.circle(frame, center, radius, CRATER_RIM_BGR, -1, lineType=cv2.LINE_AA)
                cv2.circle(frame, center, max(2, radius // 3), CRATER_CORE_BGR, -1, lineType=cv2.LINE_AA)
        return frame
