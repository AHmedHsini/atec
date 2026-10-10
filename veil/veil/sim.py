"""Synthetic crater traverse with known rocks, pits, and a stuck shadow blob."""

from __future__ import annotations

import cv2
import numpy as np

from veil.camera import HEIGHT, WIDTH, project

HAZARDS = (
    {"name": "rock-a", "cls": "rock", "x": -1.0, "y": 4.2, "r": 0.28},
    {"name": "pit-a", "cls": "pit", "x": 0.35, "y": 5.0, "r": 0.42},
    {"name": "rock-b", "cls": "rock", "x": 0.95, "y": 6.1, "r": 0.24},
    {"name": "pit-b", "cls": "pit", "x": -0.55, "y": 7.3, "r": 0.38},
)

# A dark blob glued to the image. It is not on the ground.
HALLUCINATION_UV = (36, 110)
HALLUCINATION_RADIUS = 7


def scout_pose(frame: int, step_m: float = 0.05) -> tuple[float, float]:
    return 0.0, frame * step_m


def render(frame: int, rng: np.random.Generator, step_m: float = 0.05) -> np.ndarray:
    return render_pose(scout_pose(frame, step_m), rng)


def render_pose(scout: tuple[float, float], rng: np.random.Generator, yaw: float = 0.0) -> np.ndarray:
    del rng
    return render_scene(scout, HAZARDS, yaw=yaw, artifact=True)


def render_scene(
    scout: tuple[float, float],
    hazards,
    yaw: float = 0.0,
    artifact: bool = True,
    grain_seed: int | None = None,
) -> np.ndarray:
    """Gray camera frame. Training uses this with hazards that are not the flight set."""
    image = _ground()
    for hazard in hazards:
        _paint(image, hazard, scout, yaw=yaw)
    if artifact:
        cv2.circle(image, HALLUCINATION_UV, HALLUCINATION_RADIUS, 12, -1)
    cv2.drawMarker(image, (WIDTH // 2, HEIGHT // 2), 80, cv2.MARKER_CROSS, 18, 1)
    if grain_seed is None:
        grain_seed = (int(round(scout[0] * 50)) * 10007 + int(round(scout[1] * 50))) & 0xFFFFFFFF
    grain = np.random.default_rng(grain_seed).integers(-2, 3, size=image.shape, dtype=np.int16)
    return np.clip(image.astype(np.int16) + grain, 0, 255).astype(np.uint8)


def _ground() -> np.ndarray:
    image = np.empty((HEIGHT, WIDTH), np.uint8)
    horizon = 24
    image[:horizon] = 36
    rows = np.arange(HEIGHT)
    shade = np.clip(165 - (rows - horizon) * 0.28, 70, 165).astype(np.uint8)
    image[horizon:] = shade[horizon:, None]
    return image


def render_color(scout: tuple[float, float], yaw: float = 0.0) -> np.ndarray:
    image = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    horizon = 24
    image[:horizon] = (42, 36, 28)
    rows = np.arange(horizon, HEIGHT)
    fade = np.clip(210 - (rows - horizon) * 0.35, 90, 210)
    image[horizon:, :, 0] = (fade * 0.62).astype(np.uint8)[:, None]
    image[horizon:, :, 1] = (fade * 0.78).astype(np.uint8)[:, None]
    image[horizon:, :, 2] = fade.astype(np.uint8)[:, None]
    for hazard in HAZARDS:
        color = (40, 90, 230) if hazard["cls"] == "pit" else (210, 190, 40)
        _paint(image, hazard, scout, color, yaw)
    cv2.circle(image, HALLUCINATION_UV, HALLUCINATION_RADIUS, (20, 20, 20), -1)
    cv2.drawMarker(image, (WIDTH // 2, HEIGHT // 2), (245, 245, 245), cv2.MARKER_CROSS, 18, 1)
    return image


def _paint(image: np.ndarray, hazard: dict, scout: tuple[float, float], color=None, yaw: float = 0.0) -> None:
    center = np.array([[hazard["x"], hazard["y"], 0.0]])
    edge = np.array([[hazard["x"] + hazard["r"], hazard["y"], 0.0]])
    u, v, z_cam = project(center, scout, yaw)
    ue, ve, ze = project(edge, scout, yaw)
    if z_cam[0] <= 0.4 or ze[0] <= 0.2:
        return
    if not (0 <= u[0] < WIDTH and 0 <= v[0] < HEIGHT):
        return
    radius = int(round(math_hypot(u[0] - ue[0], v[0] - ve[0])))
    radius = max(3, radius)
    if color is None:
        color = 18 if hazard["cls"] == "pit" else 228
    cv2.circle(image, (int(round(u[0])), int(round(v[0]))), radius, color, -1)
    if hazard["cls"] == "pit" and image.ndim == 2:
        cv2.circle(image, (int(round(u[0])), int(round(v[0]))), radius, 70, 1)


def math_hypot(a: float, b: float) -> float:
    return float(np.hypot(a, b))
