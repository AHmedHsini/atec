"""Labeled camera patches for the onboard hazard network.

Every sample is rendered on the scout, then tagged from the object that
was painted there. The flight rocks and pits are not the only examples:
each frame also invents new ones so the network learns the shape, not
four memorized coordinates.
"""

from __future__ import annotations

import cv2
import numpy as np

from veil.camera import HEIGHT, WIDTH, project
from veil.sim import HALLUCINATION_RADIUS, HALLUCINATION_UV, HAZARDS, render_scene, scout_pose

PATCH = 16
NAMES = ("none", "rock", "pit", "artifact")


def features(
    image: np.ndarray,
    u: float,
    v: float,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> np.ndarray:
    """262-d patch vector. width/height only scale the u,v position scalars."""
    half = PATCH // 2
    padded = np.pad(image, half, mode="edge")
    x = int(round(u)) + half
    y = int(round(v)) + half
    crop = padded[y - half : y + half, x - half : x + half].astype(np.float32) / 255.0
    flat = crop.reshape(-1)
    return np.concatenate(
        [
            flat,
            np.array(
                [
                    float(flat.mean()),
                    float(flat.std()),
                    float((flat < 0.25).mean()),
                    float((flat > 0.75).mean()),
                    u / width,
                    v / height,
                ],
                dtype=np.float32,
            ),
        ]
    )


def generate(n_frames: int = 280, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples: dict[int, list[np.ndarray]] = {0: [], 1: [], 2: [], 3: []}
    for frame in range(n_frames):
        scout = (float(rng.uniform(-1.4, 1.4)), float(rng.uniform(0.0, 8.2)))
        yaw = float(rng.uniform(-0.2, 0.2))
        hazards = [_random_hazard(rng, scout) for _ in range(int(rng.integers(2, 5)))]
        if frame % 4 == 0:
            scout = scout_pose(frame % 130)
            yaw = 0.0
            hazards = [dict(hazard) for hazard in HAZARDS]
        image = render_scene(scout, hazards, yaw=yaw, artifact=False, grain_seed=int(rng.integers(0, 2**31 - 1)))
        gain = float(rng.uniform(0.72, 1.28))
        bias = float(rng.integers(-18, 19))
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
        for hazard in hazards:
            pixel = _visible(hazard, scout, yaw)
            if pixel is None:
                continue
            label = 1 if hazard["cls"] == "rock" else 2
            samples[label].append(features(image, pixel[0], pixel[1]))
        cv2.circle(image, HALLUCINATION_UV, HALLUCINATION_RADIUS, 12, -1)
        samples[3].append(features(image, HALLUCINATION_UV[0], HALLUCINATION_UV[1]))
        if frame % 2 == 0:
            _cover_lens(samples, rng, scout)
        if frame % 5 == 0:
            late = scout_pose(100 + (frame % 30))
            late_image = render_scene(late, HAZARDS, artifact=True, grain_seed=frame)
            samples[3].append(features(late_image, HALLUCINATION_UV[0], HALLUCINATION_UV[1]))
            for hazard in HAZARDS:
                pixel = _visible(hazard, late, 0.0)
                if pixel is None:
                    continue
                if np.hypot(pixel[0] - HALLUCINATION_UV[0], pixel[1] - HALLUCINATION_UV[1]) < 18:
                    continue
                label = 1 if hazard["cls"] == "rock" else 2
                samples[label].append(features(late_image, pixel[0], pixel[1]))
        for _ in range(3):
            pixel = _empty_pixel(rng, hazards, scout, yaw)
            if pixel is not None:
                samples[0].append(features(image, pixel[0], pixel[1]))
    count = min(len(samples[label]) for label in samples)
    chosen_x = []
    chosen_y = []
    for label in samples:
        order = rng.permutation(len(samples[label]))[:count]
        for index in order:
            chosen_x.append(samples[label][index])
            chosen_y.append(label)
    return np.stack(chosen_x).astype(np.float32), np.array(chosen_y, dtype=np.int64)


def _cover_lens(samples: dict[int, list[np.ndarray]], rng: np.random.Generator, scout: tuple[float, float]) -> None:
    """A dark pit can sit under the lens spot. That spot is still the lens."""
    cover = {
        "cls": "pit" if rng.random() < 0.7 else "rock",
        "x": scout[0] + float(rng.uniform(-0.15, 0.55)),
        "y": scout[1] + float(rng.uniform(1.3, 3.2)),
        "r": float(rng.uniform(0.4, 0.8)),
    }
    image = render_scene(scout, [cover], artifact=False, grain_seed=int(rng.integers(0, 2**31 - 1)))
    pixel = _visible(cover, scout, 0.0)
    if pixel is not None and np.hypot(pixel[0] - HALLUCINATION_UV[0], pixel[1] - HALLUCINATION_UV[1]) > 24:
        label = 1 if cover["cls"] == "rock" else 2
        samples[label].append(features(image, pixel[0], pixel[1]))
    cv2.circle(image, HALLUCINATION_UV, HALLUCINATION_RADIUS, 12, -1)
    samples[3].append(features(image, HALLUCINATION_UV[0], HALLUCINATION_UV[1]))


def _random_hazard(rng: np.random.Generator, scout: tuple[float, float]) -> dict:
    return {
        "cls": "pit" if rng.random() < 0.5 else "rock",
        "x": float(rng.uniform(-2.4, 2.4)),
        "y": float(rng.uniform(scout[1] + 0.9, scout[1] + 7.0)),
        "r": float(rng.uniform(0.16, 0.55)),
    }


def _visible(hazard: dict, scout: tuple[float, float], yaw: float) -> tuple[float, float] | None:
    u, v, z_cam = project(np.array([[hazard["x"], hazard["y"], 0.0]]), scout, yaw)
    if z_cam[0] <= 0.5:
        return None
    if not (10 <= u[0] < WIDTH - 10 and 10 <= v[0] < HEIGHT - 10):
        return None
    return float(u[0]), float(v[0])


def _empty_pixel(rng: np.random.Generator, hazards, scout, yaw) -> tuple[float, float] | None:
    occupied = []
    for hazard in hazards:
        pixel = _visible(hazard, scout, yaw)
        if pixel is not None:
            occupied.append(pixel)
    occupied.append(HALLUCINATION_UV)
    for _ in range(12):
        u = float(rng.uniform(12, WIDTH - 12))
        v = float(rng.uniform(30, HEIGHT - 12))
        if all(np.hypot(u - ou, v - ov) > 22 for ou, ov in occupied):
            return u, v
    return None
