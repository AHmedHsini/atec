"""Local hazard detector. No network and no cloud APIs. Owner: Ahmed.

Baseline is classical color and shape segmentation, which the spec allows and
which stays inside a few milliseconds per frame. Swap `detect_hazards` for a
small onboard model later; keep the Detection fields stable.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from crater.scout.scene import CRATER, ICE, ROCK


@dataclass(frozen=True)
class Detection:
    kind: str
    u: float
    v: float
    confidence: float


def detect_hazards(frame: np.ndarray) -> list[Detection]:
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("expected a BGR frame")
    blue, green, red = cv2.split(frame)
    found: list[Detection] = []
    found.extend(_blobs(_ice_mask(blue, green, red), ICE, frame.shape))
    found.extend(_blobs(_rock_mask(blue, green, red), ROCK, frame.shape))
    found.extend(_blobs(_crater_mask(blue, green, red), CRATER, frame.shape))
    return found


def _ice_mask(blue: np.ndarray, green: np.ndarray, red: np.ndarray) -> np.ndarray:
    return ((blue > 220) & (green > 215) & (red > 200)).astype(np.uint8) * 255


def _rock_mask(blue: np.ndarray, green: np.ndarray, red: np.ndarray) -> np.ndarray:
    return ((red > 80) & (green < 60) & (blue < 55)).astype(np.uint8) * 255


def _crater_mask(blue: np.ndarray, green: np.ndarray, red: np.ndarray) -> np.ndarray:
    rim = (red > 145) & (green > 145) & (blue > 140) & (red < 210)
    return rim.astype(np.uint8) * 255


def _blobs(mask: np.ndarray, kind: str, shape: tuple[int, ...]) -> list[Detection]:
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    height, width = shape[:2]
    found: list[Detection] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 20:
            continue
        cu, cv = centroids[index]
        if not (0 <= cu < width and 0 <= cv < height):
            continue
        # Area saturates confidence; tiny specks stay, but they do not look certain.
        confidence = max(0.55, min(0.97, 0.55 + area / 900.0))
        found.append(Detection(kind, float(cu), float(cv), confidence))
    return found
