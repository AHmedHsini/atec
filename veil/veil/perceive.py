"""Local see-step. A trained network names each blob. Nothing leaves the scout."""

from __future__ import annotations

import cv2
import numpy as np

from veil.camera import active_camera, camera_depth, geometric_sigma, pixel_to_ground, radius_m
from veil.dataset import features
from veil.model import HazardNet, load

_net: HazardNet | None = None
_polar = False
MAX_PROPOSALS_PER_MASK = 120
MAX_HAZARDS_PER_FRAME = 64


def use_polar() -> None:
    """Classify with the PolarDB network and its 64-pixel patch recipe."""
    global _polar, _net
    from veil.model import arm_polar

    _polar = True
    _net = arm_polar()


def detect(image: np.ndarray, scout_xy: tuple[float, float], yaw: float = 0.0) -> list[dict]:
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        return []
    net = _network()
    proposals = _proposals(image, scout_xy, yaw)
    found = []
    for proposal in proposals:
        vector = _vector(image, proposal["u"], proposal["v"])
        name, confidence = net.predict(vector)
        if name == "none" or confidence < 0.45:
            continue
        if _polar and name == "artifact":
            continue
        if _polar and confidence < 0.55:
            continue
        proposal["cls"] = name
        proposal["confidence"] = confidence
        found.append(proposal)
    found = _dedupe(found, gate=32.0 if _polar else 8.0)
    return sorted(found, key=lambda item: item["confidence"], reverse=True)[:MAX_HAZARDS_PER_FRAME]


def _network() -> HazardNet:
    global _net
    if _net is None:
        _net = load()
    return _net


def _vector(image: np.ndarray, u: float, v: float) -> np.ndarray:
    if not _polar:
        return features(image, u, v)
    from veil.polar import patch_features

    return patch_features(image, u, v)


def _proposals(image: np.ndarray, scout_xy: tuple[float, float], yaw: float = 0.0) -> list[dict]:
    if not _polar:
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        return _blobs(blurred < 60, scout_xy, yaw=yaw) + _blobs(blurred > 175, scout_xy, yaw=yaw)
    blur = cv2.GaussianBlur(image, (0, 0), 7)
    signed = image.astype(np.int16)
    dark = signed < blur.astype(np.int16) - 14
    bright = signed > blur.astype(np.int16) + 18
    native = 1936 * 1216
    min_area = 160
    max_area = None
    if image.size < native * 0.75:
        scale = image.size / native
        min_area = max(48, int(round(160 * scale)))
        max_area = int(0.006 * image.size)
    return (
        _blobs(dark, scout_xy, min_area=min_area, max_area=max_area, yaw=yaw)
        + _blobs(bright, scout_xy, min_area=min_area, max_area=max_area, yaw=yaw)
    )


def _dedupe(detections: list[dict], gate: float = 8.0) -> list[dict]:
    kept = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if any(np.hypot(detection["u"] - other["u"], detection["v"] - other["v"]) < gate for other in kept):
            continue
        kept.append(detection)
    return kept


def _blobs(mask: np.ndarray, scout_xy: tuple[float, float], min_area: int = 28, max_area: int | None = None, yaw: float = 0.0) -> list[dict]:
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:MAX_PROPOSALS_PER_MASK]
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area or (max_area is not None and area > max_area):
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        u = moments["m10"] / moments["m00"]
        v = moments["m01"] / moments["m00"]
        ground = pixel_to_ground(u, v, scout_xy, yaw)
        if ground is None:
            continue
        x, y, distance = ground
        radius_px = max(2.0, float(np.sqrt(area / np.pi)))
        depth = camera_depth(x, y, scout_xy, yaw)
        _old_radius, sized = radius_m(distance, radius_px, depth, active_camera().fx)
        sigma = geometric_sigma(u, v, scout_xy, yaw=yaw, seed=(int(u) * 10007 + int(v)) & 0xFFFFFFFF)
        found.append(
            {
                "cls": "none",
                "u": u,
                "v": v,
                "x": x,
                "y": y,
                "radius": sized,
                "sigma": sigma,
                "area": area,
            }
        )
    return found
