"""Local see-step. A trained network names each blob. Nothing leaves the scout."""

from __future__ import annotations

import cv2
import numpy as np

from veil.camera import active_camera, camera_depth, geometric_sigma, pixel_to_ground, radius_m
from veil.dataset import features
from veil.model import HazardNet, load

_net: HazardNet | None = None


def detect(image: np.ndarray, scout_xy: tuple[float, float]) -> list[dict]:
    if not isinstance(image, np.ndarray) or image.ndim != 2 or image.size == 0:
        return []
    net = _network()
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    proposals = _blobs(blurred < 60, scout_xy) + _blobs(blurred > 175, scout_xy)
    found = []
    for proposal in proposals:
        name, confidence = net.predict(features(image, proposal["u"], proposal["v"]))
        if name == "none" or confidence < 0.45:
            continue
        proposal["cls"] = name
        proposal["confidence"] = confidence
        found.append(proposal)
    return _dedupe(found)


def _network() -> HazardNet:
    global _net
    if _net is None:
        _net = load()
    return _net


def _dedupe(detections: list[dict]) -> list[dict]:
    kept = []
    for detection in sorted(detections, key=lambda item: item["confidence"], reverse=True):
        if any(np.hypot(detection["u"] - other["u"], detection["v"] - other["v"]) < 8 for other in kept):
            continue
        kept.append(detection)
    return kept


def _blobs(mask: np.ndarray, scout_xy: tuple[float, float]) -> list[dict]:
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 28:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] <= 1e-6:
            continue
        u = moments["m10"] / moments["m00"]
        v = moments["m01"] / moments["m00"]
        ground = pixel_to_ground(u, v, scout_xy)
        if ground is None:
            continue
        x, y, distance = ground
        radius_px = max(2.0, float(np.sqrt(area / np.pi)))
        depth = camera_depth(x, y, scout_xy)
        _old_radius, sized = radius_m(distance, radius_px, depth, active_camera().fx)
        sigma = geometric_sigma(u, v, scout_xy, seed=(int(u) * 10007 + int(v)) & 0xFFFFFFFF)
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
