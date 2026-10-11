"""Tracking on random worlds. Thresholds are not tuned on these seeds."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from veil.perceive import detect
from veil.sim import render_scene
from veil.tracks import Tracker

ROOT = Path(__file__).resolve().parents[1]


def _world(rng: np.random.Generator) -> tuple[list[dict], list[tuple[int, int]]]:
    hazards = []
    count = int(rng.integers(2, 9))
    for index in range(count):
        hazards.append(
            {
                "name": f"h{index}",
                "cls": "pit" if rng.random() < 0.5 else "rock",
                "x": float(rng.uniform(-1.6, 1.6)),
                "y": float(rng.uniform(2.2, 7.4)),
                "r": float(rng.uniform(0.18, 0.45)),
            }
        )
    decoys = [(int(rng.integers(20, 300)), int(rng.integers(40, 200))) for _ in range(int(rng.integers(1, 4)))]
    return hazards, decoys


def _score_world(seed: int, frames: int = 50) -> dict:
    rng = np.random.default_rng(seed)
    hazards, decoys = _world(rng)
    tracker = Tracker()
    false_blobs = 0
    for frame in range(frames):
        scout = (0.0, frame * 0.05)
        image = render_scene(scout, hazards, artifact=False, grain_seed=seed + frame)
        for u, v in decoys:
            cv2.circle(image, (u, v), 6, 14, -1)
        cv2.circle(image, (200, 160), 4, 230, -1)
        found = detect(image, scout)
        false_blobs += sum(1 for item in found if item["cls"] == "artifact")
        tracker.update([item for item in found if item["cls"] in ("rock", "pit")], scout[1], frame * 0.1)
        tracker.drop_stale()
    truth = {(item["cls"], item["x"], item["y"]) for item in hazards}
    matched = []
    used = set()
    false_count = 0
    for track in tracker.confirmed():
        best = None
        best_d = 0.6
        for index, hazard in enumerate(hazards):
            if index in used or hazard["cls"] != track.cls:
                continue
            distance = float(np.hypot(track.x - hazard["x"], track.y - hazard["y"]))
            if distance < best_d:
                best = index
                best_d = distance
        if best is None:
            false_count += 1
        else:
            used.add(best)
            matched.append(best_d)
    return {
        "seed": seed,
        "truth": len(hazards),
        "hit": len(matched),
        "false": false_count,
        "rejected": len(tracker.rejected()),
        "errors": matched,
    }


def summarize(seeds: range) -> dict:
    rows = [_score_world(seed) for seed in seeds]
    truth = sum(row["truth"] for row in rows)
    hit = sum(row["hit"] for row in rows)
    false_count = sum(row["false"] for row in rows)
    errors = [value for row in rows for value in row["errors"]]
    precision = hit / (hit + false_count) if hit + false_count else 0.0
    recall = hit / truth if truth else 0.0
    return {
        "worlds": len(rows),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "mean_error_m": None if not errors else round(float(np.mean(errors)), 4),
        "p95_error_m": None if not errors else round(float(np.percentile(errors, 95)), 4),
        "false_hazards": false_count,
        "rejected_tracks": sum(row["rejected"] for row in rows),
        "tuned_on_these_seeds": False,
    }


def noise_report() -> dict:
    rng = np.random.default_rng(4)
    hazards = [
        {"name": "a", "cls": "rock", "x": -0.8, "y": 3.5, "r": 0.3},
        {"name": "b", "cls": "pit", "x": 0.4, "y": 4.6, "r": 0.35},
    ]
    conditions = {}
    for name, apply in (
        ("clean", lambda image: image),
        ("gaussian", lambda image: np.clip(image.astype(np.int16) + rng.normal(0, 12, image.shape), 0, 255).astype(np.uint8)),
        ("bright", lambda image: np.clip(image.astype(np.float32) * 1.3, 0, 255).astype(np.uint8)),
        ("dark", lambda image: np.clip(image.astype(np.float32) * 0.7, 0, 255).astype(np.uint8)),
        ("jpeg", lambda image: cv2.imdecode(cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 40])[1], cv2.IMREAD_GRAYSCALE)),
        ("blur", lambda image: cv2.GaussianBlur(image, (5, 5), 0)),
    ):
        tracker = Tracker()
        for frame in range(40):
            scout = (0.0, frame * 0.05)
            image = apply(render_scene(scout, hazards, artifact=True, grain_seed=frame))
            found = detect(image, scout)
            tracker.update([item for item in found if item["cls"] in ("rock", "pit")], scout[1], frame * 0.1)
            tracker.drop_stale()
        names = {(track.cls, round(track.x, 1), round(track.y, 1)) for track in tracker.confirmed()}
        conditions[name] = {"confirmed": len(tracker.confirmed()), "rejected": len(tracker.rejected()), "classes": sorted(track.cls for track in tracker.confirmed())}
        del names
    return conditions


def main() -> None:
    payload = {"random_worlds": summarize(range(50)), "noise": noise_report()}
    out = ROOT / "artifacts" / "robustness" / "random_worlds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["random_worlds"], indent=2))


if __name__ == "__main__":
    main()
