"""Ablations. A retrained variant is written beside the flight weights, never over them."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from veil.dataset import NAMES, generate
from veil.model import HIDDEN, WEIGHTS, load
from veil.perceive import detect
from veil.sim import HAZARDS, render_pose
from veil.tracks import Tracker

ROOT = Path(__file__).resolve().parents[1]
ASIDE = WEIGHTS.with_name("hazard_net_no_position.npz")


def _fit(train_x, train_y, val_x, val_y, seed: int) -> dict:
    mean = train_x.mean(axis=0)
    std = train_x.std(axis=0)
    std[std < 1e-6] = 1.0
    x = (train_x - mean) / std
    vx = (val_x - mean) / std
    rng = np.random.default_rng(seed)
    inputs = x.shape[1]
    outputs = len(NAMES)
    w1 = rng.normal(0, np.sqrt(2 / inputs), size=(inputs, HIDDEN)).astype(np.float32)
    b1 = np.zeros(HIDDEN, np.float32)
    w2 = rng.normal(0, np.sqrt(2 / HIDDEN), size=(HIDDEN, outputs)).astype(np.float32)
    b2 = np.zeros(outputs, np.float32)
    lr = 0.08
    for _epoch in range(48):
        order = rng.permutation(len(x))
        for start in range(0, len(x), 64):
            idx = order[start : start + 64]
            xb, yb = x[idx], train_y[idx]
            hidden = np.maximum(xb @ w1 + b1, 0)
            scores = hidden @ w2 + b2
            scores -= scores.max(axis=1, keepdims=True)
            exp = np.exp(scores)
            probs = exp / exp.sum(axis=1, keepdims=True)
            target = np.zeros_like(probs)
            target[np.arange(len(yb)), yb] = 1.0
            grad_scores = (probs - target) / len(yb)
            grad_w2 = hidden.T @ grad_scores
            grad_b2 = grad_scores.sum(axis=0)
            grad_hidden = grad_scores @ w2.T
            grad_hidden[hidden <= 0] = 0
            w2 -= lr * grad_w2.astype(np.float32)
            b2 -= lr * grad_b2.astype(np.float32)
            w1 -= lr * (xb.T @ grad_hidden).astype(np.float32)
            b1 -= lr * grad_hidden.sum(axis=0).astype(np.float32)
        lr *= 0.96
    hidden = np.maximum(vx @ w1 + b1, 0)
    guessed = np.argmax(hidden @ w2 + b2, axis=1)
    accuracy = float((guessed == val_y).mean())
    np.savez(
        ASIDE,
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        val_accuracy=np.array(accuracy, np.float32),
        samples=np.array(len(train_x), np.int32),
    )
    return {"val_accuracy": round(accuracy, 3), "path": str(ASIDE), "replaced_flight_weights": False}


def _count(require_consistency: bool, use_net: bool) -> dict:
    tracker = Tracker(require_consistency=require_consistency)
    rng = np.random.default_rng(1)
    for frame in range(80):
        scout = (0.0, frame * 0.05)
        image = render_pose(scout, rng)
        if use_net:
            found = [item for item in detect(image, scout) if item["cls"] in ("rock", "pit")]
        else:
            import cv2

            from veil.perceive import _blobs

            blurred = cv2.GaussianBlur(image, (3, 3), 0)
            found = _blobs(blurred < 60, scout) + _blobs(blurred > 175, scout)
            for item in found:
                item["cls"] = "pit" if item["v"] > 120 else "rock"
        tracker.update(found, scout[1], frame * 0.1)
        tracker.drop_stale()
    confirmed = {(track.cls, round(track.x, 1), round(track.y, 1)) for track in tracker.confirmed()}
    return {
        "confirmed": len(tracker.confirmed()),
        "rejected": len(tracker.rejected()),
        "names": sorted(track.cls for track in tracker.confirmed()),
        "distinct_positions": len(confirmed),
    }


def main() -> None:
    flight = load()
    flight_acc = round(flight.val_accuracy, 3)
    train_x, train_y = generate(280, 7)
    val_x, val_y = generate(80, 1007)
    train_x = train_x.copy()
    val_x = val_x.copy()
    train_x[:, -2:] = 0
    val_x[:, -2:] = 0
    aside = _fit(train_x, train_y, val_x, val_y, 7)
    # Reload flight net into the cache. The ablation file is separate.
    from veil import model as model_module

    model_module._cached = None
    reloaded = load()
    payload = {
        "flight_val_accuracy": round(reloaded.val_accuracy, 3),
        "flight_weights_unchanged": WEIGHTS.exists() and round(reloaded.val_accuracy, 3) == flight_acc,
        "threshold_only": _count(True, False),
        "full": _count(True, True),
        "without_geometric_test": _count(False, True),
        "without_position_features": aside,
        "position_feature_note": "The last two features are normalized pixel position. The flight network still uses them. The aside file is not loaded by the mission.",
    }
    out = ROOT / "artifacts" / "ablation" / "ablation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"flight": payload["flight_val_accuracy"], "aside": aside["val_accuracy"], "unchanged": payload["flight_weights_unchanged"]}, indent=2))


if __name__ == "__main__":
    main()
