"""Build the training set and fit the onboard hazard network."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from veil.camera import project
from veil.dataset import NAMES, features, generate

DATA = Path(__file__).resolve().parents[1] / "outputs" / "train_set.npz"
from veil.model import WEIGHTS, train
from veil.sim import HAZARDS, render_pose

REPORT = Path(__file__).resolve().parents[1] / "outputs" / "train_report.json"


def main() -> None:
    train_x, train_y = generate(280, 7)
    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez(DATA, x=train_x, y=train_y, names=np.array(NAMES))
    counts = {NAMES[label]: int((train_y == label).sum()) for label in range(len(NAMES))}
    net = train(save=True)
    seen = {hazard["name"]: {"shown": 0, "named": 0} for hazard in HAZARDS}
    for frame in range(0, 120, 2):
        scout = (0.0, frame * 0.05)
        image = render_pose(scout, np.random.default_rng(0))
        for hazard in HAZARDS:
            u, v, z_cam = project(np.array([[hazard["x"], hazard["y"], 0.0]]), scout)
            if z_cam[0] <= 0.5 or not (10 <= u[0] < 310 and 10 <= v[0] < 230):
                continue
            seen[hazard["name"]]["shown"] += 1
            name, _confidence = net.predict(features(image, float(u[0]), float(v[0])))
            if name == hazard["cls"]:
                seen[hazard["name"]]["named"] += 1
    report = {
        "samples": int(len(train_x)),
        "per_class": counts,
        "val_accuracy": round(net.val_accuracy, 3),
        "weights": str(WEIGHTS),
        "flight_hazards": seen,
    }
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
