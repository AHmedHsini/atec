"""Build the training set and fit the onboard hazard network."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from veil.camera import project
from veil.dataset import NAMES, features, generate
from veil.model import WEIGHTS, train
from veil.sim import HAZARDS, render_pose

DATA = Path(__file__).resolve().parents[1] / "outputs" / "train_set.npz"
REPORT = Path(__file__).resolve().parents[1] / "outputs" / "train_report.json"
POLAR_WEIGHTS = Path(__file__).resolve().parents[1] / "outputs" / "hazard_net_polar.npz"
POLAR_REPORT = Path(__file__).resolve().parents[1] / "outputs" / "polar_train_report.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the onboard HazardNet")
    parser.add_argument(
        "--polar",
        action="store_true",
        help="Train on NASA PolarDB patches; write hazard_net_polar.npz (keeps synthetic hazard_net.npz)",
    )
    args = parser.parse_args()
    if args.polar:
        _train_polar()
    else:
        _train_synthetic()


def _train_synthetic() -> None:
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


def _train_polar() -> None:
    from veil.polar import generate_polar

    train_x, train_y, val_x, val_y, meta = generate_polar()
    net = train(
        save=True,
        seed=11,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        path=POLAR_WEIGHTS,
        epochs=100,
        weight_decay=1e-4,
        lr=0.06,
    )
    from veil.polar import POLAR_ROOT

    download_sizes = {}
    sizes_file = POLAR_ROOT / "download_sizes.txt"
    if sizes_file.exists():
        for line in sizes_file.read_text(encoding="utf-8").splitlines():
            parts = line.split("\t")
            if len(parts) == 2 and parts[0] != "TOTAL":
                download_sizes[parts[1]] = int(parts[0])
    report = {
        "samples": int(len(train_x)),
        "val_samples": int(len(val_x)),
        "per_class": meta["train_per_class"],
        "val_per_class": meta["val_per_class"],
        "per_class_before_balance": meta["train_per_class_before_balance"],
        "val_accuracy": round(float(net.val_accuracy), 4),
        "confusion": net.confusion,
        "class_names": list(NAMES),
        "terrains_train": meta["train_terrains"],
        "terrains_held_out": meta["val_terrains"],
        "val_only_terrains": meta.get("val_only_terrains"),
        "holdout": meta.get("holdout"),
        "train_frames": len(meta["train_frames"]),
        "val_frames": len(meta["val_frames"]),
        "label_rule": meta["label_rule"],
        "download_bytes": download_sizes,
        "download_total_bytes": int(sum(download_sizes.values())),
        "weights": "veil/outputs/hazard_net_polar.npz",
        "onboard_weights": "veil/outputs/hazard_net.npz",
        "onboard_note": (
            "Keep veil/outputs/hazard_net.npz for the scripted mission / synthetic tests. "
            "Use hazard_net_polar.npz when evaluating PolarDB / lunar-analog imagery."
        ),
    }
    POLAR_REPORT.parent.mkdir(parents=True, exist_ok=True)
    POLAR_REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
