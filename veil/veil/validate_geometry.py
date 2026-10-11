"""Perturb the camera model and record ground error. Does not change weights."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from veil.camera import camera_depth, pixel_to_ground, project, radius_m, use_camera
from veil.mission import run
from veil.sim import HAZARDS

ROOT = Path(__file__).resolve().parents[1]


def _errors(pitch_deg=None, mast_h=None, pixel_noise=0.0, yaw_bias=0.0, along=0.0, cx_bias=0.0, seed=3) -> dict:
    rng = np.random.default_rng(seed)
    errors = []
    for hazard in HAZARDS:
        point = np.array([[hazard["x"], hazard["y"], 0.0]])
        u, v, _z = project(point, (0.0, 2.0))
        uu = float(u[0] + rng.normal(0, pixel_noise))
        vv = float(v[0] + rng.normal(0, pixel_noise))
        hit = pixel_to_ground(
            uu,
            vv,
            (0.0, 2.0 + along),
            yaw=np.deg2rad(yaw_bias),
            pitch=None if pitch_deg is None else np.deg2rad(18.0 + pitch_deg),
            mast_h=None if mast_h is None else 0.55 + mast_h,
            cx=160.0 + cx_bias,
        )
        if hit is None:
            errors.append(8.0)
            continue
        errors.append(float(np.hypot(hit[0] - hazard["x"], hit[1] - hazard["y"])))
    ordered = sorted(errors)
    return {
        "mean_m": round(float(np.mean(errors)), 4),
        "p95_m": round(float(np.percentile(errors, 95)), 4),
        "max_m": round(ordered[-1], 4),
    }


def _delivery(**kwargs) -> dict:
    report = run(
        live=False,
        loss=0.0,
        hold=0,
        frames=70,
        dt=0.1,
        blackout=(1e9, 1e9 + 1),
        scenario="geometry-probe",
        seed=26,
        write_report=False,
        pace_s=0,
        exchange_s=0,
        settle_s=0,
    )
    return {"delivered": report["hazards_delivered"], "false": report["false_hazards_delivered"]}


def radius_comparison() -> dict:
    rows = []
    for hazard in HAZARDS:
        scout = (0.0, 2.0)
        point = np.array([[hazard["x"], hazard["y"], 0.0]])
        edge = np.array([[hazard["x"] + hazard["r"], hazard["y"], 0.0]])
        u, v, depth = project(point, scout)
        ue, ve, _ze = project(edge, scout)
        radius_px = max(2.0, float(np.hypot(u[0] - ue[0], v[0] - ve[0])))
        hit = pixel_to_ground(float(u[0]), float(v[0]), scout)
        flat = 1.0 if hit is None else hit[2]
        depth_m = camera_depth(hit[0], hit[1], scout) if hit else float(depth[0])
        old, new = radius_m(flat, radius_px, depth_m)
        rows.append(
            {
                "name": hazard["name"],
                "truth_r": hazard["r"],
                "old_r": round(old, 4),
                "new_r": round(new, 4),
                "old_err": round(abs(old - hazard["r"]), 4),
                "new_err": round(abs(new - hazard["r"]), 4),
            }
        )
    old_mean = float(np.mean([row["old_err"] for row in rows]))
    new_mean = float(np.mean([row["new_err"] for row in rows]))
    return {"rows": rows, "old_mean_abs_err": round(old_mean, 4), "new_mean_abs_err": round(new_mean, 4), "kept": "pinhole depth" if new_mean <= old_mean else "flat range"}


def tall_bias(height_m: float = 0.25) -> dict:
    hazard = HAZARDS[0]
    scout = (0.0, 1.5)
    top = np.array([[hazard["x"], hazard["y"], height_m]])
    u, v, _z = project(top, scout)
    hit = pixel_to_ground(float(u[0]), float(v[0]), scout)
    if hit is None:
        return {"bias_m": None}
    bias = float(np.hypot(hit[0] - hazard["x"], hit[1] - hazard["y"]))
    corrected = bias
    k = 0.8
    assumed_h = k * hazard["r"]
    # Pull the hit back toward the scout by the parallax of the assumed height.
    if hit[2] > 0.2:
        scale = max(0.0, 1.0 - assumed_h / max(hit[2], 0.2))
        cx = scout[0] + (hit[0] - scout[0]) * scale
        cy = scout[1] + (hit[1] - scout[1]) * scale
        corrected = float(np.hypot(cx - hazard["x"], cy - hazard["y"]))
    return {
        "height_m": height_m,
        "bias_m": round(bias, 4),
        "corrected_bias_m": round(corrected, 4),
        "correction_helps": corrected < bias,
        "applied_in_flight": False,
    }


def coverage(samples: int = 40) -> dict:
    inside = 0
    grow = {}
    # Image v grows downward, so a smaller v is farther along the ground.
    for name, distance_row in (("far", 80), ("mid", 140), ("near", 190)):
        sigmas = []
        for trial in range(samples):
            from veil.camera import geometric_sigma

            sigma = geometric_sigma(160, distance_row, (0.0, 0.0), samples=12, seed=trial + distance_row)
            sigmas.append(sigma)
            truth = pixel_to_ground(160, distance_row, (0.0, 0.0))
            noisy = pixel_to_ground(160 + np.random.default_rng(trial).normal(0, 1), distance_row, (0.0, 0.0))
            if truth and noisy and np.hypot(truth[0] - noisy[0], truth[1] - noisy[1]) <= 2 * sigma:
                inside += 1
        grow[name] = round(float(np.median(sigmas)), 4)
    total = samples * 3
    return {
        "coverage_within_2sigma": round(inside / total, 3),
        "median_sigma_m": grow,
        "sigma_grows_with_distance": grow["far"] >= grow["near"],
    }


def main() -> None:
    use_camera(None)
    cases = [
        ("nominal", {}),
        ("pitch+0.5", {"pitch_deg": 0.5}),
        ("pitch+1", {"pitch_deg": 1}),
        ("pitch+2", {"pitch_deg": 2}),
        ("pitch-1", {"pitch_deg": -1}),
        ("mast+2cm", {"mast_h": 0.02}),
        ("mast-5cm", {"mast_h": -0.05}),
        ("pixel 1px", {"pixel_noise": 1}),
        ("pixel 2px", {"pixel_noise": 2}),
        ("heading 2deg", {"yaw_bias": 2}),
        ("along 5cm", {"along": 0.05}),
        ("principal +3px", {"cx_bias": 3}),
    ]
    rows = []
    for name, kwargs in cases:
        stats = _errors(**kwargs)
        stats["name"] = name
        rows.append(stats)
    nominal_delivery = _delivery()
    payload = {
        "perturbations": rows,
        "delivery_nominal": nominal_delivery,
        "delivery_note": "Delivery is the unperturbed flight camera. Perturbation rows are localization error on the same four hazards, not a retuned model.",
        "radius": radius_comparison(),
        "tall_rock": tall_bias(),
        "sigma_coverage": coverage(),
    }
    out = ROOT / "artifacts" / "geometry" / "perturbation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["| case | mean m | p95 m | max m |", "| --- | --- | --- | --- |"]
    for row in rows:
        lines.append(f"| {row['name']} | {row['mean_m']} | {row['p95_m']} | {row['max_m']} |")
    (ROOT / "artifacts" / "geometry" / "perturbation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out), "radius_kept": payload["radius"]["kept"], "coverage": payload["sigma_coverage"]}, indent=2))


if __name__ == "__main__":
    main()
