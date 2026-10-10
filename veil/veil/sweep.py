"""Loss sweep with no blackout, plus a separate blackout-recovery sweep."""

from __future__ import annotations

import numpy as np

from veil.mission import run

LOSS_RATES = (0.1, 0.3, 0.5, 0.7, 0.9)
SEEDS = tuple(range(10))
OPEN = (1.0e9, 1.0e9 + 1.0)
RECOVERY_BLACKOUT = (0.5, 6.0)
# No-blackout, 30% loss: every seed must deliver all 4 hazards. Higher loss is informational.
LOSS30_MIN_HAZARDS = 4


def run_sweep() -> dict:
    loss_rows = [_summarize_loss(loss, [_loss_run(loss, seed) for seed in SEEDS]) for loss in LOSS_RATES]
    recovery_runs = [_recovery_run(seed) for seed in SEEDS]
    stress = _stress_run()
    at_30 = next(row for row in loss_rows if row["loss"] == 0.3)
    latencies = [row["confirm_to_receipt_s_median"] for row in loss_rows if row["confirm_to_receipt_s_median"] is not None]
    varies = len(set(latencies)) > 1
    passed = at_30["hazards_delivered_min"] >= LOSS30_MIN_HAZARDS and varies
    return {
        "passed": passed,
        "latency_varies_with_loss": varies,
        "loss30_min_hazards": LOSS30_MIN_HAZARDS,
        "loss30_rule": "fail when the minimum delivered hazards at 30% loss with no blackout is below 4, or when receipt latency does not change across loss rates. 50% and above are informational",
        "pace_s": 0.004,
        "sender_holds_while_link_down": True,
        "pending_limit_s": 8.0,
        "max_attempts": 12,
        "loss": loss_rows,
        "blackout_recovery": _summarize_recovery(recovery_runs),
        "stress_blackout_until_end": stress,
    }


def _loss_run(loss: float, seed: int) -> dict:
    report = run(
        live=False,
        loss=loss,
        seed=seed,
        blackout=OPEN,
        scenario=f"loss-{loss}-{seed}",
        pace_s=0.004,
        exchange_s=0.004,
        settle_s=0.0,
        write_report=False,
    )
    return _pick(report)


def _recovery_run(seed: int) -> dict:
    report = run(
        live=False,
        loss=0.3,
        seed=seed,
        blackout=RECOVERY_BLACKOUT,
        scenario=f"recovery-{seed}",
        pace_s=0.004,
        exchange_s=0.004,
        settle_s=0.0,
        write_report=False,
    )
    picked = _pick(report)
    picked["first_hazard_recovery_s"] = report["first_hazard_recovery_s"]
    picked["all_queued_recovery_s"] = report["all_queued_recovery_s"]
    return picked


def _stress_run() -> dict:
    report = run(
        live=False,
        loss=0.3,
        seed=1,
        blackout=(0.0, 1.0e9),
        pending_limit_s=8.0,
        scenario="stress-closed",
        pace_s=0.004,
        exchange_s=0.0,
        settle_s=0.0,
        write_report=False,
    )
    return {
        "blackout_s": [0.0, 1.0e9],
        "pending_limit_s": 8.0,
        "hazards_delivered": report["hazards_delivered"],
        "lost_hazards": report["lost_hazards"],
        "wasted_blackout_bytes": report["wasted_blackout_bytes"],
        "note": "The link never reopens. Hazards still queued after pending_limit_s are dropped. The sender does not transmit while it knows the link is down, so wasted_blackout_bytes stays 0.",
    }


def _pick(report: dict) -> dict:
    return {
        "hazards_delivered": report["hazards_delivered"],
        "confirm_to_receipt_s": report["confirm_to_receipt_s"],
        "retransmissions_per_hazard": report["retransmissions_per_hazard"],
        "wasted_blackout_bytes": report["wasted_blackout_bytes"],
        "hazard_bytes": report["hazard_bytes"],
        "pose_bytes": report["pose_bytes"],
        "retransmission_bytes": report["retransmission_bytes"],
        "float32_bytes": report["float32_bytes"],
        "float32_confirmed_bytes": report["float32_confirmed_bytes"],
        "bytes_sent": report["bytes_sent"],
    }


def _summarize_loss(loss: float, runs: list[dict]) -> dict:
    delivered = [row["hazards_delivered"] for row in runs]
    latencies = [value for row in runs for value in row["confirm_to_receipt_s"]]
    retries = [row["retransmissions_per_hazard"] for row in runs]
    return {
        "loss": loss,
        "seeds": len(runs),
        "blackout": None,
        "hazards_delivered_mean": round(float(np.mean(delivered)), 3),
        "hazards_delivered_min": int(min(delivered)),
        "confirm_to_receipt_s_median": _stat(latencies, 50),
        "confirm_to_receipt_s_p95": _stat(latencies, 95),
        "retransmissions_per_hazard_mean": round(float(np.mean(retries)), 3),
        "wasted_blackout_bytes": int(sum(row["wasted_blackout_bytes"] for row in runs)),
        "hazard_bytes_mean": round(float(np.mean([row["hazard_bytes"] for row in runs])), 1),
        "pose_bytes_mean": round(float(np.mean([row["pose_bytes"] for row in runs])), 1),
        "retransmission_bytes_mean": round(float(np.mean([row["retransmission_bytes"] for row in runs])), 1),
        "bytes_sent_mean": round(float(np.mean([row["bytes_sent"] for row in runs])), 1),
        "float32_bytes_mean": round(float(np.mean([row["float32_bytes"] for row in runs])), 1),
        "float32_confirmed_bytes_mean": round(float(np.mean([row["float32_confirmed_bytes"] for row in runs])), 1),
        "bytes_per_delivered_hazard_mean": _bytes_per(runs),
        "informational": loss >= 0.5,
    }


def _summarize_recovery(runs: list[dict]) -> dict:
    first = [row["first_hazard_recovery_s"] for row in runs if row["first_hazard_recovery_s"] is not None]
    all_done = [row["all_queued_recovery_s"] for row in runs if row["all_queued_recovery_s"] is not None]
    delivered = [row["hazards_delivered"] for row in runs]
    return {
        "loss": 0.3,
        "blackout_s": list(RECOVERY_BLACKOUT),
        "seeds": len(runs),
        "hazards_delivered_mean": round(float(np.mean(delivered)), 3),
        "hazards_delivered_min": int(min(delivered)),
        "first_hazard_recovery_s_median": _stat(first, 50),
        "first_hazard_recovery_s_p95": _stat(first, 95),
        "all_queued_recovery_s_median": _stat(all_done, 50),
        "all_queued_recovery_s_p95": _stat(all_done, 95),
        "runs_with_both_times": len(all_done),
    }


def _bytes_per(runs: list[dict]):
    ratios = []
    for row in runs:
        if row["hazards_delivered"]:
            ratios.append(row["hazard_bytes"] / row["hazards_delivered"])
    if not ratios:
        return None
    return round(float(np.mean(ratios)), 1)


def _stat(values: list[float], percentile: float):
    if not values:
        return None
    return round(float(np.percentile(values, percentile)), 3)
