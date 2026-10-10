"""Run the automated acceptance suite and write the human summary from its JSON."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from veil.reports import write_json
from veil.sweep import run_sweep

ROOT = Path(__file__).resolve().parents[1]
ACCEPTANCE = ROOT / "artifacts" / "acceptance"


def _need(module: str) -> None:
    try:
        __import__(module)
    except ImportError as exc:
        raise SystemExit(f"missing dependency {module}. Install veil/requirements.txt") from exc


def main() -> int:
    _need("numpy")
    _need("cv2")
    weights = ROOT / "outputs" / "hazard_net.npz"
    if not weights.exists():
        raise SystemExit("missing outputs/hazard_net.npz. Run python -m veil.train")
    code = pytest.main(["-q", str(ROOT / "tests")])
    sweep = run_sweep()
    write_json(ACCEPTANCE / "loss_sweep.json", sweep)
    if not sweep.get("passed"):
        code = 1 if code == 0 else code
    summary = _summary(sweep)
    ACCEPTANCE.mkdir(parents=True, exist_ok=True)
    (ACCEPTANCE / "SUMMARY.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return int(code)


def _summary(sweep: dict | None = None) -> str:
    long_path = ACCEPTANCE / "acceptance-long.json"
    short_path = ACCEPTANCE / "acceptance-short.json"
    if not long_path.exists() or not short_path.exists():
        return "Acceptance JSON was not written. The suite did not finish a mission run.\n"
    long_report = json.loads(long_path.read_text(encoding="utf-8"))
    short_report = json.loads(short_path.read_text(encoding="utf-8"))
    failed = []
    for report in (short_report, long_report):
        if not report.get("passed"):
            failed.append(report.get("scenario", "unknown"))
    lines = [
        "VEIL acceptance summary",
        f"short passed: {short_report.get('passed')}  long passed: {long_report.get('passed')}",
        f"long hazards: {long_report.get('hazards_delivered')}/{long_report.get('hazards_expected')} false: {long_report.get('false_hazards_delivered')}",
        f"long rejected inconsistent: {long_report.get('rejected_inconsistent')} expired: {long_report.get('expired_tracks')}",
        f"long mean error m: {long_report.get('mean_error_m')} max error m: {long_report.get('max_error_m')}",
        f"long bytes sent: {long_report.get('bytes_sent')} naive json: {long_report.get('naive_json_bytes')} float32 all detections: {long_report.get('float32_bytes')} float32 confirmed: {long_report.get('float32_confirmed_bytes')}",
        f"long hazard bytes: {long_report.get('hazard_bytes')} pose bytes: {long_report.get('pose_bytes')} heartbeat bytes: {long_report.get('heartbeat_bytes')} retransmission bytes: {long_report.get('retransmission_bytes')}",
        f"long bytes per delivered hazard: {long_report.get('bytes_per_delivered_hazard')} wasted during blackout: {long_report.get('wasted_blackout_bytes')}",
        f"long fps: {long_report.get('fps')} model accuracy: {long_report.get('model_val_accuracy')}",
        f"failed scenarios: {failed or 'none'}",
    ]
    if sweep:
        lines.append(
            f"loss sweep passed: {sweep.get('passed')} "
            f"(30% min hazards >= {sweep.get('loss30_min_hazards')}, latency varies: {sweep.get('latency_varies_with_loss')})"
        )
        lines.append("loss  mean  min  receipt_median_s  receipt_p95_s  retx_per_hazard  hazard_B  pose_B  retx_B  B_per_hazard  informational")
        for row in sweep.get("loss", []):
            lines.append(
                f"{row['loss']:.1f}  {row['hazards_delivered_mean']}  {row['hazards_delivered_min']}  "
                f"{row['confirm_to_receipt_s_median']}  {row['confirm_to_receipt_s_p95']}  "
                f"{row['retransmissions_per_hazard_mean']}  {row['hazard_bytes_mean']}  {row['pose_bytes_mean']}  "
                f"{row['retransmission_bytes_mean']}  {row['bytes_per_delivered_hazard_mean']}  {row['informational']}"
            )
        recovery = sweep.get("blackout_recovery") or {}
        lines.append(
            "recovery loss 0.3 blackout "
            f"{recovery.get('blackout_s')} first_median_s {recovery.get('first_hazard_recovery_s_median')} "
            f"all_median_s {recovery.get('all_queued_recovery_s_median')} "
            f"delivered_min {recovery.get('hazards_delivered_min')}"
        )
        stress = sweep.get("stress_blackout_until_end") or {}
        lines.append(
            f"stress lost_hazards {stress.get('lost_hazards')} delivered {stress.get('hazards_delivered')} "
            f"wasted_blackout_bytes {stress.get('wasted_blackout_bytes')} limit_s {stress.get('pending_limit_s')}"
        )
    source_path = ACCEPTANCE / "source_compare.json"
    process_path = ACCEPTANCE / "acceptance-processes.json"
    determinism_path = ACCEPTANCE / "process_determinism.json"
    if source_path.exists():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        lines.append(
            f"source png hazards {source.get('png_hazards')} false {source.get('png_false')} mean_m {source.get('png_mean_error_m')} "
            f"sim mean_m {source.get('sim_mean_error_m')} mp4 hazards {source.get('mp4_hazards')} mp4 mean_m {source.get('mp4_mean_error_m')} "
            f"png_log_match {source.get('png_matches_sim_packet_log')} mp4_log_match {source.get('mp4_matches_sim_packet_log')}"
        )
    if process_path.exists():
        process = json.loads(process_path.read_text(encoding="utf-8"))
        lines.append(
            f"processes hazards {process.get('hazards_delivered')}/{process.get('hazards_expected')} "
            f"false {process.get('false_hazards_delivered')} rejected {process.get('rejected_inconsistent')} passed {process.get('passed')}"
        )
    if determinism_path.exists():
        determinism = json.loads(determinism_path.read_text(encoding="utf-8"))
        lines.append(
            f"process packet logs identical at loss 0: {determinism.get('packet_logs_identical')} packets {determinism.get('packets')}"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
