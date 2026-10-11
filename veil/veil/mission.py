"""One command from camera frames to Mission Control."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
import numpy as np

from veil.ground import GroundMap
from veil.link import BUDGET_BYTES, LossyLink
from veil.onboard import Onboard
from veil.camera import view_polygon
from veil.model import load as load_model
from veil.reports import stamp, write_json
from veil.sim import HAZARDS, render_color, render_pose, scout_pose

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "index.html"
OUTPUTS = ROOT / "outputs"


def run(
    live: bool = False,
    loss: float = 0.3,
    frames: int = 140,
    step_m: float = 0.05,
    dt: float = 0.1,
    port: int = 8877,
    hold: float = 0,
    blackout: tuple[float, float] = (0.5, 6.0),
    realtime: bool = False,
    scenario: str = "acceptance-short",
    seed: int = 26,
    pace_s: float = 0.02,
    exchange_s: float = 0.01,
    settle_s: float = 0.03,
    write_report: bool = True,
    pending_limit_s: float = 30.0,
    source=None,
    processes: bool = False,
) -> dict:
    if processes:
        from veil.procs import run_processes

        source_path = None if source is None else str(getattr(source, "path", source))
        return run_processes(
            live=live,
            loss=loss,
            frames=frames,
            step_m=step_m,
            dt=dt,
            port=port,
            hold=hold,
            blackout=blackout,
            realtime=realtime,
            scenario=scenario,
            seed=seed,
            pace_s=pace_s,
            exchange_s=exchange_s,
            settle_s=settle_s,
            write_report=write_report,
            pending_limit_s=pending_limit_s,
            source_path=source_path,
        )
    rng = np.random.default_rng(seed)
    link = LossyLink(loss=loss, blackout=blackout, seed=seed, cycle=realtime)
    onboard = Onboard()
    onboard.pending_limit_s = pending_limit_s
    ground = GroundMap()
    events: list[dict] = []
    holder = {"state": _empty_state(), "jpeg": b"", "manual": {"lander": [0.0, 0.0], "scout": None}}
    server = _serve(holder, port) if live else None
    OUTPUTS.mkdir(exist_ok=True)
    if live:
        print(f"Mission Control  http://127.0.0.1:{port}", flush=True)
    clock = time.perf_counter()
    link_was_open = True
    recoveries: list[float] = []
    reopen_t: float | None = None
    queued_ids: set[int] = set()
    first_hazard_recovery = None
    all_queued_recovery = None

    try:
        frame = 0
        played = 0
        last_sim_t = 0.0
        frame_iter = None if source is None else iter(source.frames())
        while True:
            placed = holder["manual"]["scout"]
            if placed:
                sim_t = frame * dt
                scout = (float(placed[0]), float(placed[1]))
                image = render_pose(scout, rng)
                advance = False
            elif frame_iter is None:
                if frame >= frames:
                    break
                sim_t = frame * dt
                scout = scout_pose(frame, step_m)
                image = render_pose(scout, rng)
                advance = True
            else:
                try:
                    sample = next(frame_iter)
                except StopIteration:
                    break
                frame = int(sample["frame"])
                sim_t = float(sample["t_s"])
                scout = (float(sample["x_m"]), float(sample["y_m"]))
                image = sample["image"]
                advance = True
            last_sim_t = sim_t
            if frame == 20:
                cv2.imwrite(str(OUTPUTS / "frame_20.png"), image)
            if live:
                _set_camera(holder, scout, image)
            started = time.perf_counter()
            onboard.observe(image, scout, sim_t, frame, started)
            _exchange(link, onboard, ground, sim_t, events, exchange_s)
            open_now = link.is_open(sim_t)
            if link_was_open and not open_now:
                pass
            if not link_was_open and open_now:
                reopen_t = sim_t
                queued_ids = {slot["track_id"] for slot in onboard.pending.values() if slot.get("kind") == "hazard"}
                if not queued_ids:
                    first_hazard_recovery = 0.0
                    all_queued_recovery = 0.0
            if reopen_t is not None and queued_ids:
                have = set(ground.hazards)
                if first_hazard_recovery is None and have & queued_ids:
                    first_hazard_recovery = round(sim_t - reopen_t, 3)
                if queued_ids <= have:
                    all_queued_recovery = round(sim_t - reopen_t, 3)
                    recoveries.append(all_queued_recovery)
                    queued_ids = set()
            _note_link(events, sim_t, open_now, link_was_open)
            link_was_open = open_now
            holder["pose_warning"] = getattr(source, "pose_warning", "") if source is not None else ""
            _publish(holder, ground, link, onboard, sim_t, events, blackout, driving=placed is None)
            if not advance:
                time.sleep(0.12)
                continue
            _pace(realtime, clock, frame, dt, pace_s)
            played += 1
            if frame_iter is None:
                frame += 1
        if source is not None:
            frames = played
        settle_t = last_sim_t
        parked = scout
        for step in range(40):
            tick = settle_t + (step + 1) * dt
            onboard.heartbeat(parked, tick)
            open_now = link.is_open(tick)
            if not link_was_open and open_now and reopen_t is None:
                reopen_t = tick
                queued_ids = {slot["track_id"] for slot in onboard.pending.values() if slot.get("kind") == "hazard"}
            _exchange(link, onboard, ground, tick, events, exchange_s)
            if reopen_t is not None and queued_ids:
                have = set(ground.hazards)
                if first_hazard_recovery is None and have & queued_ids:
                    first_hazard_recovery = round(tick - reopen_t, 3)
                if queued_ids <= have:
                    all_queued_recovery = round(tick - reopen_t, 3)
                    recoveries.append(all_queued_recovery)
                    queued_ids = set()
            link_was_open = open_now
            if realtime:
                time.sleep(0.02)
            else:
                time.sleep(settle_s)
        report = _report(
            ground,
            link,
            onboard,
            frames,
            loss,
            blackout,
            scenario="live" if live else scenario,
            seed=seed,
            dt=dt,
            step_m=step_m,
            recoveries=recoveries,
            first_hazard_recovery=first_hazard_recovery,
            all_queued_recovery=all_queued_recovery,
            pose_warning=getattr(source, "pose_warning", "") if source is not None else "",
            pose_mode=getattr(source, "pose_mode", "simulator") if source is not None else "simulator",
        )
        _publish(holder, ground, link, onboard, settle_t, events, blackout, driving=False, report=report)
        if write_report:
            folder = "live" if live else "acceptance"
            name = "report.json" if live else f"{scenario}.json"
            write_json(ROOT / "artifacts" / folder / name, report)
        if live and hold != 0:
            _hold(hold, onboard, link, ground, holder, events, blackout, parked, settle_t)
        return report
    finally:
        link.close()
        if source is not None:
            source.close()
        if server is not None:
            server.shutdown()


def _set_camera(holder: dict, pose: tuple[float, float], image=None) -> None:
    del image
    camera_id = f"{pose[0]:.2f}_{pose[1]:.2f}"
    if holder.get("camera_id") == camera_id and holder.get("jpeg"):
        return
    ok, encoded = cv2.imencode(".jpg", render_color(pose, 0.0), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    if not ok:
        return
    holder["jpeg"] = encoded.tobytes()
    holder["camera_id"] = camera_id
    holder["camera_pose"] = {"x": pose[0], "y": pose[1], "yaw": 0.0}
    holder["view"] = view_polygon(pose, 0.0)


def _pace(realtime: bool, clock: float, frame: int, dt: float, pace_s: float) -> None:
    if realtime:
        time.sleep(max(0.0, clock + (frame + 1) * dt - time.perf_counter()))
    elif pace_s > 0:
        time.sleep(pace_s)


def _exchange(link: LossyLink, onboard: Onboard, ground: GroundMap, sim_t: float, events: list[dict], exchange_s: float = 0.01) -> None:
    onboard.ack(link.poll_acks())
    if link.is_open(sim_t):
        for packet in onboard.packets(BUDGET_BYTES):
            link.send(packet, sim_t)
    if exchange_s > 0:
        time.sleep(exchange_s)
    for message in ground.ingest(link.take(), sim_t):
        kind = "Pit" if message["cls"] == "pit" else "Rock"
        _event(events, sim_t, f"{kind} received and placed on the map.")


def _event(events: list[dict], sim_t: float, text: str) -> None:
    if events and events[-1]["text"] == text:
        return
    events.append({"t": round(sim_t, 1), "text": text})
    del events[:-8]


def _note_link(events: list[dict], sim_t: float, open_now: bool, was_open: bool) -> None:
    if open_now and not was_open:
        _event(events, sim_t, "Radio back. Saved hazards are being sent.")
    elif was_open and not open_now:
        _event(events, sim_t, "Radio lost. The scout keeps looking and saves what it finds.")


def _phase(sim_t: float, blackout: tuple[float, float], driving: bool, link_open: bool) -> str:
    if not driving:
        return "On station" if link_open else "Blackout"
    if sim_t < blackout[0]:
        return "Survey"
    if sim_t < blackout[1]:
        return "Blackout"
    return "Burst"


def _hold(hold, onboard, link, ground, holder, events, blackout, parked, sim_t) -> None:
    deadline = None if hold < 0 else time.perf_counter() + hold
    try:
        while deadline is None or time.perf_counter() < deadline:
            sim_t += 0.2
            was_open = link.is_open(sim_t - 0.2)
            placed = holder["manual"]["scout"]
            pose = (float(placed[0]), float(placed[1])) if placed else parked
            image = render_pose(pose, np.random.default_rng(0))
            _set_camera(holder, pose, image)
            if placed:
                onboard.observe(image, pose, sim_t, 0, time.perf_counter())
            onboard.heartbeat(pose, sim_t)
            _exchange(link, onboard, ground, sim_t, events)
            _note_link(events, sim_t, link.is_open(sim_t), was_open)
            _publish(holder, ground, link, onboard, sim_t, events, blackout, driving=False, report=holder["state"].get("report"))
            time.sleep(0.2)
    except KeyboardInterrupt:
        return


def _confirm_latencies(ground: GroundMap, onboard: Onboard) -> list[float]:
    by_id = {track.track_id: track for track in onboard.tracker.tracks + onboard.tracker.archived}
    latencies = []
    for hazard in ground.hazards.values():
        track = by_id.get(hazard["track_id"])
        confirmed_at = None if track is None else track.confirmed_at
        received = hazard.get("received_sim_t")
        if confirmed_at is None or received is None:
            continue
        latencies.append(float(received) - float(confirmed_at))
    return latencies


def _retransmission_mean(onboard: Onboard) -> float:
    if not onboard.hazard_seqs:
        return 0.0
    extra = [max(0, onboard.tx_attempts.get(seq, 0) - 1) for seq in onboard.hazard_seqs]
    return float(sum(extra) / len(extra))


def _report(ground, link, onboard: Onboard, frames: int, loss: float, blackout, scenario: str, seed: int, dt: float, step_m: float, recoveries: list[float] | None = None, first_hazard_recovery=None, all_queued_recovery=None, pose_warning: str = "", pose_mode: str = "simulator") -> dict:
    matched = []
    false_count = 0
    used = set()
    for hazard in ground.hazards.values():
        best = None
        best_distance = 0.6
        for index, truth in enumerate(HAZARDS):
            if index in used or truth["cls"] != hazard["cls"]:
                continue
            distance = float(np.hypot(truth["x"] - hazard["x"], truth["y"] - hazard["y"]))
            if distance < best_distance:
                best = index
                best_distance = distance
        if best is None:
            false_count += 1
        else:
            used.add(best)
            matched.append(best_distance)
    tracker = onboard.tracker
    rejected = tracker.rejected()
    expired = tracker.expired()
    mean_error = float(np.mean(matched)) if matched else None
    fps = round(frames / onboard.detect_s, 1) if onboard.detect_s else 0
    model = load_model()
    missed = len(HAZARDS) - len(matched)
    latencies = _confirm_latencies(ground, onboard)
    retransmissions = _retransmission_mean(onboard)
    passed = (
        len(matched) == len(HAZARDS)
        and false_count == 0
        and len(rejected) >= 1
        and mean_error is not None
        and mean_error < 0.25
        and link.bytes_sent < onboard.naive_bytes
        and link.bytes_sent < onboard.float32_bytes
        and link.bytes_sent == link.sent * 20
    )
    report = {
        "fps": fps,
        "fps_method": "frames divided by time spent inside detect(), excluding radio sleep",
        "detections": onboard.detection_count,
        "hazards_delivered": len(matched),
        "hazards_expected": len(HAZARDS),
        "missed_hazards": missed,
        "false_hazards_delivered": false_count,
        "rejected_inconsistent": len(rejected),
        "expired_tracks": len(expired),
        "withheld_inconsistent_tracks": len(rejected),
        "mean_error_m": None if mean_error is None else round(mean_error, 3),
        "max_error_m": None if not matched else round(max(matched), 3),
        "error_method": "each delivered hazard matched once to the nearest same-class truth within 0.6 m",
        "packet_bytes": 20,
        "bytes_sent": link.bytes_sent,
        "bytes_sent_meaning": "application payload only, 20 bytes per send attempt, excluding UDP and IP headers",
        "naive_json_bytes": onboard.naive_bytes,
        "float32_bytes": onboard.float32_bytes,
        "float32_confirmed_bytes": onboard.float32_confirmed_bytes,
        "float32_layout": "five little-endian float32 values per detection: class code, x, y, sigma (0 at detection), timestamp",
        "hazard_bytes": onboard.hazard_bytes,
        "pose_bytes": onboard.pose_bytes,
        "heartbeat_bytes": onboard.heartbeat_bytes,
        "retransmission_bytes": onboard.retransmission_bytes,
        "wasted_blackout_bytes": link.wasted_blackout_bytes,
        "bytes_per_delivered_hazard": round(onboard.hazard_bytes / len(matched), 3) if matched else None,
        "superseded_drops": onboard.superseded_drops,
        "ack_bytes": link.acks * 2,
        "udp_ipv4_header_bytes_estimate": link.sent * 28,
        "udp_header_note": "28 bytes is the UDP plus IPv4 header per datagram. It is not measured on the wire.",
        "pose_warning": pose_warning,
        "pose_mode": pose_mode,
        "competent_binary_bytes": onboard.float32_confirmed_bytes + onboard.pose_bytes,
        "competent_binary_note": "Confirmed hazards as 20-byte float stand-in plus pose bytes already sent. No CRC and no ack in this baseline.",
        "lost_hazards": onboard.lost_hazards,
        "confirm_to_receipt_s": [round(value, 3) for value in latencies],
        "retransmissions_per_hazard": round(retransmissions, 3),
        "blackout_recovery_s": recoveries or [],
        "first_hazard_recovery_s": first_hazard_recovery,
        "all_queued_recovery_s": all_queued_recovery,
        "packet_log": onboard.packet_log,
        "packets_sent": link.sent,
        "packets_delivered": link.delivered,
        "packets_dropped": link.dropped,
        "loss": loss,
        "model_samples": model.samples,
        "model_val_accuracy": round(model.val_accuracy, 3),
        "model_confusion": model.confusion,
        "model_note": "Held-out accuracy is on synthetic patches, not lunar field images. The network classifies brightness proposals. It is not a full-image detector.",
        "blackout_s": list(blackout),
        "budget_bytes": BUDGET_BYTES,
        "passed": passed,
    }
    return stamp(report, scenario=scenario, seed=seed, frames=frames, dt=dt, step_m=step_m)


def _empty_state() -> dict:
    return {
        "sim_t": 0,
        "phase": "Survey",
        "link": "open",
        "scout": None,
        "hazards": [],
        "events": [],
        "stats": {},
        "report": None,
    }


def _publish(holder, ground, link, onboard: Onboard, sim_t, events, blackout, driving: bool, report=None) -> None:
    tracker = onboard.tracker
    link_open = link.is_open(sim_t)
    previous = holder["state"].get("report")
    manual = holder.get("manual") or {"lander": [0.0, 0.0], "scout": None}
    scout_view = ground.scout(sim_t, link_open)
    if manual.get("scout"):
        scout_view = {
            "x": float(manual["scout"][0]),
            "y": float(manual["scout"][1]),
            "predicted": False,
            "age_s": 0,
            "manual": True,
        }
    state = {
        "sim_t": round(sim_t, 2),
        "phase": _phase(sim_t, blackout, driving, link_open),
        "link": "open" if link_open else "blackout",
        "link_state": ground.link_state(sim_t),
        "mode": onboard.mode,
        "pose_warning": holder.get("pose_warning") or "",
        "packet_tail": onboard.packet_log[-40:],
        "manual": bool(manual.get("scout")),
        "camera_id": holder.get("camera_id") or "",
        "camera_pose": holder.get("camera_pose"),
        "view": holder.get("view") or [],
        "lander": {"x": float(manual["lander"][0]), "y": float(manual["lander"][1])},
        "scout": scout_view,
        "hazards": ground.view(sim_t, link_open),
        "events": list(events),
        "stats": {
            "fps": round(onboard.frames / onboard.detect_s, 1) if onboard.detect_s else 0,
            "detections": onboard.detection_count,
            "confirmed": len(tracker.confirmed()),
            "withheld": sum(1 for track in tracker.tracks if not track.confirmed and len(track.points) >= 5),
            "queued": len(onboard.pending),
            "bytes_sent": link.bytes_sent,
            "naive_bytes": onboard.naive_bytes,
            "delivered": link.delivered,
            "dropped": link.dropped,
            "budget": BUDGET_BYTES,
            "hazard_bytes": onboard.hazard_bytes,
            "pose_bytes": onboard.pose_bytes,
            "heartbeat_bytes": onboard.heartbeat_bytes,
            "retransmission_bytes": onboard.retransmission_bytes,
            "superseded_drops": onboard.superseded_drops,
            "ack_bytes": link.acks * 2,
            "bytes_per_delivered_hazard": round(onboard.hazard_bytes / max(1, len(ground.hazards)), 3),
            "model_accuracy": round(load_model().val_accuracy, 3),
            "model_note": "Synthetic held-out accuracy. Not the PolarDB score.",
            "model_samples": load_model().samples,
        },
        "report": report if report is not None else previous,
    }
    holder["state"] = state


def _serve(holder: dict, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            if not self.path.startswith("/api/move"):
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            manual = holder["manual"]
            if payload.get("resume"):
                manual["scout"] = None
            if "lander" in payload:
                manual["lander"] = [float(payload["lander"][0]), float(payload["lander"][1])]
            if "scout" in payload:
                manual["scout"] = [float(payload["scout"][0]), float(payload["scout"][1])]
            self._send(b"{}", "application/json")

        def do_GET(self):
            if self.path.startswith("/api/state"):
                payload = json.dumps(holder["state"]).encode()
                self._send(payload, "application/json")
                return
            if self.path.startswith("/camera.jpg"):
                self._send(holder.get("jpeg") or b"", "image/jpeg")
                return
            if self.path in ("/", "/index.html"):
                self._send(WEB.read_bytes(), "text/html; charset=utf-8")
                return
            self.send_response(404)
            self.end_headers()

        def _send(self, payload: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, fmt: str, *args) -> None:
            return

    class BoundServer(ThreadingHTTPServer):
        allow_reuse_address = False

    server = BoundServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
