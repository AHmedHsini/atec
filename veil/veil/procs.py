"""Onboard, radio, and ground as separate processes over UDP.

The parent is the launcher. It starts the three children, and for a live
mission it also serves the dashboard from the files those children write.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import random
import socket
import time
from pathlib import Path

import cv2
import numpy as np

from veil.ground import GroundMap
from veil.link import BUDGET_BYTES, OutboundLink
from veil.mission import _exchange
from veil.onboard import Onboard
from veil.packet import decode
from veil.sim import HAZARDS, render_color, scout_pose
from veil.source import SimFrameSource, VideoFrameSource


def run_processes(
    live: bool = False,
    loss: float = 0.3,
    frames: int = 140,
    step_m: float = 0.05,
    dt: float = 0.1,
    port: int = 8877,
    hold: float = 0,
    blackout: tuple[float, float] = (0.5, 6.0),
    realtime: bool = False,
    scenario: str = "acceptance-processes",
    seed: int = 26,
    pace_s: float = 0.004,
    exchange_s: float = 0.004,
    settle_s: float = 0.0,
    write_report: bool = True,
    pending_limit_s: float = 30.0,
    source_path: str | None = None,
    work_dir: str | None = None,
) -> dict:
    ctx = mp.get_context("spawn")
    stop = ctx.Event()
    ports: mp.Queue = ctx.Queue()
    work = Path(work_dir) if work_dir else Path(__file__).resolve().parents[1] / "artifacts" / "processes" / scenario
    work.mkdir(parents=True, exist_ok=True)
    onboard_path = work / "onboard.json"
    ground_path = work / "ground.json"
    children = [
        ctx.Process(target=_ground_main, args=(ports, stop, str(ground_path), str(work / "state.json"))),
    ]
    children[0].start()
    ground_port = ports.get(timeout=10)
    radio = ctx.Process(target=_radio_main, args=(ground_port, loss, seed, ports, stop))
    radio.start()
    children.append(radio)
    radio_port = ports.get(timeout=10)
    onboard = ctx.Process(
        target=_onboard_main,
        args=(
            radio_port,
            frames,
            step_m,
            dt,
            seed,
            blackout,
            pace_s,
            exchange_s,
            settle_s,
            pending_limit_s,
            str(onboard_path),
            stop,
            live,
            source_path,
            str(work),
        ),
    )
    onboard.start()
    children.append(onboard)
    server = None
    try:
        if live:
            from veil.mission import _empty_state, _serve

            holder = {"state": _empty_state(), "jpeg": b"", "manual": {"lander": [0.0, 0.0], "scout": None}}
            server = _serve(holder, port)
            print(f"Mission Control  http://127.0.0.1:{port}", flush=True)
            _wait_live(hold, stop, onboard, holder, work)
        else:
            onboard.join(timeout=180)
        stop.set()
        for child in children:
            child.join(timeout=5)
        report = _assemble(onboard_path, ground_path, scenario, seed, frames, dt, step_m, loss, blackout, write_report)
        report["process_mode"] = True
        report["child_pids"] = [child.pid for child in children]
        return report
    finally:
        stop.set()
        if server is not None:
            server.shutdown()
        for child in children:
            child.join(timeout=2)
            if child.is_alive():
                child.terminate()
                child.join(timeout=2)
        report_alive = [child.pid for child in children if child.is_alive()]
        if report_alive:
            raise RuntimeError(f"child processes still alive: {report_alive}")


def _wait_live(hold: float, stop, onboard, holder: dict, work: Path) -> None:
    deadline = None if hold < 0 else time.perf_counter() + hold
    state_path = work / "state.json"
    jpeg_path = work / "camera.jpg"
    manual_path = work / "manual.json"
    try:
        while onboard.is_alive() and (deadline is None or time.perf_counter() < deadline):
            manual_path.write_text(json.dumps(holder["manual"]), encoding="utf-8")
            if state_path.exists():
                try:
                    incoming = json.loads(state_path.read_text(encoding="utf-8"))
                    holder["state"].update(incoming)
                except json.JSONDecodeError:
                    pass
            if jpeg_path.exists():
                holder["jpeg"] = jpeg_path.read_bytes()
            time.sleep(0.1)
    except KeyboardInterrupt:
        return
    stop.set()
    onboard.join(timeout=5)


def _assemble(onboard_path, ground_path, scenario, seed, frames, dt, step_m, loss, blackout, write_report) -> dict:
    onboard_info = json.loads(Path(onboard_path).read_text(encoding="utf-8"))
    ground_info = json.loads(Path(ground_path).read_text(encoding="utf-8"))
    matched = []
    false_count = 0
    used = set()
    for hazard in ground_info["hazards"]:
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
    mean_error = float(np.mean(matched)) if matched else None
    report = {
        "scenario": scenario,
        "seed": seed,
        "frames": frames,
        "dt": dt,
        "step_m": step_m,
        "loss": loss,
        "blackout_s": list(blackout),
        "process_mode": True,
        "seeded_determinism_note": onboard_info.get("determinism_note", ""),
        "packet_log": onboard_info["packet_log"],
        "hazards_delivered": len(matched),
        "hazards_expected": len(HAZARDS),
        "false_hazards_delivered": false_count,
        "rejected_inconsistent": onboard_info["rejected_inconsistent"],
        "mean_error_m": None if mean_error is None else round(mean_error, 3),
        "bytes_sent": onboard_info["bytes_sent"],
        "wasted_blackout_bytes": onboard_info["wasted_blackout_bytes"],
        "hazard_bytes": onboard_info["hazard_bytes"],
        "pose_bytes": onboard_info["pose_bytes"],
        "heartbeat_bytes": onboard_info["heartbeat_bytes"],
        "retransmission_bytes": onboard_info["retransmission_bytes"],
        "lost_hazards": onboard_info["lost_hazards"],
        "passed": len(matched) == len(HAZARDS) and false_count == 0 and onboard_info["rejected_inconsistent"] >= 1,
    }
    if write_report:
        from veil.reports import write_json

        root = Path(__file__).resolve().parents[1]
        write_json(root / "artifacts" / "acceptance" / f"{scenario}.json", report)
        hex_path = root / "artifacts" / "acceptance" / f"{scenario}.hex"
        hex_path.write_text("\n".join(onboard_info["packet_log"]) + "\n", encoding="utf-8")
    return report


def _onboard_main(radio_port, frames, step_m, dt, seed, blackout, pace_s, exchange_s, settle_s, pending_limit_s, onboard_path, stop, live, source_path, work) -> None:
    link = OutboundLink(("127.0.0.1", radio_port), blackout)
    onboard = Onboard()
    onboard.pending_limit_s = pending_limit_s
    ground = GroundMap()
    events: list[dict] = []
    source = VideoFrameSource(source_path) if source_path else SimFrameSource(frames, step_m, dt, seed)
    work_path = Path(work)
    manual_path = work_path / "manual.json"
    last = scout_pose(0, step_m)
    sim_t = 0.0
    try:
        for sample in source.frames():
            if stop.is_set():
                break
            placed = _manual_scout(manual_path)
            if placed is not None:
                scout = placed
                image = sample["image"]
            else:
                scout = (float(sample["x_m"]), float(sample["y_m"]))
                image = sample["image"]
            last = scout
            sim_t = float(sample["t_s"])
            if live:
                ok, encoded = cv2.imencode(".jpg", render_color(scout, 0.0), [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if ok:
                    (work_path / "camera.jpg").write_bytes(encoded.tobytes())
            onboard.observe(image, scout, sim_t, int(sample["frame"]), time.perf_counter())
            _exchange(link, onboard, ground, sim_t, events, exchange_s)
            _drain_acks(link, onboard)
            if live:
                time.sleep(max(pace_s, 0.05))
            elif pace_s > 0:
                time.sleep(pace_s)
        for step in range(40):
            if stop.is_set():
                break
            tick = sim_t + (step + 1) * dt
            onboard.heartbeat(last, tick)
            _exchange(link, onboard, ground, tick, events, exchange_s)
            _drain_acks(link, onboard)
            if settle_s > 0:
                time.sleep(settle_s)
        if live:
            while not stop.is_set():
                onboard.heartbeat(last, sim_t)
                _exchange(link, onboard, ground, sim_t, events, exchange_s)
                time.sleep(0.2)
    finally:
        source.close()
        payload = {
            "packet_log": onboard.packet_log,
            "rejected_inconsistent": len(onboard.tracker.rejected()),
            "lost_hazards": onboard.lost_hazards,
            "bytes_sent": link.bytes_sent,
            "wasted_blackout_bytes": link.wasted_blackout_bytes,
            "hazard_bytes": onboard.hazard_bytes,
            "pose_bytes": onboard.pose_bytes,
            "heartbeat_bytes": onboard.heartbeat_bytes,
            "retransmission_bytes": onboard.retransmission_bytes,
            "determinism_note": "The onboard packet log is the bytes offered to the radio. Ack timing can change how many retries appear.",
        }
        Path(onboard_path).write_text(json.dumps(payload), encoding="utf-8")
        link.close()


def _radio_main(ground_port: int, loss: float, seed: int, ports, stop) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.05)
    ports.put(sock.getsockname()[1])
    rng = random.Random(seed)
    ground_addr = ("127.0.0.1", ground_port)
    onboard_addr = None
    try:
        while not stop.is_set():
            try:
                data, addr = sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) == 20:
                onboard_addr = addr
                if rng.random() < loss:
                    continue
                sock.sendto(data, ground_addr)
            elif len(data) == 2 and onboard_addr is not None:
                if rng.random() < loss:
                    continue
                sock.sendto(data, onboard_addr)
    finally:
        sock.close()


def _ground_main(ports, stop, ground_path: str, state_path: str) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    sock.settimeout(0.05)
    ports.put(sock.getsockname()[1])
    ground = GroundMap()
    sim_t = 0.0
    try:
        while not stop.is_set():
            try:
                data, addr = sock.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            if len(data) != 20:
                continue
            decoded = decode(data)
            if decoded is None:
                continue
            sim_t = max(sim_t, decoded["t_ms"] / 1000.0)
            ground.ingest([decoded], sim_t)
            sock.sendto(decoded["seq"].to_bytes(2, "little"), addr)
            _write_state(ground, sim_t, state_path)
    finally:
        rows = []
        for hazard in ground.hazards.values():
            rows.append({"track_id": hazard["track_id"], "cls": hazard["cls"], "x": hazard["x"], "y": hazard["y"]})
        Path(ground_path).write_text(json.dumps({"hazards": rows}), encoding="utf-8")
        sock.close()


def _write_state(ground: GroundMap, sim_t: float, state_path: str) -> None:
    state = {
        "sim_t": round(sim_t, 2),
        "link_state": ground.link_state(sim_t),
        "link": "open" if ground.link_state(sim_t) != "DOWN" else "blackout",
        "scout": ground.scout(sim_t, ground.link_state(sim_t) == "UP"),
        "hazards": ground.view(sim_t, ground.link_state(sim_t) == "UP"),
        "phase": "Survey",
    }
    Path(state_path).write_text(json.dumps(state), encoding="utf-8")


def _drain_acks(link, onboard, wait_s: float = 0.02) -> None:
    """Give a zero-loss ack time to arrive before the next simulated step."""
    deadline = time.perf_counter() + wait_s
    while time.perf_counter() < deadline:
        acked = link.poll_acks()
        if acked:
            onboard.ack(acked)
        if not onboard.pending:
            return
        time.sleep(0.001)


def _manual_scout(path: Path):
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    scout = payload.get("scout")
    if not scout:
        return None
    return float(scout[0]), float(scout[1])
