"""One pass from a camera frame to Mission Control. Shared integration."""

from __future__ import annotations

import argparse
import json
import math
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from crater.control.board import MissionBoard
from crater.control.server import serve
from crater.link.channel import LossyLink, UdpRadio
from crater.link.packet import TelemetryPacket, TrackFix, decode_packet, encode_packet, json_snapshot
from crater.scout.camera import CameraModel
from crater.scout.detect import detect_hazards
from crater.scout.scene import CraterScene
from crater.scout.track import Tracker


@dataclass
class MissionResult:
    report: dict
    board: MissionBoard


def run_mission(
    seconds: float = 36.0,
    drop_rate: float = 0.12,
    realtime: bool = False,
    board: MissionBoard | None = None,
    seed: int = 7,
) -> MissionResult:
    board = board or MissionBoard()
    board.start()
    camera = CameraModel()
    scene = CraterScene(camera)
    tracker = Tracker()
    link = LossyLink(drop_rate=drop_rate, seed=seed)
    tx, rx = _radio_pair()
    stop = threading.Event()
    listener = threading.Thread(target=_listen, args=(rx, board, stop), daemon=True)
    listener.start()

    dt = 0.1
    steps = max(1, int(seconds / dt))
    detect_s = 0.0
    frames = 0
    error_sum = 0.0
    error_n = 0
    hits = 0
    visible_n = 0
    last_bytes = 0
    last_json = 0

    try:
        for step in range(steps):
            mission_t = step * dt
            rover_x = min(3.2, mission_t * 0.28)
            started = time.perf_counter()
            frame = scene.render(rover_x, 0.0, 0.0)
            detect_started = time.perf_counter()
            detections = detect_hazards(frame)
            localized = []
            for detection in detections:
                point = camera.pixel_to_ground(detection.u, detection.v, rover_x, 0.0, 0.0)
                if point is None:
                    continue
                localized.append((detection, point[0], point[1]))
            tracks = tracker.update(localized)
            detect_s += time.perf_counter() - detect_started
            frames += 1

            visible = scene.visible(rover_x, 0.0, 0.0)
            visible_n += len(visible)
            for item in visible:
                best = None
                for track in tracks:
                    if track.kind != item.hazard.kind:
                        continue
                    dist = math.hypot(track.x - item.hazard.x, track.y - item.hazard.y)
                    best = dist if best is None else min(best, dist)
                if best is not None and best <= 0.75:
                    hits += 1
                    error_sum += best
                    error_n += 1

            packet = TelemetryPacket(
                sequence=step,
                time_ms=int(mission_t * 1000),
                rover_x=rover_x,
                rover_y=0.0,
                yaw_deg=0.0,
                tracks=tuple(
                    TrackFix(track.track_id, track.kind, track.confidence, track.x, track.y) for track in tracks
                ),
            )
            raw = encode_packet(packet)
            as_json = json_snapshot(packet)
            last_bytes = len(raw)
            last_json = len(as_json)
            status = link.consider(mission_t)
            fps = frames / detect_s if detect_s else 0.0
            mean_error = error_sum / error_n if error_n else 0.0
            board.note_perception(fps, mean_error)
            board.note_link(mission_t, status.delivered, status.blackout, last_bytes, last_json)
            if status.delivered:
                tx.send(raw)
            if realtime:
                elapsed = time.perf_counter() - started
                time.sleep(max(0.0, dt - elapsed))
        time.sleep(0.15)
    finally:
        stop.set()
        listener.join(timeout=1.0)
        tx.close()
        rx.close()

    board.finish()
    recall = hits / visible_n if visible_n else 0.0
    mean_error = error_sum / error_n if error_n else 0.0
    fps = frames / detect_s if detect_s else 0.0
    report = {
        "project": "C.R.A.T.E.R.",
        "frames": frames,
        "perception_fps": round(fps, 1),
        "localization_error_m": round(mean_error, 3),
        "recall": round(recall, 3),
        "packet_bytes": last_bytes,
        "json_bytes": last_json,
        "bytes_saved_vs_json": last_json - last_bytes,
        "link_survival": round(link.survival, 3),
        "packets_sent": link.sent,
        "packets_delivered": link.delivered,
        "story_lines": len(board.chronicle.lines),
        "chapters": sorted({line.chapter for line in board.chronicle.lines}),
    }
    return MissionResult(report, board)


def run_benchmark(seconds: float = 8.0) -> dict:
    result = run_mission(seconds=seconds, drop_rate=0.12, realtime=False)
    out = Path("output")
    out.mkdir(exist_ok=True)
    path = out / "preselection_report.json"
    path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")
    result.report["report_path"] = str(path)
    return result.report


def _listen(radio: UdpRadio, board: MissionBoard, stop: threading.Event) -> None:
    while not stop.is_set():
        payload = radio.recv()
        if not payload:
            continue
        try:
            board.apply_packet(decode_packet(payload))
        except ValueError:
            continue


def _radio_pair() -> tuple[UdpRadio, UdpRadio]:
    left = _ephemeral_port()
    right = _ephemeral_port()
    return UdpRadio(left, right), UdpRadio(right, left)


def _ephemeral_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="C.R.A.T.E.R. scout pipeline")
    parser.add_argument("--benchmark", action="store_true", help="write output/preselection_report.json and exit")
    parser.add_argument("--seconds", type=float, default=36.0)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--drop-rate", type=float, default=0.12)
    args = parser.parse_args(argv)

    if args.benchmark:
        report = run_benchmark(args.seconds)
        print(json.dumps({key: report[key] for key in report if key != "chapters"}, indent=2))
        print("chapters:", ", ".join(report["chapters"]))
        return

    board = MissionBoard()
    server = serve(board, args.port)
    worker = threading.Thread(
        target=run_mission,
        kwargs={"seconds": args.seconds, "drop_rate": args.drop_rate, "realtime": True, "board": board},
        daemon=True,
    )
    worker.start()
    url = f"http://127.0.0.1:{args.port}"
    print(f"Mission Control at {url}")
    print("Pipeline is live: perception, lossy UDP telemetry, chronicle.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
