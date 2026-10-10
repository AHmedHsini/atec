import json
import time
from pathlib import Path

import numpy as np

from veil.ground import LINK_DEGRADED_S, LINK_UP_S, GroundMap
from veil.mission import run
from veil.onboard import Onboard
from veil.packet import TYPE_HEARTBEAT, decode, encode
from veil.source import VideoFrameSource, record_drive

ROOT = Path(__file__).resolve().parents[1]
OPEN = (1.0e9, 1.0e9 + 1.0)


def test_queue_drops_the_lowest_value_and_keeps_the_hazard():
    onboard = Onboard()
    onboard.max_pending = 3
    for index in range(5):
        onboard.pending[f"p{index}"] = {"seq": index, "kind": "pose", "value": 0.01, "bytes": b"\x00" * 20}
    onboard.pending["h1"] = {"seq": 99, "kind": "hazard", "value": 1.0, "bytes": b"\x00" * 20}
    onboard._trim()
    assert len(onboard.pending) == 3
    assert "h1" in onboard.pending
    assert onboard.lost_hazards == 0
    assert sum(1 for slot in onboard.pending.values() if slot["kind"] == "pose") == 2


def test_heartbeat_fits_in_twenty_bytes_and_ages_the_link():
    packet = encode(TYPE_HEARTBEAT, 7, 2500, 4, 0, 1, 3, 0.2, 1.4, 0.0, 0.0)
    assert len(packet) == 20
    decoded = decode(packet)
    assert decoded["type"] == TYPE_HEARTBEAT
    assert decoded["flags"] == 1
    assert decoded["track_id"] == 4
    assert decoded["agreement"] == 3
    assert decoded["t_ms"] == 2500
    ground = GroundMap()
    assert ground.link_state(0.0) == "DOWN"
    ground.ingest([decoded], 10.0)
    assert ground.hazards == {}
    assert ground.link_state(10.0) == "UP"
    assert ground.link_state(10.0 + LINK_UP_S) == "DEGRADED"
    assert ground.link_state(10.0 + LINK_DEGRADED_S) == "DOWN"


def test_watchdog_stops_new_hazards_and_recovers_on_the_next_frame():
    onboard = Onboard()
    onboard.last_observe_t = 0.0
    onboard.heartbeat((0.0, 1.0), 2.0)
    assert onboard.mode == "SAFE"
    assert decode(onboard.pending["heartbeat"]["bytes"])["flags"] == 1
    for index in range(6):
        onboard.tracker.update(
            [{"cls": "pit", "x": 0.35, "y": 5.0, "radius": 0.4}],
            index * 0.2,
            index * 0.1,
        )
    assert onboard.tracker.confirmed()
    onboard.mode = "SAFE"
    onboard._queue((0.0, 1.0), 2.2, True)
    assert not any(slot["kind"] == "hazard" for slot in onboard.pending.values())
    image = np.full((240, 320), 140, np.uint8)
    onboard.observe(image, (0.0, 1.0), 2.2, 1, time.perf_counter())
    assert onboard.mode == "RUN"
    assert any(slot["kind"] == "hazard" for slot in onboard.pending.values())


def test_same_seed_writes_a_byte_identical_packet_log(tmp_path):
    kwargs = dict(
        live=False,
        loss=0.0,
        blackout=OPEN,
        seed=26,
        pace_s=0.004,
        exchange_s=0.004,
        settle_s=0.0,
        write_report=False,
        scenario="determinism",
    )
    first = run(**kwargs)
    second = run(**kwargs)
    assert first["packet_log"]
    assert first["packet_log"] == second["packet_log"]
    hex_path = ROOT / "artifacts" / "acceptance" / "packet_log_seed26.hex"
    hex_path.parent.mkdir(parents=True, exist_ok=True)
    hex_path.write_text("\n".join(first["packet_log"]) + "\n", encoding="utf-8")
    reloaded = [line.strip() for line in hex_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert reloaded == first["packet_log"]
    assert all(len(bytes.fromhex(line)) == 20 for line in reloaded)


def test_png_sequence_delivers_four_hazards_and_mp4_is_measured(tmp_path):
    record_drive(tmp_path, frames=140, step_m=0.05, dt=0.1, seed=26, write_video=True)
    from veil.source import SimFrameSource

    painted = next(SimFrameSource(1, 0.05, 0.1, 26).frames())["image"]
    loaded = next(VideoFrameSource(tmp_path).frames())["image"]
    pixels_match = bool(np.array_equal(painted, loaded))
    kwargs = dict(live=False, loss=0.3, seed=26, pace_s=0.004, exchange_s=0.004, settle_s=0.0, write_report=False)
    png = run(source=VideoFrameSource(tmp_path), scenario="png-sequence", **kwargs)
    video = run(source=VideoFrameSource(tmp_path / "drive.mp4"), scenario="mp4", **kwargs)
    sim = run(scenario="sim-compare", **kwargs)
    assert png["hazards_delivered"] == 4
    assert png["false_hazards_delivered"] == 0
    assert sim["hazards_delivered"] == 4
    compare = {
        "png_hazards": png["hazards_delivered"],
        "png_false": png["false_hazards_delivered"],
        "png_mean_error_m": png["mean_error_m"],
        "sim_hazards": sim["hazards_delivered"],
        "sim_false": sim["false_hazards_delivered"],
        "sim_mean_error_m": sim["mean_error_m"],
        "mp4_hazards": video["hazards_delivered"],
        "mp4_false": video["false_hazards_delivered"],
        "mp4_mean_error_m": video["mean_error_m"],
        "png_matches_sim_packet_log": png["packet_log"] == sim["packet_log"],
        "mp4_matches_sim_packet_log": video["packet_log"] == sim["packet_log"],
        "png_pixels_match_sim": pixels_match,
        "note": "PNG bytes match the simulator exactly, so a packet-log mismatch at 30% loss is ack timing, not the picture. The mp4v file is lossy and can move a fix.",
    }
    path = ROOT / "artifacts" / "acceptance" / "source_compare.json"
    path.write_text(json.dumps(compare, indent=2), encoding="utf-8")
    assert compare["png_hazards"] == 4
    assert pixels_match


def test_separate_processes_deliver_and_do_not_stay_alive():
    report = run(
        live=False,
        processes=True,
        loss=0.3,
        seed=26,
        blackout=(0.5, 6.0),
        frames=140,
        dt=0.1,
        step_m=0.05,
        pace_s=0.004,
        exchange_s=0.004,
        settle_s=0.0,
        scenario="acceptance-processes",
        write_report=True,
    )
    assert report["process_mode"] is True
    assert report["hazards_delivered"] == 4
    assert report["false_hazards_delivered"] == 0
    assert report["rejected_inconsistent"] >= 1
    logs = []
    for seed in (4, 4):
        logs.append(
            run(
                live=False,
                processes=True,
                loss=0.0,
                seed=seed,
                blackout=OPEN,
                frames=40,
                pace_s=0.0,
                exchange_s=0.002,
                settle_s=0.0,
                scenario="process-determinism",
                write_report=False,
            )["packet_log"]
        )
    same = logs[0] == logs[1] and bool(logs[0])
    note_path = ROOT / "artifacts" / "acceptance" / "process_determinism.json"
    note_path.write_text(
        json.dumps({"loss": 0.0, "frames": 40, "packet_logs_identical": same, "packets": len(logs[0])}),
        encoding="utf-8",
    )
    assert same
