"""Onboard computer. It never sees the ground map, only its own camera and queue."""

from __future__ import annotations

import json
import struct
import time

import numpy as np

from veil.packet import CLASS_PIT, CLASS_ROCK, MODE_RUN, MODE_SAFE, TYPE_HAZARD, TYPE_HEARTBEAT, TYPE_POSE, encode
from veil.perceive import detect
from veil.tracks import Tracker


class Onboard:
    def __init__(self):
        import os

        if os.environ.get("VEIL_MODEL") == "polar":
            from veil.perceive import use_polar

            use_polar()
        self.tracker = Tracker()
        self.pending: dict[str, dict] = {}
        self.finished: set[str] = set()
        self.seq = 1
        self.last_pose_t = -1.0
        self.detect_s = 0.0
        self.detection_count = 0
        self.frames = 0
        self.naive_bytes = 0
        self.float32_bytes = 0
        self.tx_attempts: dict[int, int] = {}
        self.hazard_seqs: set[int] = set()
        self.queued_at: dict[int, float] = {}
        self.lost_hazards = 0
        self.hazard_bytes = 0
        self.pose_bytes = 0
        self.heartbeat_bytes = 0
        self.retransmission_bytes = 0
        self.last_heartbeat_t = -1.0
        self.float32_confirmed_bytes = 0
        self.packet_log: list[str] = []
        self.mode = "RUN"
        self.last_detections: list[dict] = []
        self.last_observe_t = 0.0
        self.confirmed_sent = 0
        self.pending_limit_s = 30.0
        self.max_attempts = 12
        self.max_pending = 8
        self.superseded_drops = 0
        self.pose_interval_s = 0.5

    def observe(self, image, scout: tuple[float, float], sim_t: float, frame: int, started: float, yaw: float = 0.0) -> None:
        self.frames += 1
        detections = detect(image, scout, yaw)
        self.last_detections = detections
        self.detect_s += time.perf_counter() - started
        self.detection_count += len(detections)
        for detection in detections:
            self.naive_bytes += len(
                json.dumps(
                    {
                        "cls": detection["cls"],
                        "x": round(detection["x"], 3),
                        "y": round(detection["y"], 3),
                        "frame": frame,
                    }
                )
            )
            cls_code = {"rock": 1.0, "pit": 2.0}.get(detection["cls"], 0.0)
            self.float32_bytes += len(
                struct.pack("<5f", cls_code, float(detection["x"]), float(detection["y"]), 0.0, float(sim_t))
            )
        self.tracker.update(detections, scout[1], sim_t)
        self.tracker.drop_stale()
        self.last_observe_t = sim_t
        self.mode = "RUN"
        self._expire(sim_t)
        self._queue(scout, sim_t, hazards=True)

    def heartbeat(self, scout: tuple[float, float], sim_t: float) -> None:
        self._watch(sim_t)
        self._expire(sim_t)
        self._queue(scout, sim_t, hazards=False)

    def note_sent(self, packet: bytes, kind: str, attempt: int) -> None:
        self.packet_log.append(packet.hex())
        if attempt <= 1:
            if kind == "hazard":
                self.hazard_bytes += len(packet)
            elif kind == "heartbeat":
                self.heartbeat_bytes += len(packet)
            else:
                self.pose_bytes += len(packet)
        else:
            self.retransmission_bytes += len(packet)

    def _watch(self, sim_t: float) -> None:
        if sim_t - self.last_observe_t > 1.0:
            self.mode = "SAFE"

    def _expire(self, sim_t: float) -> None:
        for key, slot in list(self.pending.items()):
            if key == "pose" or key == "heartbeat":
                continue
            attempts = self.tx_attempts.get(slot["seq"], 0)
            age = sim_t - self.queued_at.get(slot["seq"], sim_t)
            if attempts >= self.max_attempts or age >= self.pending_limit_s:
                self.pending.pop(key, None)
                self.finished.add(key)
                self.lost_hazards += 1

    def ack(self, seqs: set[int]) -> None:
        for acked in seqs:
            for key, slot in list(self.pending.items()):
                if slot["seq"] != acked:
                    continue
                self.pending.pop(key, None)
                if key != "pose":
                    self.finished.add(key)

    def packets(self, budget: int) -> list[bytes]:
        ready = sorted(self.pending.values(), key=lambda slot: slot["value"], reverse=True)
        chosen = []
        left = budget
        for slot in ready:
            if left < 20:
                break
            kind = slot.get("kind", "pose")
            if kind in ("pose", "heartbeat") and self.tx_attempts.get(slot["seq"], 0) >= 1:
                continue
            chosen.append(slot["bytes"])
            attempt = self.tx_attempts.get(slot["seq"], 0) + 1
            self.tx_attempts[slot["seq"]] = attempt
            self.note_sent(slot["bytes"], slot.get("kind", "pose"), attempt)
            left -= 20
        return chosen

    def _queue(self, scout: tuple[float, float], sim_t: float, hazards: bool) -> None:
        self.pose_interval_s = 1.5 if len(self.pending) >= 5 else 0.5
        if sim_t - self.last_pose_t >= self.pose_interval_s:
            previous = self.pending.get("pose")
            if previous is not None and self.tx_attempts.get(previous["seq"], 0) >= 1:
                self.superseded_drops += 1
            seq = self.seq
            self.seq += 1
            mode_flag = MODE_SAFE if self.mode == "SAFE" else MODE_RUN
            depth = min(100, len(self.pending))
            confirmed = len(self.tracker.confirmed())
            self.pending["pose"] = {
                "seq": seq,
                "kind": "pose",
                "value": 0.01,
                "bytes": encode(TYPE_POSE, seq, int(sim_t * 1000), confirmed, 0, mode_flag, depth, scout[0], scout[1], 0.05, 0),
            }
            self.last_pose_t = sim_t
        if sim_t - self.last_heartbeat_t >= 1.0:
            previous = self.pending.get("heartbeat")
            if previous is not None and self.tx_attempts.get(previous["seq"], 0) >= 1:
                self.superseded_drops += 1
            seq = self.seq
            self.seq += 1
            depth = min(100, len(self.pending))
            confirmed = len(self.tracker.confirmed())
            mode_flag = MODE_SAFE if self.mode == "SAFE" else MODE_RUN
            self.pending["heartbeat"] = {
                "seq": seq,
                "kind": "heartbeat",
                "value": 0.001,
                "bytes": encode(
                    TYPE_HEARTBEAT,
                    seq,
                    int(sim_t * 1000),
                    confirmed,
                    0,
                    mode_flag,
                    depth,
                    scout[0],
                    scout[1],
                    0.0,
                    0.0,
                ),
            }
            self.last_heartbeat_t = sim_t
        if hazards and self.mode != "SAFE":
            self._queue_hazards(scout, sim_t)
        self._trim()

    def _queue_hazards(self, scout: tuple[float, float], sim_t: float) -> None:
        for track in self.tracker.confirmed():
            if track.cls not in ("pit", "rock"):
                continue
            key = f"h{track.track_id}"
            if key in self.finished:
                continue
            slot = self.pending.get(key)
            seq = self.seq if slot is None else slot["seq"]
            if slot is None:
                self.seq += 1
                self.hazard_seqs.add(seq)
                self.queued_at[seq] = sim_t
                self.float32_confirmed_bytes += 20
                self.confirmed_sent += 1
            distance = max(0.4, float(np.hypot(track.x - scout[0], track.y - scout[1])))
            severity = 2.0 if track.cls == "pit" else 1.0
            cls = CLASS_PIT if track.cls == "pit" else CLASS_ROCK
            try:
                packet = encode(
                    TYPE_HAZARD,
                    seq,
                    int(sim_t * 1000),
                    track.track_id,
                    cls,
                    0,
                    track.agreement,
                    track.x,
                    track.y,
                    min(2.55, max(0.0, float(track.sigma))),
                    min(2.55, max(0.0, float(track.radius))),
                )
            except ValueError:
                if slot is None:
                    self.seq -= 1
                    self.hazard_seqs.discard(seq)
                    self.queued_at.pop(seq, None)
                    self.float32_confirmed_bytes -= 20
                    self.confirmed_sent -= 1
                self.finished.add(key)
                continue
            self.pending[key] = {
                "seq": seq,
                "kind": "hazard",
                "track_id": track.track_id,
                "value": severity * (0.15 + track.sigma) / distance,
                "bytes": packet,
            }

    def _trim(self) -> None:
        """Drop the lowest-value slots until the queue fits. Poses and heartbeats lose to hazards."""
        while len(self.pending) > self.max_pending:
            key = min(self.pending, key=lambda name: (self.pending[name]["value"], name))
            slot = self.pending.pop(key)
            if slot.get("kind") == "hazard":
                self.lost_hazards += 1
