"""Mission Control's map. It only knows what survived the link.

Older packets cannot replace a newer fix for the same track. While the
radio is down, the scout is coasted for at most 1.2 s and then held.
Hazard sigma grows by 0.05 m for each second the link has been down
since that hazard's timestamp.
"""

from __future__ import annotations

from veil.packet import seq_is_newer

POSE = 2
HAZARD = 1
HEARTBEAT = 3
COAST_S = 1.2
SIGMA_GROWTH_M_PER_S = 0.05
# Link state is based on time since the last accepted heartbeat, in simulated seconds.
LINK_UP_S = 1.0
LINK_DEGRADED_S = 3.0


class GroundMap:
    def __init__(self):
        self.hazards: dict[int, dict] = {}
        self.poses: list[dict] = []
        self.heartbeat: dict | None = None
        self.heartbeat_at: float | None = None

    def link_state(self, sim_t: float) -> str:
        if self.heartbeat_at is None:
            return "DOWN"
        age = sim_t - self.heartbeat_at
        if age < LINK_UP_S:
            return "UP"
        if age < LINK_DEGRADED_S:
            return "DEGRADED"
        return "DOWN"

    def ingest(self, messages: list[dict], sim_t: float) -> list[dict]:
        fresh = []
        for message in messages:
            if message.get("type") == HEARTBEAT:
                self.heartbeat = dict(message)
                self.heartbeat_at = sim_t
                continue
            if message.get("type") not in (HAZARD, POSE):
                continue
            if message["cls"] not in ("rock", "pit", "pose"):
                continue
            message = dict(message)
            message["age_s"] = max(0.0, sim_t - message["t_ms"] / 1000.0)
            if message["type"] == POSE:
                self._ingest_pose(message, sim_t)
            elif message["type"] == HAZARD and message["cls"] in ("rock", "pit"):
                if self._ingest_hazard(message, sim_t):
                    fresh.append(message)
        return fresh

    def _ingest_pose(self, message: dict, sim_t: float) -> None:
        message["received_sim_t"] = sim_t
        if not self.poses:
            self.poses.append(message)
            return
        last = self.poses[-1]
        if message["seq"] == last["seq"] or not seq_is_newer(message["seq"], last["seq"]):
            return
        if message["t_ms"] < last["t_ms"]:
            return
        self.poses.append(message)
        if len(self.poses) > 40:
            self.poses = self.poses[-40:]

    def _ingest_hazard(self, message: dict, sim_t: float) -> bool:
        current = self.hazards.get(message["track_id"])
        if current is None:
            message["received_sim_t"] = sim_t
            self.hazards[message["track_id"]] = message
            return True
        if message["t_ms"] < current["t_ms"]:
            return False
        if message["seq"] == current["seq"]:
            if message["t_ms"] > current["t_ms"]:
                message["received_sim_t"] = current["received_sim_t"]
                self.hazards[message["track_id"]] = message
            return False
        if not seq_is_newer(message["seq"], current["seq"]):
            return False
        message["received_sim_t"] = current["received_sim_t"]
        self.hazards[message["track_id"]] = message
        return False

    def scout(self, sim_t: float, link_open: bool) -> dict | None:
        if not self.poses:
            return None
        last = self.poses[-1]
        age = max(0.0, sim_t - last["t_ms"] / 1000.0)
        predicted = False
        x, y = last["x"], last["y"]
        if not link_open and len(self.poses) >= 2 and age <= COAST_S:
            prev = self.poses[-2]
            dt = (last["t_ms"] - prev["t_ms"]) / 1000.0
            if dt > 0:
                x += (last["x"] - prev["x"]) / dt * age
                y += (last["y"] - prev["y"]) / dt * age
                predicted = True
        elif not link_open:
            predicted = True
        return {"x": x, "y": y, "predicted": predicted, "age_s": round(age, 2)}

    def view(self, sim_t: float, link_open: bool) -> list[dict]:
        rows = []
        for hazard in self.hazards.values():
            age = max(0.0, sim_t - hazard["t_ms"] / 1000.0)
            growth = 0.0 if link_open else SIGMA_GROWTH_M_PER_S * age
            rows.append(
                {
                    "id": hazard["track_id"],
                    "cls": hazard["cls"],
                    "x": hazard["x"],
                    "y": hazard["y"],
                    "sigma": round(hazard["sigma"] + growth, 3),
                    "radius": hazard["radius"],
                    "age_s": round(age, 2),
                    "agreement": hazard["agreement"],
                    "t_ms": hazard["t_ms"],
                }
            )
        return rows
