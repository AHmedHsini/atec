"""Hold-step. A detection becomes a hazard only if later views agree.

Outcomes are explicit:

- tentative: not enough travel yet
- confirmed: a 0.7 m window agrees within 0.32 m
- rejected: that window does not agree (the points slide)
- expired: the track went quiet before it could be judged

A rejected track is kept in the archive when it later goes quiet, so a
missed detection is not counted as a geometric rejection and a geometric
rejection is not forgotten.
"""

from __future__ import annotations

import numpy as np

BASELINE_M = 0.7
MAX_SPREAD_M = 0.32
ASSOCIATE_M = 0.55
MISS_LIMIT = 6

TENTATIVE = "tentative"
CONFIRMED = "confirmed"
REJECTED = "rejected"
EXPIRED = "expired"


class Track:
    def __init__(self, track_id: int, detection: dict, travel: float):
        self.track_id = track_id
        self.cls = detection["cls"]
        self.points: list[tuple[float, float]] = [(detection["x"], detection["y"])]
        self.travels: list[float] = [travel]
        self.radii: list[float] = [detection["radius"]]
        self.misses = 0
        self.outcome = TENTATIVE
        self.reason = ""
        self.confirmed = False
        self.confirmed_at = None
        self.x = detection["x"]
        self.y = detection["y"]
        self.sigma = 1.0
        self.radius = detection["radius"]
        self.agreement = 0

    def add(self, detection: dict, travel: float, sim_t: float | None = None) -> None:
        self.points.append((detection["x"], detection["y"]))
        self.travels.append(travel)
        self.radii.append(detection["radius"])
        self.misses = 0
        self.cls = detection["cls"]
        recent = np.array(self.points[-8:], dtype=float)
        center = np.median(recent, axis=0)
        self.x = float(center[0])
        self.y = float(center[1])
        was_confirmed = self.confirmed
        self._evaluate()
        if self.confirmed and not was_confirmed:
            self.confirmed_at = sim_t

    def _evaluate(self) -> None:
        end = self.travels[-1]
        start_index = 0
        for index, travel in enumerate(self.travels):
            if end - travel >= BASELINE_M:
                start_index = index
        if end - self.travels[start_index] < BASELINE_M:
            return
        window = np.array(self.points[start_index:], dtype=float)
        if len(window) < 3:
            return
        center = np.median(window, axis=0)
        spread = float(np.max(np.linalg.norm(window - center, axis=1)))
        if spread > MAX_SPREAD_M:
            self.outcome = REJECTED
            self.reason = "inconsistent_ground"
            self.confirmed = False
            self.sigma = spread
            self.agreement = 0
            return
        self.outcome = CONFIRMED
        self.reason = "geometric_agreement"
        self.confirmed = True
        self.x = float(center[0])
        self.y = float(center[1])
        self.sigma = max(0.05, spread)
        self.radius = float(np.median(self.radii[start_index:]))
        self.agreement = int(round(100 * (1 - spread / MAX_SPREAD_M)))


class Tracker:
    def __init__(self):
        self.tracks: list[Track] = []
        self.archived: list[Track] = []
        self._next_id = 1

    def update(self, detections: list[dict], travel: float, sim_t: float | None = None) -> None:
        unused = set(range(len(self.tracks)))
        for detection in detections:
            best = None
            best_distance = ASSOCIATE_M
            for index in unused:
                track = self.tracks[index]
                if track.cls != detection["cls"]:
                    continue
                distance = float(np.hypot(track.x - detection["x"], track.y - detection["y"]))
                if distance < best_distance:
                    best = index
                    best_distance = distance
            if best is None:
                self.tracks.append(Track(self._next_id, detection, travel))
                self._next_id += 1
            else:
                unused.remove(best)
                self.tracks[best].add(detection, travel, sim_t)
        for index in unused:
            self.tracks[index].misses += 1

    def drop_stale(self) -> None:
        kept = []
        for track in self.tracks:
            if track.outcome == CONFIRMED or track.misses <= MISS_LIMIT:
                kept.append(track)
                continue
            if track.outcome != REJECTED:
                track.outcome = EXPIRED
                track.reason = "missed_detections"
            self.archived.append(track)
        self.tracks = kept

    def confirmed(self) -> list[Track]:
        return [track for track in self.tracks if track.outcome == CONFIRMED]

    def rejected(self) -> list[Track]:
        return [track for track in self.tracks + self.archived if track.outcome == REJECTED]

    def expired(self) -> list[Track]:
        return [track for track in self.archived if track.outcome == EXPIRED]
