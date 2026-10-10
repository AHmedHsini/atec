"""Greedy world-frame tracker. Owner: Ahmed.

Detections are matched to existing tracks of the same kind inside a gate.
Identity is what the link sends, so Mission Control can age a specific hazard
instead of treating every frame as a new object.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from crater.scout.detect import Detection


@dataclass
class Track:
    track_id: int
    kind: str
    x: float
    y: float
    confidence: float
    missed: int = 0


class Tracker:
    def __init__(self, gate_m: float = 1.25):
        self.gate_m = gate_m
        self._next_id = 1
        self.tracks: list[Track] = []

    def update(self, localized: list[tuple[Detection, float, float]]) -> list[Track]:
        unused = set(range(len(self.tracks)))
        assigned: list[Track] = []
        for detection, x, y in localized:
            best_i = None
            best_d = self.gate_m
            for i in unused:
                track = self.tracks[i]
                if track.kind != detection.kind:
                    continue
                dist = math.hypot(track.x - x, track.y - y)
                if dist < best_d:
                    best_d = dist
                    best_i = i
            if best_i is None:
                track = Track(self._next_id, detection.kind, x, y, detection.confidence)
                self._next_id += 1
            else:
                unused.discard(best_i)
                track = self.tracks[best_i]
                track.x = x
                track.y = y
                track.confidence = detection.confidence
                track.missed = 0
            assigned.append(track)

        for i in unused:
            self.tracks[i].missed += 1
        self.tracks = [track for track in self.tracks if track.missed < 4]
        # Keep newly created tracks that were not yet stored.
        known = {track.track_id for track in self.tracks}
        for track in assigned:
            if track.track_id not in known:
                self.tracks.append(track)
        return sorted(assigned, key=lambda track: track.track_id)
