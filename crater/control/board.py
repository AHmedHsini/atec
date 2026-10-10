"""Live board Mission Control keeps from received packets only. Owner: Hammoud."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from crater.control.story import Chronicle
from crater.control.uncertainty import uncertainty_m
from crater.link.packet import TelemetryPacket


@dataclass
class BoardTrack:
    track_id: int
    kind: str
    x: float
    y: float
    confidence: float
    time_s: float
    missed: int = 0


@dataclass
class MissionBoard:
    chronicle: Chronicle = field(default_factory=Chronicle)
    tracks: dict[int, BoardTrack] = field(default_factory=dict)
    time_s: float = 0.0
    rover_x: float = 0.0
    rover_y: float = 0.0
    yaw_deg: float = 0.0
    rover_time_s: float = 0.0
    blackout: bool = False
    packets_sent: int = 0
    packets_received: int = 0
    last_packet_bytes: int = 0
    json_bytes: int = 0
    fps: float = 0.0
    localization_error_m: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def start(self) -> None:
        with self._lock:
            self.chronicle.start(0.0)

    def note_perception(self, fps: float, localization_error_m: float) -> None:
        with self._lock:
            self.fps = fps
            self.localization_error_m = localization_error_m

    def note_link(self, time_s: float, delivered: bool, blackout: bool, packet_bytes: int, json_bytes: int) -> None:
        with self._lock:
            self.time_s = time_s
            self.blackout = blackout
            self.packets_sent += 1
            self.last_packet_bytes = packet_bytes
            self.json_bytes = json_bytes
            if delivered:
                self.packets_received += 1
            self.chronicle.on_link(time_s, blackout)

    def apply_packet(self, packet: TelemetryPacket) -> None:
        time_s = packet.time_ms / 1000.0
        with self._lock:
            self.time_s = time_s
            self.rover_x = packet.rover_x
            self.rover_y = packet.rover_y
            self.yaw_deg = packet.yaw_deg
            self.rover_time_s = time_s
            seen: set[int] = set()
            for fix in packet.tracks:
                seen.add(fix.track_id)
                self.tracks[fix.track_id] = BoardTrack(
                    track_id=fix.track_id,
                    kind=fix.kind,
                    x=fix.x,
                    y=fix.y,
                    confidence=fix.confidence,
                    time_s=time_s,
                )
            for track_id, track in list(self.tracks.items()):
                if track_id in seen:
                    continue
                track.missed += 1
                if track.missed >= 3:
                    del self.tracks[track_id]
            self.chronicle.on_packet(time_s, list(packet.tracks))

    def finish(self) -> None:
        with self._lock:
            self.chronicle.finish(self.time_s, self.packets_received, self.packets_sent)

    def snapshot(self) -> dict:
        with self._lock:
            tracks = []
            for track in self.tracks.values():
                age = max(0.0, self.time_s - track.time_s)
                tracks.append(
                    {
                        "id": track.track_id,
                        "kind": track.kind,
                        "x": round(track.x, 2),
                        "y": round(track.y, 2),
                        "confidence": round(track.confidence, 2),
                        "age_s": round(age, 2),
                        "uncertainty_m": round(
                            uncertainty_m(track.confidence, track.x, track.y, age),
                            2,
                        ),
                    }
                )
            tracks.sort(key=lambda item: item["id"])
            story = [
                {"time_s": round(line.time_s, 2), "chapter": line.chapter, "text": line.text}
                for line in self.chronicle.lines[-16:]
            ]
            sent = self.packets_sent
            survival = 1.0 if sent == 0 else self.packets_received / sent
            return {
                "time_s": round(self.time_s, 2),
                "blackout": self.blackout,
                "chapter": self.chronicle.chapter,
                "rover": {
                    "x": round(self.rover_x, 2),
                    "y": round(self.rover_y, 2),
                    "yaw": round(self.yaw_deg, 1),
                    "age_s": round(max(0.0, self.time_s - self.rover_time_s), 2),
                },
                "tracks": tracks,
                "story": story,
                "link": {
                    "sent": sent,
                    "received": self.packets_received,
                    "survival": round(survival, 3),
                    "packet_bytes": self.last_packet_bytes,
                    "json_bytes": self.json_bytes,
                },
                "perception": {
                    "fps": round(self.fps, 1),
                    "localization_error_m": round(self.localization_error_m, 3),
                },
            }
