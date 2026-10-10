"""Fixed-width binary telemetry. Owner: Wissal.

Layout, big-endian, no variable-length strings:

    0  u8   magic 0xC7
    1  u8   version 1
    2  u16  sequence
    4  u32  mission time, milliseconds
    8  i16  rover x, centimeters
   10  i16  rover y, centimeters
   12  i16  yaw, centidegrees
   14  u8   detection count (0..8)
   15  u8   reserved
   16  8*N  records

Each record is 8 bytes: track id u16, class u8, confidence u8 (0..100),
x cm i16, y cm i16. Six hazards fit in 64 bytes. A JSON snapshot of the
same fields is several times larger; the benchmark records that ratio.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass

MAGIC = 0xC7
VERSION = 1
MAX_TRACKS = 8
HEADER = struct.Struct(">BBHIhhhBB")
RECORD = struct.Struct(">HBbhh")

CLASS_TO_ID = {"rock": 1, "crater": 2, "ice": 3}
ID_TO_CLASS = {value: key for key, value in CLASS_TO_ID.items()}


@dataclass(frozen=True)
class TrackFix:
    track_id: int
    kind: str
    confidence: float
    x: float
    y: float


@dataclass(frozen=True)
class TelemetryPacket:
    sequence: int
    time_ms: int
    rover_x: float
    rover_y: float
    yaw_deg: float
    tracks: tuple[TrackFix, ...]

    @property
    def nbytes(self) -> int:
        return HEADER.size + RECORD.size * len(self.tracks)


def encode_packet(packet: TelemetryPacket) -> bytes:
    tracks = packet.tracks[:MAX_TRACKS]
    if len(tracks) > MAX_TRACKS:
        tracks = tracks[:MAX_TRACKS]
    header = HEADER.pack(
        MAGIC,
        VERSION,
        packet.sequence & 0xFFFF,
        packet.time_ms & 0xFFFFFFFF,
        _cm(packet.rover_x),
        _cm(packet.rover_y),
        _centi_deg(packet.yaw_deg),
        len(tracks),
        0,
    )
    body = bytearray()
    for track in tracks:
        if track.kind not in CLASS_TO_ID:
            raise ValueError(f"unknown class {track.kind}")
        body.extend(
            RECORD.pack(
                track.track_id & 0xFFFF,
                CLASS_TO_ID[track.kind],
                _conf(track.confidence),
                _cm(track.x),
                _cm(track.y),
            )
        )
    return header + bytes(body)


def decode_packet(payload: bytes) -> TelemetryPacket:
    if len(payload) < HEADER.size:
        raise ValueError("packet shorter than header")
    magic, version, sequence, time_ms, rover_x, rover_y, yaw, count, _reserved = HEADER.unpack_from(payload)
    if magic != MAGIC or version != VERSION:
        raise ValueError("bad magic or version")
    expected = HEADER.size + RECORD.size * count
    if len(payload) < expected:
        raise ValueError("packet truncated")
    tracks = []
    offset = HEADER.size
    for _ in range(count):
        track_id, class_id, confidence, x_cm, y_cm = RECORD.unpack_from(payload, offset)
        offset += RECORD.size
        if class_id not in ID_TO_CLASS:
            raise ValueError(f"unknown class id {class_id}")
        tracks.append(
            TrackFix(
                track_id=track_id,
                kind=ID_TO_CLASS[class_id],
                confidence=confidence / 100.0,
                x=x_cm / 100.0,
                y=y_cm / 100.0,
            )
        )
    return TelemetryPacket(
        sequence=sequence,
        time_ms=time_ms,
        rover_x=rover_x / 100.0,
        rover_y=rover_y / 100.0,
        yaw_deg=yaw / 100.0,
        tracks=tuple(tracks),
    )


def json_snapshot(packet: TelemetryPacket) -> bytes:
    """Uncompressed JSON of the same fields, used only as a size baseline."""
    payload = {
        "sequence": packet.sequence,
        "time_ms": packet.time_ms,
        "rover": {"x": round(packet.rover_x, 2), "y": round(packet.rover_y, 2), "yaw": round(packet.yaw_deg, 2)},
        "tracks": [
            {
                "id": track.track_id,
                "kind": track.kind,
                "confidence": round(track.confidence, 2),
                "x": round(track.x, 2),
                "y": round(track.y, 2),
            }
            for track in packet.tracks
        ],
    }
    return json.dumps(payload, separators=(",", ":")).encode("utf-8")


def _cm(meters: float) -> int:
    return max(-32768, min(32767, int(round(meters * 100.0))))


def _centi_deg(degrees: float) -> int:
    return max(-32768, min(32767, int(round(degrees * 100.0))))


def _conf(confidence: float) -> int:
    return max(0, min(100, int(round(confidence * 100.0))))
