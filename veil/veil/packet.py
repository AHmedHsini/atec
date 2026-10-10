"""20-byte hazard packet. A JSON detection is several times larger."""

from __future__ import annotations

import struct

FMT = "<BBHIHBBBhhBBB"
assert struct.calcsize(FMT) == 20

MAGIC = 0xC7
VERSION = 1
TYPE_HAZARD = 1
TYPE_POSE = 2
TYPE_HEARTBEAT = 3
MODE_RUN = 0
MODE_SAFE = 1
CLASS_ROCK = 1
CLASS_PIT = 2
# CRC-8, polynomial 0x07, initial value 0x00, no reflection, xorout 0x00.
CRC8_POLY = 0x07


def crc8(body: bytes) -> int:
    value = 0
    for byte in body:
        value ^= byte
        for _ in range(8):
            if value & 0x80:
                value = ((value << 1) ^ CRC8_POLY) & 0xFF
            else:
                value = (value << 1) & 0xFF
    return value


def seq_is_newer(candidate: int, current: int) -> bool:
    """True when candidate is ahead of current in 16-bit serial space (RFC 1982)."""
    if candidate == current:
        return False
    return ((candidate - current) & 0xFFFF) < 0x8000


def _centimeters(value: float, signed: bool) -> int:
    if not np_finite(value):
        raise ValueError("non-finite measurement")
    scaled = int(round(value * 100))
    if signed and not -32768 <= scaled <= 32767:
        raise ValueError("coordinate does not fit in a signed 16-bit centimeter field")
    if not signed and not 0 <= scaled <= 255:
        raise ValueError("measurement does not fit in an unsigned byte of centimeters")
    return scaled


def np_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


def encode(
    msg_type: int,
    seq: int,
    t_ms: int,
    track_id: int,
    cls: int,
    flags: int,
    agreement: int,
    x_m: float,
    y_m: float,
    sigma_m: float,
    radius_m: float,
) -> bytes:
    if msg_type not in (TYPE_HAZARD, TYPE_POSE, TYPE_HEARTBEAT) or not 0 <= int(seq) <= 0xFFFF:
        raise ValueError("message type or sequence is outside the packet")
    if not 0 <= int(t_ms) <= 0xFFFFFFFF or not 0 <= int(track_id) <= 0xFFFF:
        raise ValueError("time or track id is outside the packet")
    if not 0 <= int(cls) <= 255 or not 0 <= int(flags) <= 255 or not 0 <= int(agreement) <= 100:
        raise ValueError("class, flags, or agreement is outside the packet")
    raw = struct.pack(
        FMT,
        MAGIC,
        (VERSION << 6) | msg_type,
        int(seq),
        int(t_ms),
        int(track_id),
        int(cls),
        int(flags),
        int(agreement),
        _centimeters(x_m, True),
        _centimeters(y_m, True),
        _centimeters(sigma_m, False),
        _centimeters(radius_m, False),
        0,
    )
    body, _ = raw[:-1], raw[-1]
    return body + bytes([crc8(body)])


def decode(data: bytes) -> dict | None:
    if len(data) != 20 or data[0] != MAGIC or crc8(data[:-1]) != data[-1]:
        return None
    (
        _magic,
        combined,
        seq,
        t_ms,
        track_id,
        cls,
        flags,
        agreement,
        x_cm,
        y_cm,
        sigma_cm,
        radius_cm,
        _crc_byte,
    ) = struct.unpack(FMT, data)
    version = combined >> 6
    msg_type = combined & 0x3F
    if version != VERSION:
        return None
    if msg_type not in (TYPE_HAZARD, TYPE_POSE, TYPE_HEARTBEAT):
        return None
    if msg_type == TYPE_HAZARD and cls not in (CLASS_ROCK, CLASS_PIT):
        return None
    names = {CLASS_ROCK: "rock", CLASS_PIT: "pit", 0: "heartbeat" if msg_type == TYPE_HEARTBEAT else "pose"}
    return {
        "version": version,
        "type": msg_type,
        "seq": seq,
        "t_ms": t_ms,
        "track_id": track_id,
        "cls": names.get(cls, "pose"),
        "flags": flags,
        "agreement": agreement,
        "x": x_cm / 100.0,
        "y": y_cm / 100.0,
        "sigma": sigma_cm / 100.0,
        "radius": radius_cm / 100.0,
    }
