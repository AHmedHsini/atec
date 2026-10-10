"""Emulated crater radio: dropouts, blackouts, and a real UDP socket. Owner: Wissal.

Packets that the channel rejects are never written to the socket. Packets that
survive are sent with `socket.sendto` to loopback. Mission Control reads them
with `recvfrom`. There is no cloud hop and no retransmission yet — a lost
packet becomes age on the dashboard, which is the right behavior for a
high-latency link that cannot wait for an ACK.
"""

from __future__ import annotations

import random
import socket
from dataclasses import dataclass


@dataclass
class LinkStatus:
    delivered: bool
    blackout: bool
    drop_rate: float


class LossyLink:
    def __init__(
        self,
        drop_rate: float = 0.12,
        first_blackout_s: float = 9.0,
        blackout_period_s: float = 16.0,
        blackout_length_s: float = 2.4,
        seed: int = 7,
    ):
        if not 0.0 <= drop_rate < 1.0:
            raise ValueError("drop_rate must be in [0, 1)")
        self.drop_rate = drop_rate
        self.first_blackout_s = first_blackout_s
        self.blackout_period_s = blackout_period_s
        self.blackout_length_s = blackout_length_s
        self._rng = random.Random(seed)
        self.sent = 0
        self.delivered = 0

    def in_blackout(self, time_s: float) -> bool:
        if time_s < self.first_blackout_s:
            return False
        phase = (time_s - self.first_blackout_s) % self.blackout_period_s
        return phase < self.blackout_length_s

    def consider(self, time_s: float) -> LinkStatus:
        self.sent += 1
        blackout = self.in_blackout(time_s)
        if blackout or self._rng.random() < self.drop_rate:
            return LinkStatus(False, blackout, self.drop_rate)
        self.delivered += 1
        return LinkStatus(True, False, self.drop_rate)

    @property
    def survival(self) -> float:
        if self.sent == 0:
            return 1.0
        return self.delivered / self.sent


class UdpRadio:
    """Loopback UDP endpoints. One process can hold both sides."""

    def __init__(self, bind_port: int, peer_port: int, host: str = "127.0.0.1"):
        self.host = host
        self.peer = (host, peer_port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((host, bind_port))
        self.sock.settimeout(0.05)

    def send(self, payload: bytes) -> None:
        self.sock.sendto(payload, self.peer)

    def recv(self) -> bytes | None:
        try:
            data, _addr = self.sock.recvfrom(2048)
        except socket.timeout:
            return None
        return data

    def close(self) -> None:
        self.sock.close()
