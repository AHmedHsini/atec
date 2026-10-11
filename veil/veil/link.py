"""Lossy crater link over UDP. 80-byte bursts, then silence."""

from __future__ import annotations

import random
import socket
import threading

from veil.packet import decode

BUDGET_BYTES = 80


class LossyLink:
    def __init__(
        self,
        loss: float = 0.3,
        blackout: tuple[float, float] = (0.5, 6.0),
        seed: int = 26,
        cycle: bool = False,
    ):
        self.loss = loss
        self.blackout = blackout
        self.cycle = cycle
        self._rng = random.Random(seed)
        self._rng_lock = threading.Lock()
        self.scout = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.ground = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.scout.bind(("127.0.0.1", 0))
        self.ground.bind(("127.0.0.1", 0))
        self.scout_addr = self.scout.getsockname()
        self.ground_addr = self.ground.getsockname()
        self.scout.setblocking(False)
        self.ground.settimeout(0.05)
        self.sent = 0
        self.delivered = 0
        self.dropped = 0
        self.bytes_sent = 0
        self.wasted_blackout_bytes = 0
        self.acks = 0
        self.corrupt_rejected = 0
        self._inbox: list[dict] = []
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._listen, daemon=True)
        self._thread.start()

    def is_open(self, sim_t: float) -> bool:
        start, end = self.blackout
        if start <= sim_t < end:
            return False
        if self.cycle and sim_t >= end + 6:
            phase = (sim_t - (end + 6)) % 14
            if phase < 4:
                return False
        return True

    def _chance(self) -> float:
        with self._rng_lock:
            return self._rng.random()

    def send(self, packet: bytes, sim_t: float) -> bool:
        self.sent += 1
        self.bytes_sent += len(packet)
        if not self.is_open(sim_t):
            self.dropped += 1
            self.wasted_blackout_bytes += len(packet)
            return False
        if self._chance() < self.loss:
            self.dropped += 1
            return False
        self.scout.sendto(packet, self.ground_addr)
        return True

    def _listen(self) -> None:
        while not self._stop.is_set():
            try:
                data, _addr = self.ground.recvfrom(256)
            except socket.timeout:
                continue
            except OSError:
                break
            decoded = decode(data)
            if decoded is None:
                self.corrupt_rejected += 1
                continue
            self.delivered += 1
            with self._lock:
                self._inbox.append(decoded)
            if self._stop.is_set():
                break
            if self._chance() >= self.loss:
                ack = decoded["seq"].to_bytes(2, "little")
                try:
                    self.ground.sendto(ack, self.scout_addr)
                    self.acks += 1
                except OSError:
                    break

    def poll_acks(self) -> set[int]:
        acked: set[int] = set()
        while True:
            try:
                data, _addr = self.scout.recvfrom(16)
            except BlockingIOError:
                break
            except OSError:
                break
            if len(data) >= 2:
                acked.add(int.from_bytes(data[:2], "little"))
        return acked

    def take(self) -> list[dict]:
        with self._lock:
            batch = self._inbox
            self._inbox = []
        return batch

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=0.3)
        try:
            self.ground.close()
        except OSError:
            pass
        try:
            self.scout.close()
        except OSError:
            pass


class OutboundLink:
    """Scout-side socket. Loss and forwarding belong to the radio process."""

    def __init__(self, radio_addr: tuple[str, int], blackout: tuple[float, float]):
        self.radio_addr = radio_addr
        self.blackout = blackout
        self.cycle = False
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.setblocking(False)
        self.sent = 0
        self.delivered = 0
        self.dropped = 0
        self.bytes_sent = 0
        self.wasted_blackout_bytes = 0
        self.acks = 0

    def is_open(self, sim_t: float) -> bool:
        start, end = self.blackout
        return not (start <= sim_t < end)

    def send(self, packet: bytes, sim_t: float) -> bool:
        self.sent += 1
        self.bytes_sent += len(packet)
        if not self.is_open(sim_t):
            self.dropped += 1
            self.wasted_blackout_bytes += len(packet)
            return False
        self.sock.sendto(packet, self.radio_addr)
        return True

    def poll_acks(self) -> set[int]:
        acked: set[int] = set()
        while True:
            try:
                data, _addr = self.sock.recvfrom(16)
            except BlockingIOError:
                break
            except OSError:
                break
            if len(data) >= 2:
                acked.add(int.from_bytes(data[:2], "little"))
        return acked

    def take(self) -> list:
        return []

    def close(self) -> None:
        self.sock.close()
