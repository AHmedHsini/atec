"""Sealed transfer of the 20-byte package.

The hazard record stays 20 bytes. On the way to the radio it is encrypted
with SHA-256 in counter mode and authenticated with HMAC-SHA256. Each sealed
package carries the hash of the previous one, so the ground can see whether
the packages it received still form one chain.

This is a local chain between the rover and the ground station. It is not a
public network and it does not publish positions off the machine.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import struct
from pathlib import Path

MAGIC = b"\xC7\x51"
VERSION = 1
GENESIS = b"\x00" * 32
MAC_LEN = 16
HEAD_LEN = 67
SEAL_LEN = HEAD_LEN + MAC_LEN + 32


def load_or_create_key(path: Path) -> bytes:
    """32-byte mission key shared by the rover and the ground. Not written into packets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        key = path.read_bytes()
        if len(key) == 32:
            return key
    key = secrets.token_bytes(32)
    path.write_bytes(key)
    return key


def _keys(master: bytes) -> tuple[bytes, bytes]:
    enc = hmac.new(master, b"veil-seal-enc", hashlib.sha256).digest()
    mac = hmac.new(master, b"veil-seal-mac", hashlib.sha256).digest()
    return enc, mac


def _keystream(enc_key: bytes, nonce: bytes, size: int) -> bytes:
    out = b""
    counter = 0
    while len(out) < size:
        block = hashlib.sha256(enc_key + nonce + counter.to_bytes(4, "little")).digest()
        out += block
        counter += 1
    return out[:size]


class Ledger:
    """Sender chain and receiver chain for one shared mission key."""

    def __init__(self, master: bytes):
        if len(master) != 32:
            raise ValueError("mission key must be 32 bytes")
        self.enc_key, self.mac_key = _keys(master)
        self.send_prev = GENESIS
        self.recv_prev = GENESIS
        self.send_height = 0
        self.recv_height = 0
        self.sealed_bytes = 0
        self.gaps = 0
        self.rejected = 0
        self.seen: set[int] = set()

    def seal(self, packet: bytes) -> bytes:
        if len(packet) != 20:
            raise ValueError("only the 20-byte package is sealed")
        index = self.send_height
        nonce = index.to_bytes(8, "little")
        cipher = bytes(a ^ b for a, b in zip(packet, _keystream(self.enc_key, nonce, 20)))
        head = MAGIC + bytes([VERSION]) + struct.pack("<IQ", index, index) + self.send_prev + cipher
        mac = hmac.new(self.mac_key, head, hashlib.sha256).digest()[:MAC_LEN]
        block = hashlib.sha256(head + mac).digest()
        self.send_prev = block
        self.send_height += 1
        self.sealed_bytes += SEAL_LEN
        return head + mac + block

    def open(self, frame: bytes) -> bytes | None:
        if len(frame) != SEAL_LEN or frame[:2] != MAGIC or frame[2] != VERSION:
            self.rejected += 1
            return None
        head, mac, block = frame[:HEAD_LEN], frame[HEAD_LEN:HEAD_LEN + MAC_LEN], frame[HEAD_LEN + MAC_LEN:]
        expect_mac = hmac.new(self.mac_key, head, hashlib.sha256).digest()[:MAC_LEN]
        expect_block = hashlib.sha256(head + mac).digest()
        if not hmac.compare_digest(expect_mac, mac) or not hmac.compare_digest(expect_block, block):
            self.rejected += 1
            return None
        index = struct.unpack_from("<I", frame, 3)[0]
        if index in self.seen:
            self.rejected += 1
            return None
        prev = frame[15:47]
        if prev != self.recv_prev:
            self.gaps += 1
        cipher = frame[47:HEAD_LEN]
        nonce = frame[7:15]
        plain = bytes(a ^ b for a, b in zip(cipher, _keystream(self.enc_key, nonce, 20)))
        self.recv_prev = block
        self.recv_height += 1
        self.seen.add(index)
        return plain

    def snapshot(self) -> dict:
        last = self.recv_prev if self.recv_height else self.send_prev
        return {
            "cipher": "SHA-256-CTR",
            "mac": "HMAC-SHA256",
            "chain": "SHA-256",
            "sealed": self.send_height,
            "verified": self.recv_height,
            "gaps": self.gaps,
            "rejected": self.rejected,
            "sealed_bytes": self.sealed_bytes,
            "frame_bytes": SEAL_LEN,
            "last_hash": last.hex(),
            "note": "Local hash chain of sealed packages. Not a public blockchain.",
        }
