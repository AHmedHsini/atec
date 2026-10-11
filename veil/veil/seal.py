"""Encrypt each 20-byte package and record it on a hash chain.

The scout and the ground station share one key, loaded before launch.
The wire datagram is the encrypted package plus a tag. The ground station
opens it, then appends a block whose hash covers the previous block.
One sender and one receiver: there is no miner and no outside network.
"""

from __future__ import annotations

import hashlib
import hmac

MAGIC = b"\xe1"
NONCE_LEN = 8
TAG_LEN = 16
PACKET_LEN = 20
ENVELOPE_LEN = 1 + NONCE_LEN + PACKET_LEN + TAG_LEN

_MASTER = hashlib.sha256(b"VEIL-CRATER-PRESHARED-2026").digest()
_ENC = hmac.new(_MASTER, b"enc", hashlib.sha256).digest()
_MAC = hmac.new(_MASTER, b"mac", hashlib.sha256).digest()


def _keystream(nonce: bytes, size: int) -> bytes:
    out = bytearray()
    block = 0
    while len(out) < size:
        out.extend(hmac.new(_ENC, nonce + block.to_bytes(4, "little"), hashlib.sha256).digest())
        block += 1
    return bytes(out[:size])


def protect(packet: bytes, nonce: bytes) -> bytes:
    """Return the sealed datagram for one 20-byte package."""
    if len(packet) != PACKET_LEN or len(nonce) != NONCE_LEN:
        raise ValueError("package and nonce must be fixed width")
    body = bytes(left ^ right for left, right in zip(packet, _keystream(nonce, PACKET_LEN)))
    tag = hmac.new(_MAC, nonce + body, hashlib.sha256).digest()[:TAG_LEN]
    return MAGIC + nonce + body + tag


def open_envelope(blob: bytes) -> bytes | None:
    """Return the 20-byte package, or None when the tag does not match."""
    if len(blob) != ENVELOPE_LEN or blob[:1] != MAGIC:
        return None
    nonce = blob[1 : 1 + NONCE_LEN]
    body = blob[1 + NONCE_LEN : 1 + NONCE_LEN + PACKET_LEN]
    tag = blob[-TAG_LEN:]
    expect = hmac.new(_MAC, nonce + body, hashlib.sha256).digest()[:TAG_LEN]
    if not hmac.compare_digest(tag, expect):
        return None
    return bytes(left ^ right for left, right in zip(body, _keystream(nonce, PACKET_LEN)))


class PackageChain:
    """Hash-linked record of packages the ground station accepted."""

    def __init__(self) -> None:
        self.blocks: list[dict] = []

    def append(self, envelope: bytes, sim_t: float) -> dict:
        prev = self.blocks[-1]["hash"] if self.blocks else "0" * 64
        digest = hashlib.sha256(envelope).hexdigest()
        index = len(self.blocks)
        stamp = f"{sim_t:.3f}"
        raw = f"{index}|{prev}|{digest}|{stamp}".encode()
        block = {
            "index": index,
            "prev": prev,
            "digest": digest,
            "sim_t": stamp,
            "hash": hashlib.sha256(raw).hexdigest(),
        }
        self.blocks.append(block)
        return block

    def intact(self) -> bool:
        prev = "0" * 64
        for block in self.blocks:
            raw = f"{block['index']}|{prev}|{block['digest']}|{block['sim_t']}".encode()
            if block["prev"] != prev or hashlib.sha256(raw).hexdigest() != block["hash"]:
                return False
            prev = block["hash"]
        return True

    def view(self) -> dict:
        last = self.blocks[-1]["hash"][:16] if self.blocks else ""
        return {
            "enabled": True,
            "height": len(self.blocks),
            "last": last,
            "verified": self.intact(),
            "envelope_bytes": ENVELOPE_LEN,
            "note": "HMAC-SHA256 seal. Each delivered package is the next block.",
        }
