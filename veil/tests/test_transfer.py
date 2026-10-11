from veil.packet import decode, encode
from veil.transfer import GENESIS, SEAL_LEN, Ledger


def _packet(seq: int) -> bytes:
    return encode(1, seq, 1000, 1, 1, 0, 80, -1.0, 4.2, 0.2, 0.28)


def test_seal_hides_the_package_and_the_chain_links():
    ledger = Ledger(b"m" * 32)
    first = ledger.seal(_packet(1))
    second = ledger.seal(_packet(2))
    assert len(first) == SEAL_LEN
    assert _packet(1) not in first
    assert first[15:47] == GENESIS
    assert second[15:47] == first[-32:]


def test_ground_opens_the_package_and_rejects_a_tampered_block():
    sender = Ledger(b"k" * 32)
    ground = Ledger(b"k" * 32)
    frame = bytearray(sender.seal(_packet(4)))
    opened = ground.open(bytes(frame))
    assert decode(opened)["seq"] == 4
    assert ground.gaps == 0
    frame[-1] ^= 0xFF
    assert ground.open(bytes(frame)) is None
    assert ground.rejected == 1


def test_a_missing_package_is_a_gap_and_the_next_one_still_opens():
    sender = Ledger(b"k" * 32)
    ground = Ledger(b"k" * 32)
    first = sender.seal(_packet(1))
    sender.seal(_packet(2))
    third = sender.seal(_packet(3))
    assert decode(ground.open(first))["seq"] == 1
    assert decode(ground.open(third))["seq"] == 3
    assert ground.gaps == 1
    assert ground.recv_height == 2


def test_the_same_frame_is_not_accepted_twice():
    sender = Ledger(b"k" * 32)
    ground = Ledger(b"k" * 32)
    frame = sender.seal(_packet(1))
    assert ground.open(frame) is not None
    assert ground.open(frame) is None
    assert ground.rejected == 1
