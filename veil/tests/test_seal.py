import time

from veil.link import LossyLink
from veil.packet import encode
from veil.seal import PackageChain, open_envelope, protect


def test_seal_roundtrip_and_tamper():
    packet = encode(1, 4, 1500, 2, 2, 0, 80, 0.35, 5.0, 0.12, 0.4)
    sealed = protect(packet, (7).to_bytes(8, "little"))
    assert open_envelope(sealed) == packet
    broken = bytearray(sealed)
    broken[12] ^= 0xFF
    assert open_envelope(bytes(broken)) is None


def test_chain_links_each_package():
    chain = PackageChain()
    first = protect(b"\x11" * 20, (1).to_bytes(8, "little"))
    second = protect(b"\x22" * 20, (2).to_bytes(8, "little"))
    chain.append(first, 1.0)
    chain.append(second, 2.0)
    assert chain.blocks[1]["prev"] == chain.blocks[0]["hash"]
    assert chain.intact()
    chain.blocks[1]["digest"] = "0" * 64
    assert chain.intact() is False


def test_secure_link_delivers_the_opened_package():
    link = LossyLink(loss=0.0, blackout=(100.0, 100.0), secure=True)
    try:
        packet = encode(1, 3, 1000, 1, 1, 0, 90, 1.25, 4.5, 0.2, 0.3)
        assert link.send(packet, 0.0)
        got = []
        for _ in range(50):
            got = link.take()
            if got:
                break
            time.sleep(0.02)
        assert got and got[0]["x"] == 1.25
        assert link.chain.intact()
        assert link.chain.view()["height"] == 1
        assert link.bytes_sent == 20
    finally:
        link.close()
