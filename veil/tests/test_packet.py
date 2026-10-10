import pytest

from veil.packet import crc8, decode, encode


def test_packet_is_twenty_bytes_and_round_trips_signed_centimeters():
    packet = encode(1, 4, 1500, 2, 2, 0, 80, -1.25, 5.0, 0.12, 0.4)
    assert len(packet) == 20
    assert packet[1] >> 6 == 1
    assert packet[1] & 0x3F == 1
    decoded = decode(packet)
    assert decoded["x"] == -1.25
    assert decoded["y"] == 5.0
    assert decoded["cls"] == "pit"
    assert decoded["sigma"] == 0.12
    assert decoded["version"] == 1


def test_corruption_unknown_type_and_unknown_class_are_rejected():
    packet = encode(1, 4, 1500, 2, 2, 0, 80, 0.35, 5.0, 0.12, 0.4)
    broken = bytearray(packet)
    broken[12] ^= 0xFF
    assert decode(bytes(broken)) is None
    assert decode(packet[:19]) is None
    unknown = bytearray(packet)
    unknown[1] = 9
    unknown[-1] = 0
    from veil.packet import crc8 as packet_crc

    unknown[-1] = packet_crc(bytes(unknown[:-1]))
    assert decode(bytes(unknown)) is None
    bad_class = bytearray(packet)
    bad_class[10] = 9
    bad_class[-1] = packet_crc(bytes(bad_class[:-1]))
    assert decode(bytes(bad_class)) is None


def test_crc8_rejects_a_swap_that_xor_would_accept():
    packet = bytearray(encode(1, 4, 1500, 2, 2, 0, 80, 0.35, 5.0, 0.12, 0.4))
    left, right = 4, 8
    assert packet[left] != packet[right]
    packet[left], packet[right] = packet[right], packet[left]
    xor = 0
    for byte in packet[:-1]:
        xor ^= byte
    original = bytearray(encode(1, 4, 1500, 2, 2, 0, 80, 0.35, 5.0, 0.12, 0.4))
    original_xor = 0
    for byte in original[:-1]:
        original_xor ^= byte
    assert xor == original_xor
    assert crc8(bytes(packet[:-1])) != packet[-1]
    assert decode(bytes(packet)) is None


def test_unknown_version_is_rejected():
    packet = bytearray(encode(1, 4, 1500, 2, 2, 0, 80, 0.35, 5.0, 0.12, 0.4))
    packet[1] = (2 << 6) | 1
    packet[-1] = crc8(bytes(packet[:-1]))
    assert decode(bytes(packet)) is None


def test_values_outside_the_field_width_are_refused():
    with pytest.raises(ValueError):
        encode(1, 1, 0, 1, 1, 0, 80, 400.0, 0.0, 0.1, 0.1)
    with pytest.raises(ValueError):
        encode(1, 1, 0, 1, 1, 0, 80, 0.0, 0.0, 3.0, 0.1)
    with pytest.raises(ValueError):
        encode(1, 70000, 0, 1, 1, 0, 80, 0.0, 0.0, 0.1, 0.1)


def test_sequence_and_timestamp_limits_round_trip():
    packet = encode(2, 65535, 0xFFFFFFFF, 65535, 0, 0, 0, -327.68, 327.67, 0, 2.55)
    decoded = decode(packet)
    assert decoded["seq"] == 65535
    assert decoded["t_ms"] == 0xFFFFFFFF
    assert decoded["type"] == 2
