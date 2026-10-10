import unittest

from crater.link.packet import TelemetryPacket, TrackFix, decode_packet, encode_packet, json_snapshot


class PacketTests(unittest.TestCase):
    def test_roundtrip_and_size(self):
        packet = TelemetryPacket(
            sequence=41,
            time_ms=12500,
            rover_x=2.34,
            rover_y=-0.5,
            yaw_deg=8.0,
            tracks=(
                TrackFix(1, "rock", 0.91, 6.2, 1.25),
                TrackFix(2, "ice", 0.8, 13.4, -0.8),
            ),
        )
        raw = encode_packet(packet)
        self.assertLessEqual(len(raw), 64)
        self.assertLess(len(raw), len(json_snapshot(packet)) / 2)
        back = decode_packet(raw)
        self.assertEqual(back.sequence, 41)
        self.assertEqual(back.tracks[1].kind, "ice")
        self.assertAlmostEqual(back.tracks[0].x, 6.2, delta=0.02)
        self.assertAlmostEqual(back.rover_y, -0.5, delta=0.02)

    def test_truncated_packet_is_rejected(self):
        packet = TelemetryPacket(1, 10, 0, 0, 0, (TrackFix(1, "rock", 0.5, 1, 1),))
        with self.assertRaises(ValueError):
            decode_packet(encode_packet(packet)[:10])
