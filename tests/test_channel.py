import unittest

from crater.link.channel import LossyLink, UdpRadio
from crater.link.packet import TelemetryPacket, decode_packet, encode_packet
from crater.pipeline import _radio_pair


class ChannelTests(unittest.TestCase):
    def test_blackout_window_drops_everything(self):
        link = LossyLink(drop_rate=0.0, first_blackout_s=2.0, blackout_period_s=10.0, blackout_length_s=1.0)
        self.assertTrue(link.consider(2.2).blackout)
        self.assertFalse(link.consider(2.2).delivered)
        self.assertTrue(link.consider(4.0).delivered)

    def test_udp_loopback_carries_a_packet(self):
        tx, rx = _radio_pair()
        packet = TelemetryPacket(3, 500, 1.0, 0.2, 0.0, ())
        try:
            tx.send(encode_packet(packet))
            payload = None
            for _ in range(10):
                payload = rx.recv()
                if payload:
                    break
            self.assertIsNotNone(payload)
            self.assertEqual(decode_packet(payload).sequence, 3)
        finally:
            tx.close()
            rx.close()
            self.assertIsInstance(tx, UdpRadio)
