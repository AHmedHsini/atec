import unittest

from crater.control.story import Chronicle
from crater.link.packet import TrackFix


class StoryTests(unittest.TestCase):
    def test_chapters_follow_the_link(self):
        chronicle = Chronicle()
        chronicle.start(0.0)
        chronicle.on_packet(1.0, [TrackFix(1, "rock", 0.9, 3.2, 0.4)])
        chronicle.on_packet(2.0, [TrackFix(2, "ice", 0.88, 12.0, -0.5)])
        chronicle.on_link(9.0, True)
        chronicle.on_link(11.5, False)
        text = " ".join(line.text for line in chronicle.lines)
        chapters = [line.chapter for line in chronicle.lines]
        self.assertIn("Blackout", chapters)
        self.assertIn("The Find", chapters)
        self.assertIn("Signal Restored", chapters)
        self.assertIn("inside 6 m", text)
        self.assertEqual(sum(1 for line in chronicle.lines if "Bright return" in line.text), 1)
        chronicle.on_packet(12.0, [TrackFix(2, "ice", 0.88, 12.0, -0.5)])
        self.assertEqual(sum(1 for line in chronicle.lines if "Bright return" in line.text), 1)
