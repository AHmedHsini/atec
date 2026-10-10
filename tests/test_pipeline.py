import unittest

from crater.pipeline import run_mission


class PipelineTests(unittest.TestCase):
    def test_short_pass_reaches_the_board(self):
        result = run_mission(seconds=3.0, drop_rate=0.0, realtime=False, seed=1)
        report = result.report
        self.assertGreater(report["perception_fps"], 5)
        self.assertGreater(report["recall"], 0.7)
        self.assertLess(report["localization_error_m"], 0.5)
        self.assertGreater(report["packets_delivered"], 0)
        self.assertGreater(report["story_lines"], 2)
        self.assertLess(report["packet_bytes"], report["json_bytes"])
