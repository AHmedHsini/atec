import math
import unittest

from crater.scout.camera import CameraModel


class CameraTests(unittest.TestCase):
    def test_ground_point_roundtrip(self):
        camera = CameraModel()
        rover_x, rover_y, yaw = 1.2, -0.4, 12.0
        for x, y in ((4.0, 0.2), (7.5, -1.4), (11.0, 1.0)):
            pixel = camera.project(x, y, 0.0, rover_x, rover_y, yaw)
            self.assertIsNotNone(pixel)
            back = camera.pixel_to_ground(pixel[0], pixel[1], rover_x, rover_y, yaw)
            self.assertIsNotNone(back)
            self.assertLess(math.hypot(back[0] - x, back[1] - y), 0.05)

    def test_point_behind_camera_is_rejected(self):
        camera = CameraModel()
        self.assertIsNone(camera.project(-2.0, 0.0, 0.0, 0.0, 0.0, 0.0))
