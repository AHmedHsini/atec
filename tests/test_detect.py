import math
import unittest

from crater.scout.camera import CameraModel
from crater.scout.detect import detect_hazards
from crater.scout.scene import CraterScene


class DetectTests(unittest.TestCase):
    def test_visible_hazards_are_found_near_truth(self):
        camera = CameraModel()
        scene = CraterScene(camera)
        rover_x = 1.0
        frame = scene.render(rover_x, 0.0, 0.0)
        detections = detect_hazards(frame)
        visible = scene.visible(rover_x, 0.0, 0.0)
        self.assertGreaterEqual(len(visible), 3)
        for item in visible:
            point = None
            for detection in detections:
                if detection.kind != item.hazard.kind:
                    continue
                ground = camera.pixel_to_ground(detection.u, detection.v, rover_x, 0.0, 0.0)
                if ground is None:
                    continue
                dist = math.hypot(ground[0] - item.hazard.x, ground[1] - item.hazard.y)
                if point is None or dist < point:
                    point = dist
            self.assertIsNotNone(point, item.hazard.kind)
            self.assertLess(point, 0.6)
