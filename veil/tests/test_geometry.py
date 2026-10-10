import math

import numpy as np

from veil.camera import CX, CY, HEIGHT, WIDTH, pixel_to_ground, project


def test_round_trip_across_the_ground_in_front_of_the_scout():
    scout = (0.2, 1.0)
    for x in (-1.5, 0.0, 1.2):
        for y in (2.0, 4.5, 8.0):
            u, v, z_cam = project(np.array([[x, y, 0.0]]), scout)
            assert z_cam[0] > 0
            hit = pixel_to_ground(float(u[0]), float(v[0]), scout)
            assert hit is not None
            assert abs(hit[0] - x) < 0.02
            assert abs(hit[1] - y) < 0.02


def test_points_behind_the_camera_do_not_invent_a_ground_hit():
    u, v, z_cam = project(np.array([[0.0, -2.0, 0.0]]), (0.0, 0.0))
    assert z_cam[0] < 0
    assert pixel_to_ground(float(u[0]), float(v[0]), (0.0, 0.0)) is None


def test_parallel_and_non_finite_rays_miss():
    assert pixel_to_ground(float("nan"), 10, (0.0, 0.0)) is None
    assert pixel_to_ground(CX, CY, (float("inf"), 0.0)) is None
    looking_up = pixel_to_ground(CX, 0, (0.0, 0.0))
    assert looking_up is None


def test_image_edges_that_look_into_the_ground_are_finite():
    hit = pixel_to_ground(WIDTH - 2, HEIGHT - 2, (0.0, 0.0))
    assert hit is not None
    assert all(math.isfinite(value) for value in hit)
