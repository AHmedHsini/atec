import numpy as np

from veil.camera import pixel_to_ground, project
from veil.mission import run
from veil.packet import decode, encode


def test_ground_plane_roundtrip():
    point = np.array([[1.1, 4.0, 0.0]])
    u, v, z_cam = project(point, (0.0, 0.0))
    assert z_cam[0] > 0
    hit = pixel_to_ground(float(u[0]), float(v[0]), (0.0, 0.0))
    assert hit is not None
    assert abs(hit[0] - 1.1) < 0.02
    assert abs(hit[1] - 4.0) < 0.02


def test_packet_rejects_corruption():
    packet = encode(1, 4, 1500, 2, 2, 0, 80, 0.35, 5.0, 0.12, 0.4)
    assert len(packet) == 20
    assert decode(packet)["y"] == 5.0
    broken = bytearray(packet)
    broken[12] ^= 0xFF
    assert decode(bytes(broken)) is None


def test_hazard_net_is_trained_on_generated_views():
    from veil.model import load

    net = load()
    assert net.samples >= 1000
    assert net.val_accuracy >= 0.85


def test_pipeline_keeps_real_hazards_and_drops_the_shadow():
    report = run(live=False, loss=0.3, hold=0, scenario="acceptance-short")
    assert report["passed"], report
    assert report["packet_bytes"] == 20
    assert report["bytes_sent"] < report["naive_json_bytes"] / 2
    assert report["bytes_sent"] < report["float32_bytes"]
    assert report["fps"] > 15
    assert report["rejected_inconsistent"] >= 1
    assert report["false_hazards_delivered"] == 0


def test_long_blackout_still_rejects_the_sliding_track():
    report = run(
        live=False,
        loss=0.3,
        hold=0,
        frames=130,
        dt=0.2,
        blackout=(4.0, 12.0),
        scenario="acceptance-long",
    )
    assert report["hazards_delivered"] == 4
    assert report["false_hazards_delivered"] == 0
    assert report["rejected_inconsistent"] >= 1
    assert report["passed"], report
