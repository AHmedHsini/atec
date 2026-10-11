import numpy as np

from veil.camera import load_camera
from veil.packet import decode, encode
from veil.source import VideoFrameSource


def test_corrupted_truncated_and_reordered_packets_do_not_invent_hazards():
    packet = encode(1, 4, 1500, 2, 2, 0, 80, 0.35, 5.0, 0.12, 0.4)
    rejected = 0
    rng = np.random.default_rng(1)
    for _ in range(40):
        broken = bytearray(packet)
        broken[int(rng.integers(0, 20))] ^= 1 << int(rng.integers(0, 8))
        if decode(bytes(broken)) is None:
            rejected += 1
    assert rejected >= 30
    assert decode(packet[:10]) is None
    first = decode(packet)
    second = decode(encode(1, 9, 2000, 2, 2, 0, 80, 0.40, 5.0, 0.12, 0.4))
    assert first["seq"] == 4 and second["seq"] == 9


def test_garbage_config_and_empty_video(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text("[]", encoding="utf-8")
    try:
        load_camera(path)
        raised = False
    except ValueError:
        raised = True
    assert raised
    empty = tmp_path / "empty.mp4"
    empty.write_bytes(b"")
    source = VideoFrameSource(empty, pose_source="constant")
    assert list(source.frames()) == []
    assert source.bad_frames >= 1
