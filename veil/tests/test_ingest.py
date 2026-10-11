import json

import cv2
import numpy as np

from veil.camera import CameraSpec, load_camera, pixel_to_ground, use_camera
from veil.mission import run
from veil.source import VideoFrameSource, record_drive


def test_mp4_without_pose_file_uses_constant_velocity(tmp_path):
    record_drive(tmp_path, frames=80, step_m=0.05, dt=0.1, seed=26, write_video=True)
    (tmp_path / "drive_poses.csv").unlink()
    source = VideoFrameSource(tmp_path / "drive.mp4", speed=0.5, pose_source="constant")
    try:
        report = run(
            live=False,
            loss=0.0,
            hold=0,
            frames=80,
            dt=0.1,
            blackout=(1e9, 1e9 + 1),
            scenario="ingest-cv",
            seed=26,
            source=source,
            write_report=False,
            pace_s=0,
            exchange_s=0,
            settle_s=0,
        )
    finally:
        source.close()
        use_camera(None)
    assert "assumed constant velocity" in source.pose_warning
    assert report["hazards_delivered"] >= 1
    assert report["pose_warning"]


def test_odd_resolution_and_corrupt_frame_do_not_crash(tmp_path):
    video = tmp_path / "odd.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (100, 80), False)
    for index in range(5):
        frame = np.full((80, 100), 40 + index, np.uint8)
        writer.write(frame)
    writer.release()
    source = VideoFrameSource(video, speed=0.25, pose_source="constant")
    frames = list(source.frames())
    source.close()
    use_camera(None)
    assert frames
    assert frames[0]["image"].shape == (240, 320)
    junk = tmp_path / "bad.mp4"
    junk.write_text("not a video", encoding="utf-8")
    broken = VideoFrameSource(junk, pose_source="constant")
    assert list(broken.frames()) == []
    assert broken.bad_frames >= 1
    broken.close()
    use_camera(None)


def test_camera_config_override(tmp_path):
    path = tmp_path / "camera.json"
    path.write_text(json.dumps({"mast_height_m": 0.8, "pitch_deg": 25, "fx": 280}), encoding="utf-8")
    spec = load_camera(path)
    assert spec.mast_height_m == 0.8
    assert spec.pitch_deg == 25
    use_camera(spec)
    try:
        shifted = pixel_to_ground(160, 200, (0.0, 0.0))
        use_camera(CameraSpec())
        nominal = pixel_to_ground(160, 200, (0.0, 0.0))
    finally:
        use_camera(None)
    assert shifted is not None and nominal is not None
    assert abs(shifted[1] - nominal[1]) > 0.05
