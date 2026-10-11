"""Pull a few PolarDB photos and tone-map them for the live demo.

The archives are about a gigabyte each. This reads only the zip directory and
the left-camera frames it needs, from a terrain the PolarDB training held out.
"""

from __future__ import annotations

import json
import struct
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from veil.polar import NATIVE_H, NATIVE_W, _parse_calib, _rectify_maps, _tone_map

ROOT = Path(__file__).resolve().parents[2] / "dataset" / "polar"
DEMO = ROOT / "demo"
URL = "https://ti.arc.nasa.gov/dataset/IRG_PolarDB/PolarDB_download/Terrain09_MediumOnly.zip"
WANTED = 1


def main() -> None:
    DEMO.mkdir(parents=True, exist_ok=True)
    names = _pick_names(_central_directory(URL))
    print(f"frames {len(names)}", flush=True)
    images = []
    calib = None
    for index, (name, offset) in enumerate(names):
        raw = _read_member(URL, offset)
        if name.endswith("stereo.calibration"):
            calib = _parse_calib(raw.decode("utf-8", "replace"))
            continue
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        images.append((index, name, frame))
        print(name, None if frame is None else frame.shape, flush=True)
    if calib is None or not images:
        raise SystemExit("PolarDB demo download did not return a calibration and photos")
    maps = _rectify_maps(calib)
    fx, fy, cx, cy = _intrinsics(calib)
    written = 0
    with (DEMO / "poses.csv").open("w", encoding="utf-8") as handle:
        handle.write("frame,t_s,x_m,y_m,yaw_rad\n")
        for index, _name, frame in images:
            if frame is None:
                continue
            rect = cv2.remap(frame, maps[0], maps[1], cv2.INTER_LINEAR)
            gray = _tone_map(rect)
            for _repeat in range(4):
                cv2.imwrite(str(DEMO / f"{written:04d}.png"), gray)
                handle.write(f"{written},{written * 0.4:.1f},0.0,0.0,0.0\n")
                written += 1
    camera = {
        "width": NATIVE_W,
        "height": NATIVE_H,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "mast_height_m": 1.1,
        "pitch_deg": 32.0,
        "distortion": [],
    }
    (DEMO / "camera.json").write_text(json.dumps(camera, indent=2), encoding="utf-8")
    print(json.dumps({"frames": written, "demo": str(DEMO), "fx": round(fx, 1)}, indent=2))


def _intrinsics(calib: dict[str, np.ndarray]) -> tuple[float, float, float, float]:
    k1 = calib["CAMERA_MATRIX_LEFT"].reshape(3, 3)
    d1 = calib["DISTORTION_COEFFICIENTS_LEFT"][:4]
    k2 = calib["CAMERA_MATRIX_RIGHT"].reshape(3, 3)
    d2 = calib["DISTORTION_COEFFICIENTS_RIGHT"][:4]
    rotation = calib["ROTATION_MATRIX"].reshape(3, 3)
    translation = calib["TRANSLATION_VECTOR"].reshape(3, 1)
    _r1, _r2, p1, _p2, _q, _roi1, _roi2 = cv2.stereoRectify(
        k1, d1, k2, d2, (NATIVE_W, NATIVE_H), rotation, translation, alpha=0
    )
    return float(p1[0, 0]), float(p1[1, 1]), float(p1[0, 2]), float(p1[1, 2])


def _pick_names(entries: list[tuple[str, int]]) -> list[tuple[str, int]]:
    calib = [item for item in entries if item[0].endswith("stereo.calibration")]
    photos = [
        item
        for item in entries
        if "/CamL_" in item[0] and item[0].endswith("_0512.png") and "_Loff/" in item[0] and "/PosA_" in item[0]
    ]
    if not photos:
        photos = [item for item in entries if "/CamL_" in item[0] and item[0].endswith("_0512.png")]
    photos = photos[:WANTED]
    if not calib or not photos:
        raise SystemExit("held-out terrain zip has no left-camera frames")
    return calib[:1] + photos


def _central_directory(url: str) -> list[tuple[str, int]]:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        length = int(response.headers["Content-Length"])
    tail = _fetch(url, max(0, length - 65536), length - 1)
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0:
        raise SystemExit("zip directory is missing; the server did not return the archive tail")
    count = struct.unpack_from("<H", tail, marker + 10)[0]
    size = struct.unpack_from("<I", tail, marker + 12)[0]
    offset = struct.unpack_from("<I", tail, marker + 16)[0]
    blob = _fetch(url, offset, offset + size - 1)
    entries = []
    cursor = 0
    for _ in range(count):
        if blob[cursor : cursor + 4] != b"PK\x01\x02":
            break
        compressed = struct.unpack_from("<I", blob, cursor + 20)[0]
        name_len = struct.unpack_from("<H", blob, cursor + 28)[0]
        extra_len = struct.unpack_from("<H", blob, cursor + 30)[0]
        comment_len = struct.unpack_from("<H", blob, cursor + 32)[0]
        local_offset = struct.unpack_from("<I", blob, cursor + 42)[0]
        name = blob[cursor + 46 : cursor + 46 + name_len].decode("utf-8", "replace")
        entries.append((name, local_offset))
        cursor += 46 + name_len + extra_len + comment_len
    return entries


def _read_member(url: str, offset: int) -> bytes:
    head = _fetch(url, offset, offset + 4096)
    if head[:4] != b"PK\x03\x04":
        raise ValueError("not a zip member")
    compressed = struct.unpack_from("<I", head, 18)[0]
    name_len = struct.unpack_from("<H", head, 26)[0]
    extra_len = struct.unpack_from("<H", head, 28)[0]
    need = 30 + name_len + extra_len + compressed
    if len(head) < need:
        head += _fetch(url, offset + len(head), offset + need - 1)
    return _inflate_member(head[:need])


def _inflate_member(local: bytes) -> bytes:
    if local[:4] != b"PK\x03\x04":
        raise ValueError("not a zip member")
    method = struct.unpack_from("<H", local, 8)[0]
    compressed = struct.unpack_from("<I", local, 18)[0]
    name_len = struct.unpack_from("<H", local, 26)[0]
    extra_len = struct.unpack_from("<H", local, 28)[0]
    start = 30 + name_len + extra_len
    data = local[start : start + compressed]
    if method == 0:
        return data
    import zlib

    return zlib.decompress(data, -15)


def _fetch(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}", "User-Agent": "atec-veil"})
    with urllib.request.urlopen(request, timeout=180) as response:
        return response.read()


if __name__ == "__main__":
    main()
