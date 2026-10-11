"""PolarDB patches for HazardNet.

Reads NASA POLAR Stereo zips from dataset/polar/ (gitignored). Labels come from
GroundTruth_RectifiedRangeMaps organized XYZ TIFFs: local height residual above
a blurred ground surface. High residuals → rock, low → pit, near zero → none.
Artifact samples are synthesized as a dark lens disk (same idea as the sim).

Images are stereo-rectified to match the range maps and tone-mapped to 8-bit.
16x16 patches are taken at native 1936x1216 resolution (same feature recipe as
veil.dataset.features; u/v are normalized by native width/height).
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import cv2
import numpy as np

from veil.camera import HEIGHT, WIDTH
from veil.dataset import NAMES, PATCH
from veil.sim import HALLUCINATION_RADIUS, HALLUCINATION_UV

POLAR_ROOT = Path(__file__).resolve().parents[2] / "dataset" / "polar"
NATIVE_W = 1936
NATIVE_H = 1216
# Native 16px is only a few cm; crop a wider window and shrink to PATCH so the
# 262-d vector still matches HazardNet while seeing a whole rock/pit.
CONTEXT = 64

# All registered terrains with organized range maps. Hold-out is by frame hash
# (and Terrains 9/12 contribute only to val) so lighting/pose still generalizes.
ALL_TERRAINS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
VAL_ONLY_TERRAINS = (9, 12)

TERRAIN_ZIPS = {
    1: "Terrain01_LargeOnly.zip",
    2: "Terrain02_1BigRock.zip",
    3: "Terrain03_MeanDistr.zip",
    4: "Terrain04_MeanDistr.zip",
    5: "Terrain05_MeanDistr.zip",
    6: "Terrain06_Smooth.zip",
    7: "Terrain07_LargeSmallRocksSide.zip",
    8: "Terrain08_1BigRock.zip",
    9: "Terrain09_MediumOnly.zip",
    10: "Terrain10_MediumOnly.zip",
    11: "Terrain11_FreshCrater.zip",
    12: "Terrain12_OldCrater.zip",
    13: "Terrain13_LHT.zip",
}

GT_ZIP = "GroundTruth_RectifiedRangeMaps.zip"
# Organized XYZ TIFFs store rectified camera-frame points as (Z_depth, X, Y) in mm.
ROCK_M = 0.025
PIT_LOCAL_M = -0.028
PIT_PLANE_M = -0.055
FLAT_LOCAL_M = 0.010
FLAT_PLANE_M = 0.022
LOCAL_KERNEL = 61
MIN_BLOB = 60


def generate_polar(
    root: Path | None = None,
    seed: int = 11,
    per_class: int = 1200,
    max_frames_per_terrain: int = 36,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return train/val arrays from PolarDB plus a metadata dict for the report."""
    root = Path(root or POLAR_ROOT)
    rng = np.random.default_rng(seed)
    train_samples = {i: [] for i in range(len(NAMES))}
    val_samples = {i: [] for i in range(len(NAMES))}
    used_train: list[str] = []
    used_val: list[str] = []

    train_terrains_seen: set[int] = set()
    val_terrains_seen: set[int] = set()
    for terrain in ALL_TERRAINS:
        frames = list(_iter_frames(root, terrain, max_frames_per_terrain, rng))
        for frame_id, image_u8, labels in frames:
            # Deterministic hold-out: ~25% of frames + entire val-only terrains.
            hold = terrain in VAL_ONLY_TERRAINS or (_frame_hash(frame_id) % 4 == 0)
            if hold:
                bucket, used = val_samples, used_val
                val_terrains_seen.add(terrain)
                augment = False
            else:
                bucket, used = train_samples, used_train
                train_terrains_seen.add(terrain)
                augment = True
            used.append(frame_id)
            _sample_frame(bucket, image_u8, labels, rng, augment=augment)

    # Synthesize artifact class (PolarDB has no lens-smudge labels).
    train_art_terrains = tuple(sorted(train_terrains_seen)) or ALL_TERRAINS
    val_art_terrains = tuple(sorted(val_terrains_seen)) or VAL_ONLY_TERRAINS
    _add_artifacts(train_samples, root, train_art_terrains, rng, n=per_class)
    _add_artifacts(val_samples, root, val_art_terrains, rng, n=max(120, per_class // 5))

    train_x, train_y, train_counts = _balance(train_samples, per_class, rng)
    val_cap = max(120, per_class // 5)
    val_x, val_y, val_counts = _balance(val_samples, val_cap, rng)

    meta = {
        "train_terrains": sorted(train_terrains_seen),
        "val_terrains": sorted(val_terrains_seen),
        "val_only_terrains": list(VAL_ONLY_TERRAINS),
        "holdout": "Terrains 9+12 fully held out; ~25% of other frames by path hash",
        "train_frames": used_train,
        "val_frames": used_val,
        "train_per_class_before_balance": {NAMES[i]: len(train_samples[i]) for i in range(4)},
        "val_per_class_before_balance": {NAMES[i]: len(val_samples[i]) for i in range(4)},
        "train_per_class": train_counts,
        "val_per_class": val_counts,
        "label_rule": {
            "source": "GroundTruth_RectifiedRangeMaps Pos*_org-XYZ.tif",
            "point_channels": "(Z_depth, X, Y) mm in rectified camera frame",
            "height": "signed distance to fitted ground plane (meters), plus local residual vs blur",
            "rock": f"local plane-residual > {ROCK_M} m",
            "pit": f"local residual < {PIT_LOCAL_M} m OR plane height < {PIT_PLANE_M} m (craters)",
            "none": f"|local| <= {FLAT_LOCAL_M} m and |plane height| <= {FLAT_PLANE_M} m",
            "artifact": "synthetic dark disk on tone-mapped image (sim-style)",
            "features": (
                f"rectify → 8-bit tone-map at native 1936x1216 → "
                f"{CONTEXT}x{CONTEXT} crop resized to {PATCH}x{PATCH} → "
                "same 262-d recipe as veil.dataset.features (u/v ÷ native size); train-time gain/bias"
            ),
        },
    }
    return train_x, train_y, val_x, val_y, meta


def _polar_features(image: np.ndarray, u: float, v: float) -> np.ndarray:
    """262-d vector: CONTEXT crop → PATCH, plus the same six scalars as features()."""
    half = CONTEXT // 2
    padded = np.pad(image, half, mode="edge")
    x = int(round(u)) + half
    y = int(round(v)) + half
    crop = padded[y - half : y + half, x - half : x + half]
    small = cv2.resize(crop, (PATCH, PATCH), interpolation=cv2.INTER_AREA)
    flat = small.astype(np.float32).reshape(-1) / 255.0
    return np.concatenate(
        [
            flat,
            np.array(
                [
                    float(flat.mean()),
                    float(flat.std()),
                    float((flat < 0.25).mean()),
                    float((flat > 0.75).mean()),
                    u / NATIVE_W,
                    v / NATIVE_H,
                ],
                dtype=np.float32,
            ),
        ]
    )


def _iter_frames(root: Path, terrain: int, limit: int, rng: np.random.Generator):
    zip_path = root / TERRAIN_ZIPS[terrain]
    gt_path = root / GT_ZIP
    if not zip_path.exists() or not gt_path.exists():
        raise FileNotFoundError(f"missing PolarDB archive under {root}")

    with zipfile.ZipFile(zip_path) as zf:
        prefix = _zip_prefix(zf)
        calib = _parse_calib(zf.read(f"{prefix}/stereo.calibration").decode("utf-8", "replace"))
        remap = _rectify_maps(calib)
        cam_names = [
            n
            for n in zf.namelist()
            if "/CamL_" in n
            and n.endswith(".png")
            and ("_Loff/" in n or "_Lon/" in n)
            and re.search(r"_(0256|0512|1024)\.png$", n)
        ]
        # Prefer mid exposures; shuffle for diversity across sun/pose.
        rng.shuffle(cam_names)
        cam_names = cam_names[:limit]

        # Cache range labels per Pos letter.
        label_cache: dict[str, np.ndarray] = {}
        for name in cam_names:
            pos_letter = _pos_letter(name)
            if pos_letter not in label_cache:
                label_cache[pos_letter] = _load_label_map(gt_path, terrain, pos_letter)
            labels_native = label_cache[pos_letter]
            if labels_native is None:
                continue
            raw = _read_png(zf, name)
            if raw is None:
                continue
            rect = cv2.remap(raw, remap[0], remap[1], cv2.INTER_LINEAR)
            image_u8 = _tone_map(rect)
            yield name, image_u8, labels_native


def _sample_frame(
    bucket: dict[int, list[np.ndarray]],
    image: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    per_label: int = 14,
    augment: bool = True,
) -> None:
    h, w = labels.shape[:2]
    for label in (0, 1, 2):
        ys, xs = np.where(labels == label)
        if len(xs) == 0:
            continue
        keep = (xs >= 16) & (xs < w - 16) & (ys >= 16) & (ys < h - 16)
        xs, ys = xs[keep], ys[keep]
        if len(xs) == 0:
            continue
        if label in (1, 2):
            points = _blob_centers(labels == label, rng, per_label)
            # Extra interior pixels so small rocks still contribute.
            extra = min(per_label, len(xs))
            if extra:
                idx = rng.choice(len(xs), size=extra, replace=False)
                points.extend((int(xs[j]), int(ys[j])) for j in idx)
        else:
            idx = rng.choice(len(xs), size=min(per_label, len(xs)), replace=False)
            points = list(zip(xs[idx].tolist(), ys[idx].tolist()))
        for u, v in points:
            if not _confident_patch(labels, u, v, label):
                continue
            view = _augment(image, rng) if augment else image
            bucket[label].append(_polar_features(view, float(u), float(v)))
            if augment and rng.random() < 0.5:
                bucket[label].append(_polar_features(_augment(image, rng), float(u), float(v)))


def _add_artifacts(
    bucket: dict[int, list[np.ndarray]],
    root: Path,
    terrains: tuple[int, ...],
    rng: np.random.Generator,
    n: int,
) -> None:
    """Paint a dark disk on PolarDB frames the same way the sim marks lens smudge."""
    made = 0
    tries = 0
    while made < n and tries < n * 8:
        tries += 1
        terrain = int(rng.choice(terrains))
        zip_path = root / TERRAIN_ZIPS[terrain]
        with zipfile.ZipFile(zip_path) as zf:
            prefix = _zip_prefix(zf)
            names = [
                nm
                for nm in zf.namelist()
                if "/CamL_" in nm and nm.endswith("_0512.png") and "_Loff/" in nm
            ]
            if not names:
                continue
            name = names[int(rng.integers(0, len(names)))]
            raw = _read_png(zf, name)
            calib = _parse_calib(zf.read(f"{prefix}/stereo.calibration").decode("utf-8", "replace"))
        if raw is None:
            continue
        remap = _rectify_maps(calib)
        rect = cv2.remap(raw, remap[0], remap[1], cv2.INTER_LINEAR)
        image = _tone_map(rect)
        # Scale the sim hallucination locus into native pixels on some frames.
        if rng.random() < 0.4:
            u = int(HALLUCINATION_UV[0] * NATIVE_W / WIDTH)
            v = int(HALLUCINATION_UV[1] * NATIVE_H / HEIGHT)
            radius = max(8, int(HALLUCINATION_RADIUS * NATIVE_W / WIDTH))
        else:
            u = int(rng.integers(40, NATIVE_W - 40))
            v = int(rng.integers(40, NATIVE_H - 40))
            radius = int(rng.integers(10, 28))
        cv2.circle(image, (u, v), radius, int(rng.integers(4, 28)), -1)
        bucket[3].append(_polar_features(image, float(u), float(v)))
        made += 1


def _balance(
    samples: dict[int, list[np.ndarray]], per_class: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, dict]:
    available = min(per_class, min(len(samples[i]) for i in samples))
    if available < 1:
        raise RuntimeError(f"PolarDB produced empty classes: {[len(samples[i]) for i in samples]}")
    xs: list[np.ndarray] = []
    ys: list[int] = []
    counts = {}
    for label in range(len(NAMES)):
        order = rng.permutation(len(samples[label]))[:available]
        for index in order:
            xs.append(samples[label][index])
            ys.append(label)
        counts[NAMES[label]] = int(available)
    return np.stack(xs).astype(np.float32), np.array(ys, dtype=np.int64), counts


def _load_label_map(gt_zip: Path, terrain: int, pos_letter: str) -> np.ndarray | None:
    import tempfile

    cache = gt_zip.parent / "cache" / f"labels_T{terrain:02d}_Pos{pos_letter}.png"
    if cache.exists():
        labels = cv2.imread(str(cache), cv2.IMREAD_UNCHANGED)
        if labels is not None:
            return labels

    member = f"GroundTruth_RectifiedRangeMaps/Terrain{terrain:02d}/Pos{pos_letter}_org-XYZ.tif"
    with zipfile.ZipFile(gt_zip) as zf:
        if member not in zf.namelist():
            return None
        blob = zf.read(member)
    # Multi-channel float TIFF needs a real path; imdecode often fails.
    with tempfile.NamedTemporaryFile(suffix=".tif") as tmp:
        tmp.write(blob)
        tmp.flush()
        xyz = cv2.imread(tmp.name, cv2.IMREAD_UNCHANGED)
    if xyz is None or xyz.ndim != 3 or xyz.shape[2] < 3:
        return None
    labels = _height_labels(xyz)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache), labels)
    return labels


def _height_labels(xyz: np.ndarray) -> np.ndarray:
    """Map organized XYZ (mm) to class ids at native resolution.

    TIFF channels are (Z_depth, X, Y) in the rectified camera frame (mm).
    Height is signed distance to a fitted ground plane. Rocks are local highs
    versus a ~60px blur of that height; craters also use absolute plane depth.
    """
    depth = xyz[:, :, 0].astype(np.float32) / 1000.0
    x = xyz[:, :, 1].astype(np.float32) / 1000.0
    y = xyz[:, :, 2].astype(np.float32) / 1000.0
    valid = ((xyz[:, :, 0] != 0) | (xyz[:, :, 1] != 0) | (xyz[:, :, 2] != 0)) & (depth > 0.5) & (
        depth < 8.0
    )
    height = np.zeros(depth.shape, np.float32)
    if int(valid.sum()) > 1000:
        idx = np.flatnonzero(valid.ravel())
        rng = np.random.default_rng(0)
        pick = idx[rng.choice(len(idx), size=min(80000, len(idx)), replace=False)]
        pts = np.stack([x.ravel()[pick], y.ravel()[pick], depth.ravel()[pick]], axis=1)
        center = pts.mean(axis=0)
        _, _, vh = np.linalg.svd(pts - center, full_matrices=False)
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-12)
        if np.dot(center, normal) < 0:
            normal = -normal
        height[valid] = (np.stack([x[valid], y[valid], depth[valid]], axis=1) - center) @ normal

    mask = valid.astype(np.float32)
    local = cv2.blur(np.where(valid, height, 0.0), (LOCAL_KERNEL, LOCAL_KERNEL))
    count = cv2.blur(mask, (LOCAL_KERNEL, LOCAL_KERNEL))
    ground = local / np.maximum(count, 1e-6)
    local_res = np.where(valid, height - ground, 0.0)

    labels = np.full(depth.shape, 255, np.uint8)  # 255 = ignore
    flat = valid & (np.abs(local_res) <= FLAT_LOCAL_M) & (np.abs(height) <= FLAT_PLANE_M)
    labels[flat] = 0
    pit = valid & ((local_res <= PIT_LOCAL_M) | (height <= PIT_PLANE_M))
    rock = valid & (local_res >= ROCK_M)
    labels[pit] = 2
    labels[rock] = 1  # rock wins on raised crater rims

    for cls in (1, 2):
        blob = (labels == cls).astype(np.uint8)
        blob = cv2.morphologyEx(blob, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        blob = cv2.morphologyEx(blob, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n_comp, markers, stats, _ = cv2.connectedComponentsWithStats(blob, connectivity=8)
        keep = np.zeros_like(blob)
        for i in range(1, n_comp):
            if stats[i, cv2.CC_STAT_AREA] >= MIN_BLOB:
                keep[markers == i] = 1
        labels[labels == cls] = 255
        labels[keep > 0] = cls
    return labels


def _blob_centers(mask: np.ndarray, rng: np.random.Generator, limit: int) -> list[tuple[int, int]]:
    u8 = mask.astype(np.uint8)
    h, w = u8.shape
    n, _, stats, centroids = cv2.connectedComponentsWithStats(u8, connectivity=8)
    points: list[tuple[int, int]] = []
    order = list(range(1, n))
    rng.shuffle(order)
    for i in order:
        if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB // 2:
            continue
        u = int(round(centroids[i][0]))
        v = int(round(centroids[i][1]))
        if 16 <= u < w - 16 and 16 <= v < h - 16:
            points.append((u, v))
        if len(points) >= limit:
            break
    if not points:
        ys, xs = np.where(mask)
        if len(xs):
            idx = rng.choice(len(xs), size=min(limit, len(xs)), replace=False)
            points = [(int(xs[j]), int(ys[j])) for j in idx]
    return points


def _tone_map(image16: np.ndarray) -> np.ndarray:
    x = np.clip(image16.astype(np.float32), 0, 4095)
    lo, hi = np.percentile(x, [2, 98])
    if hi <= lo + 1:
        lo, hi = 0.0, 4095.0
    return np.clip((x - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _augment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    gain = float(rng.uniform(0.65, 1.35))
    bias = float(rng.integers(-28, 29))
    out = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        k = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def _confident_patch(labels: np.ndarray, u: int, v: int, label: int) -> bool:
    """Keep patches whose neighborhood mostly agrees with the class label."""
    # Small core around the sample — rocks are compact vs CONTEXT windows.
    half = 12 if label == 1 else (CONTEXT // 4 if label == 2 else CONTEXT // 2)
    y0, y1 = max(0, v - half), min(labels.shape[0], v + half)
    x0, x1 = max(0, u - half), min(labels.shape[1], u + half)
    window = labels[y0:y1, x0:x1]
    if window.size == 0:
        return False
    if label == 0:
        return float((window == 0).mean()) >= 0.80
    if label == 1:
        return float((window == 1).mean()) >= 0.20
    return float((window == 2).mean()) >= 0.30


def _parse_calib(text: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        out[key.strip()] = np.array([float(x) for x in value.split()], dtype=np.float64)
    return out


def _rectify_maps(calib: dict[str, np.ndarray]):
    k1 = calib["CAMERA_MATRIX_LEFT"].reshape(3, 3)
    d1 = calib["DISTORTION_COEFFICIENTS_LEFT"][:4]
    k2 = calib["CAMERA_MATRIX_RIGHT"].reshape(3, 3)
    d2 = calib["DISTORTION_COEFFICIENTS_RIGHT"][:4]
    r = calib["ROTATION_MATRIX"].reshape(3, 3)
    t = calib["TRANSLATION_VECTOR"].reshape(3, 1)
    size = (NATIVE_W, NATIVE_H)
    r1, _, p1, _, _, _, _ = cv2.stereoRectify(k1, d1, k2, d2, size, r, t, alpha=0)
    return cv2.initUndistortRectifyMap(k1, d1, r1, p1, size, cv2.CV_32FC1)


def _zip_prefix(zf: zipfile.ZipFile) -> str:
    for name in zf.namelist():
        if name.endswith("/stereo.calibration"):
            return name.rsplit("/", 1)[0]
    raise FileNotFoundError("stereo.calibration missing from terrain zip")


def _pos_letter(member: str) -> str:
    match = re.search(r"/Pos([ABC])_", member)
    if not match:
        raise ValueError(f"no Pos letter in {member}")
    return match.group(1)


def _read_png(zf: zipfile.ZipFile, name: str) -> np.ndarray | None:
    buf = np.frombuffer(zf.read(name), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def patch_features(image: np.ndarray, u: float, v: float) -> np.ndarray:
    """Public name for the 262-d crop the PolarDB network was trained on."""
    return _polar_features(image, u, v)


def _frame_hash(frame_id: str) -> int:
    h = 2166136261
    for ch in frame_id.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)
