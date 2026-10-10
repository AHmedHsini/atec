"""Download every POLAR Stereo archive from NASA IRG into this folder."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

BASE = "https://ti.arc.nasa.gov/dataset/IRG_PolarDB/"
FILES = [
    "PolarDB_Readme.pdf",
    "PolarDB_download/Terrain01_LargeOnly.zip",
    "PolarDB_download/Terrain02_1BigRock.zip",
    "PolarDB_download/Terrain03_MeanDistr.zip",
    "PolarDB_download/Terrain04_MeanDistr.zip",
    "PolarDB_download/Terrain05_MeanDistr.zip",
    "PolarDB_download/Terrain06_Smooth.zip",
    "PolarDB_download/Terrain07_LargeSmallRocksSide.zip",
    "PolarDB_download/Terrain08_1BigRock.zip",
    "PolarDB_download/Terrain09_MediumOnly.zip",
    "PolarDB_download/Terrain10_MediumOnly.zip",
    "PolarDB_download/Terrain11_FreshCrater.zip",
    "PolarDB_download/Terrain12_OldCrater.zip",
    "PolarDB_download/Terrain13_LHT.zip",
    "PolarDB_download/RadiometricCalibrationData.zip",
    "PolarDB_download/GroundTruth_RectifiedRangeMaps.zip",
]
OUT = Path(__file__).resolve().parent / "polar"


def _curl() -> str:
    for name in ("curl", "curl.exe"):
        path = shutil.which(name)
        if path:
            return path
    raise FileNotFoundError("curl not found on PATH")


def main() -> None:
    curl = _curl()
    OUT.mkdir(parents=True, exist_ok=True)
    names = sys.argv[1:] or FILES
    for name in names:
        if name not in FILES and not name.startswith("PolarDB"):
            # allow bare filenames that match the known list
            matches = [f for f in FILES if Path(f).name == name or f.endswith(name)]
            if len(matches) != 1:
                raise SystemExit(f"unknown file: {name}")
            name = matches[0]
        dest = OUT / Path(name).name
        if dest.exists() and dest.stat().st_size > 0:
            print(f"SKIP {dest.name} ({dest.stat().st_size} bytes)", flush=True)
            continue
        print(f"GET {name} -> {dest}", flush=True)
        subprocess.run(
            [
                curl,
                "-L",
                "--fail",
                "--retry",
                "5",
                "--retry-all-errors",
                "-C",
                "-",
                "-o",
                str(dest),
                BASE + name,
            ],
            check=True,
        )
        print(f"DONE {dest.name} {dest.stat().st_size}", flush=True)


if __name__ == "__main__":
    main()
