"""Upload the VEIL panorama demo to a Hugging Face Docker Space.

Usage (after `hf auth login` with a write token):
    python deploy/huggingface/deploy.py <user>/<space-name>
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FILES = [
    "veil/run_mission.py",
    "veil/requirements.txt",
    "veil/outputs/hazard_net.npz",
    "veil/outputs/hazard_net_polar.npz",
]
FOLDERS = ["veil/veil", "veil/web", "dataset/panorama"]


def stage(target: Path) -> None:
    shutil.copy(HERE / "Dockerfile", target / "Dockerfile")
    shutil.copy(HERE / "README.md", target / "README.md")
    for name in FILES:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / name, target / name)
    for name in FOLDERS:
        shutil.copytree(REPO / name, target / name, ignore=shutil.ignore_patterns("__pycache__"))


def main() -> None:
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("usage: python deploy/huggingface/deploy.py <user>/<space-name>")
    space = sys.argv[1]
    api = HfApi()
    api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        stage(Path(folder))
        api.upload_folder(repo_id=space, repo_type="space", folder_path=folder, commit_message="Deploy VEIL panorama demo")
    print(f"https://huggingface.co/spaces/{space}")


if __name__ == "__main__":
    main()
