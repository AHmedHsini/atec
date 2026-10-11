"""Atomic report files. A live run never writes the acceptance report."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_revision(root: Path) -> str | None:
    cursor = root.resolve()
    for _ in range(5):
        git_dir = cursor / ".git"
        if git_dir.exists():
            try:
                return subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=cursor,
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                head = git_dir / "HEAD"
                if head.is_file():
                    text = head.read_text(encoding="utf-8").strip()
                    if text.startswith("ref:"):
                        ref = cursor / ".git" / text.split(" ", 1)[1].strip()
                        if ref.is_file():
                            return ref.read_text(encoding="utf-8").strip()[:12]
                    return text[:12]
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    return None


def stamp(report: dict, scenario: str, seed: int, frames: int, dt: float, step_m: float) -> dict:
    stamped = dict(report)
    stamped["scenario"] = scenario
    stamped["seed"] = seed
    stamped["frames"] = frames
    stamped["dt_s"] = dt
    stamped["step_m"] = step_m
    stamped["generated_at"] = datetime.now(timezone.utc).isoformat()
    stamped["git_revision"] = git_revision(Path(__file__).resolve().parents[1])
    return stamped


def write_json(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    os.replace(temporary, path)
