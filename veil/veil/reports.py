"""Atomic report files. A live run never writes the acceptance report."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
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
