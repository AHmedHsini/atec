"""One command: tests, acceptance, and the measurement scripts."""

from __future__ import annotations

import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from veil.reports import git_revision

ROOT = Path(__file__).resolve().parents[1]


def _run(args: list[str]) -> int:
    print("+", " ".join(args), flush=True)
    return subprocess.call(args, cwd=ROOT)


def main() -> None:
    junit = ROOT / "artifacts" / "verify" / "junit.xml"
    junit.parent.mkdir(parents=True, exist_ok=True)
    steps = [
        ("pytest", [sys.executable, "-m", "pytest", "tests", "-q", f"--junitxml={junit}"]),
        ("acceptance", [sys.executable, "-m", "veil.accept"]),
        ("bench", [sys.executable, "-m", "veil.bench"]),
        ("geometry", [sys.executable, "-m", "veil.validate_geometry"]),
        ("robustness", [sys.executable, "-m", "veil.robustness"]),
        ("ablation", [sys.executable, "-m", "veil.ablation"]),
    ]
    results = []
    failed = False
    for name, command in steps:
        code = _run(command)
        results.append((name, code == 0))
        failed = failed or code != 0
    lines = [
        "# VEIL verification",
        "",
        f"generated: {datetime.now(timezone.utc).isoformat()}",
        f"git: {git_revision(ROOT)}",
        f"python: {platform.python_version()}",
        "",
        "| category | result |",
        "| --- | --- |",
    ]
    for name, ok in results:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} |")
    lines.append("")
    lines.append("Flight weights are `outputs/hazard_net.npz`. PolarDB 0.581 is not this score. The ablation file `hazard_net_no_position.npz` is not loaded by the mission.")
    summary = ROOT / "artifacts" / "verify" / "SUMMARY.md"
    summary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(summary.read_text(encoding="utf-8"))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
