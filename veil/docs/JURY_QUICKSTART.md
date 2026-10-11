# Jury quickstart

Python 3.13 is what this checkout was run with. Dependencies are pinned in `requirements.txt` (`numpy`, `opencv-python`).

```bash
cd veil
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m veil.verify
python run_mission.py
```

`verify` runs the tests, the acceptance sweep, the bench, the geometry perturbations, the random-world tracker, and the ablation. It writes `artifacts/verify/SUMMARY.md` and exits non-zero if a step fails. It takes several minutes.

The live dashboard is http://127.0.0.1:8877. The camera picture is labeled onboard-only. The radio carries 20-byte packets.

To feed a video that has no pose file:

```bash
python -m veil.ingest path\to\clip.mp4 --speed 0.25 --camera camera.json
```

The report says `pose source: assumed constant velocity` when the pose file is missing. That is an assumption, not a survey.

The flight network is `outputs/hazard_net.npz`. Do not quote a PolarDB accuracy as this network's score.
