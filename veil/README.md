# VEIL

Software-emulated lunar scout for ATEC'26, project C.R.A.T.E.R.

The camera picture stays on the rover. A local network names rocks and pits. A hazard is radioed only after a second view agrees. Mission Control draws what survived a lossy UDP link, with age and uncertainty.

The spec requires a local, emulated pipeline: perception, telemetry, dashboard. It does not set the packet size or the model. This repo uses a 20-byte message and a small NumPy network. See `docs/LIMITATIONS.md` for what that does not prove.

## Layout

`python run_mission.py` runs onboard, radio, and ground as separate processes. Tests can stay in one process with `--in-process`. `docs/ARCHITECTURE.md` is the map. `docs/TELEMETRY_PROTOCOL.md` is the 20-byte contract.

## Setup

```bash
cd veil
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS or Linux the activate path is `.venv/bin/activate`. Dependencies are `numpy` and `opencv-python` only. This checkout was verified with Python 3.13.7. Pinned versions are in `requirements.txt`.

`python -m veil.verify` runs tests, acceptance, bench, geometry, robustness, and ablation. `python -m veil.ingest video.mp4` plays a clip that has no pose file and records the constant-velocity warning. See `docs/JURY_QUICKSTART.md`.

## Train

```bash
python -m veil.train
```

This renders labeled patches and writes `outputs/hazard_net.npz` and `outputs/train_set.npz`. Inference does not retrain. If the weights file is missing, the detector raises.

## Test

```bash
python -m pytest tests -q
python -m veil.accept
```

Acceptance JSON lands in `artifacts/acceptance/`. A live run cannot write those files. How to read them is in `docs/TESTING_AND_ACCEPTANCE.md`.

## Live dashboard

```bash
python run_mission.py
```

Open http://127.0.0.1:8877. Ctrl+C stops it. The live report is `artifacts/live/report.json`. One process may own port 8877. A second start fails instead of attaching to the first.

The corner image is a local preview. The radio carries the 20-byte notes.

## Troubleshooting

- `missing onboard weights`: run `python -m veil.train`.
- Port already in use: stop the other `run_mission.py`.
- Acceptance `passed: false`: read `rejected_inconsistent`, `false_hazards_delivered`, and `missed_hazards` in the scenario JSON. Do not replace that file by hand.

## Known limits

Synthetic images, threshold proposals, and a flat ground. Details are in `docs/LIMITATIONS.md`. The audit is `docs/ENGINEERING_AUDIT.md`.
