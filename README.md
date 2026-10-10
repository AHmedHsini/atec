# C.R.A.T.E.R.

ATEC'26 software pipeline for a lunar scout that never had GPS, a cloud, or a fat radio.

Perception runs on the frame. Coordinates are packed into a fixed binary packet. A UDP socket on loopback drops packets and goes silent in blackouts. Mission Control draws what arrived, how old it is, and a chronicle that only speaks when the telemetry changes.

```powershell
python -m crater
```

Open http://127.0.0.1:8765 . The pass lasts about 36 seconds, then the board holds the debrief.

```powershell
python -m crater --benchmark
python -m unittest discover -s tests -t . -v
```

The benchmark writes `output/preselection_report.json`.

## Who owns what

The spec scores five technical slices plus a live pitch. The build is split so each person owns one vertical piece, including its tests. The pipeline file is the shared seam: do not change another person's package without them.

### Ahmed — onboard scout (`crater/scout/`)

Perception and the pixel-to-ground math. This is the 15% edge-perception score and the 15% spatial-math score.

- `camera.py` — pinhole model, flat floor, known height and pitch. You defend this in Q&A.
- `detect.py` — local detector. No cloud APIs. The baseline is classical segmentation so it runs offline.
- `track.py` — stable track ids across frames.
- `scene.py` — synthetic crater feed with known hazard positions, so error is measurable.
- Tests: `tests/test_camera.py`, `tests/test_detect.py`.

Your next work: replace the color detector with a small onboard model if you want a higher perception score, keep `Detection` fields unchanged, and write down the transform assumptions for the 7-minute pitch.

### Wissal — crater link (`crater/link/`)

Compact telemetry and the lossy radio. This is the 15% telemetry score, and it is what makes the dashboard's age honest.

- `packet.py` — fixed 16-byte header plus 8 bytes per hazard. Centimeter integers, no JSON on the wire.
- `channel.py` — random drops, timed blackouts, and a real localhost UDP socket.
- Tests: `tests/test_packet.py`, `tests/test_channel.py`.

Your next work: a bandwidth budget (bytes per second at 10 Hz versus the JSON baseline), sequence-gap handling when the counter wraps, and an optional repeat of the last packet if you want survival higher without growing the format.

### Hammoud — Mission Control and the chronicle (`crater/control/`)

The operator view and the storytelling layer. This is the 15% dashboard score. The chronicle is the extra feature: it narrates the pass from packets, and it does not invent hazards.

- `board.py` — tracks, ages, and link counters built only from received packets.
- `uncertainty.py` — circle grows with low confidence, range, and age.
- `story.py` — chapters: Descent, First Survey, The Find, Blackout, Signal Restored, Debrief.
- `static/` — map, freshness, and the story column.
- Tests: `tests/test_story.py`.

Your next work: tighten the operator layout for the live demo, and keep every new story line tied to a track id or a link event so Q&A can ask "where did this sentence come from?"

### Shared

`crater/pipeline.py` is the automated path from video frame to dashboard. `tests/test_pipeline.py` is the integration check. Ahmed wires scout changes through it, Wissal wires packet changes, Hammoud wires board changes. One person runs the benchmark before the code freeze and commits `output/preselection_report.json` if the jury wants the test output in the repo.
