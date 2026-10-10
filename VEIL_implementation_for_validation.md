# VEIL — implementation brief for review

Upload this file together with the ATEC'26 Specification Book. The code is in the `veil` folder next to this file. This brief describes the system as of 10 October 2026, after the engineering hardening pass. It is a status report, not the specification.

Ask the reviewer to compare this implementation with the specification for project C.R.A.T.E.R. (Constrained Reconnaissance, Autonomy & Telemetry in Extreme Regions): what is required, what is flexible, what is ambiguous, and what is still missing before a preselection submission. Do not treat this file as proof by itself. The measured numbers below come from `veil/artifacts/acceptance/`.

## Product

VEIL is a software-emulated lunar scout. The camera picture stays on the rover. A small local network names rocks and pits. A hazard is radioed only after two views agree on the same ground point. Mission Control plots only what survived a lossy link, and it shows how old and how uncertain each fix is.

The black dot on the left of the camera picture is a smudge stuck on the lens, painted at pixel (36, 110) with radius 7. It is not on the ground. It stays on the same pixel while the scout drives, so its guessed ground point slides. The tracker rejects that track. It is not drawn on the map.

## What the spec fixes, and what this repo chose

Fixed by the spec:

- 100% software-emulated. No rover hardware.
- Data flow: Perception, then Telemetry, then Dashboard.
- Local perception. No cloud inference API.
- Severe bandwidth. This repo chose a fixed 20-byte message. The book does not set a byte count.
- A pixel becomes a spatial coordinate relative to the landing site.
- The dashboard shows data age or uncertainty.
- The lossy link is real sockets, not a comment.

Left open by the spec: model, math, packet layout, language, and UI. This build uses Python, NumPy, OpenCV, UDP, and one HTML/JS page.

Ambiguities, recorded rather than invented:

- The book sets no numeric bandwidth cap, loss rate, frame-rate gate, hazard list, or blackout length. 20 bytes, 30% loss, and the blackout windows are design choices.
- "2D pixel to 3D" is implemented as a 3D ray intersected with the flat floor z = 0. The stored fix is x, y. Object height is not estimated.
- "Video ingestion" in the integration rubric is local simulated frames. The radio does not carry images.
- The book does not require three operating-system processes. Onboard, radio, and ground are threads in one process.

## What a run does

1. The scout drives from the lander along +Y, 0.05 m per step. The live run is 130 frames at 0.2 s (0.25 m/s, about 6.5 m).
2. A mast camera renders 320×240. The human sees a color JPEG on the dashboard. That JPEG is local only. The badge says "STAYS ON ROVER".
3. Perception proposes dark blobs (value under 60) and bright blobs (value over 175) with area at least 28 px. A trained network then labels each blob as none, rock, pit, or lens artifact. It does not scan the whole image.
4. A tracker confirms a hazard only after the scout has traveled at least 0.7 m and the ground points in that window stay within 0.32 m. Otherwise the outcome is explicit:
   - `rejected`, reason `inconsistent_ground` — the points slide (the lens spot).
   - `expired`, reason `missed_detections` — the track went quiet before it could be judged.
   A rejection is archived. It is not later reported as zero. An expiration does not count as a rejection.
5. Confirmed rocks and pits, plus a low-priority pose at most every 0.5 s, are packed into 20-byte datagrams. The link drops about 30% of packets, closes for a blackout, and allows 80 bytes per burst. The scout keeps the same sequence and retries until a 2-byte ack. A same-sequence retry with a newer timestamp can tighten the ground fix. An older packet cannot roll it back.
6. The ground map draws a marker only when a valid packet arrives. During silence it may coast the scout for at most 1.2 s, then it holds the last pose and marks it predicted. Hazard sigma grows by 0.05 m per second of silence. Age is seconds since the packet timestamp.

Live blackout is 4 s to 12 s. After the drive the scout stays parked and the radio closes for 4 s of every 14 s. Dashboard: http://127.0.0.1:8877 (`python run_mission.py`). The live report is `veil/artifacts/live/report.json`. It does not overwrite the acceptance files.

## Architecture

| Role | File | What it may know |
| --- | --- | --- |
| Onboard computer | `veil/onboard.py` | Camera, tracker, send queue. Not the ground map. |
| Radio | `veil/link.py` | UDP on 127.0.0.1, two sockets, seeded loss, blackout, 80-byte budget, acks. |
| Ground map | `veil/ground.py` | Decoded packets only. Not images. |
| Dashboard | `veil/web/index.html` | Ground state and a local camera preview. |

`veil/mission.py` `run()` is the one pipeline. Tests and the live command both call it. `run` closes sockets and shuts the HTTP server down. The server does not reuse the address, so a second live process fails instead of hiding behind the first.

## Camera

`veil/camera.py`. Mast height 0.55 m, pitch 18° down, fx = fy = 300, principal point (160, 120). Yaw 0 looks along +Y. Right is +X. Up on the map is +Y. Image v grows downward.

`pixel_to_ground` intersects the pixel ray with z = 0. Non-finite inputs, rays that do not point down, and points behind the camera return no hit. The image center meets the ground about 1.7 m in front of the scout. The pale map wedge is `view_polygon()`, the same rays as the picture, yaw locked to 0.

A grid of ground points in front of the scout projects to pixels and back within 2 cm (`tests/test_geometry.py`).

## Known world

Flat ground. Lander origin. Four hazards in `veil/sim.py`:

| Name | Class | x (m) | y (m) | radius (m) |
| --- | --- | --- | --- | --- |
| rock-a | rock | -1.0 | 4.2 | 0.28 |
| pit-a | pit | 0.35 | 5.0 | 0.42 |
| rock-b | rock | 0.95 | 6.1 | 0.24 |
| pit-b | pit | -0.55 | 7.3 | 0.38 |

## Perception

Training: `python -m veil.train`. The simulator renders labeled 16×16 patches (262 features: pixels, mean, spread, dark fraction, bright fraction, and normalized pixel position). Frames invent new rocks and pits, shift brightness, and add a little yaw. Some frames use the flight path. Extra frames paint a pit or rock under the lens and still label that pixel as the lens.

- 1,272 training patches, 318 each of none, rock, pit, and artifact. File: `veil/outputs/train_set.npz`.
- Network: NumPy MLP, 262 inputs, 64 ReLU units, 4 softmax outputs. Weights: `veil/outputs/hazard_net.npz`.
- Held-out set is `generate(80, seed=1007)`, not the training seed 7. Accuracy 0.9 on 448 synthetic patches. This is not lunar field performance.
- Confusion matrix, rows and columns ordered none, rock, pit, artifact:

```text
[[97, 10, 3, 2],
 [9, 99, 2, 2],
 [1, 3, 106, 2],
 [0, 0, 11, 101]]
```

Eleven held-out lens patches were called pits. Confirmation still requires geometric agreement, so a pit label on the sliding spot does not place a map hazard.

Inference loads the npz. If the file is missing or incomplete it raises. It does not train. `tests/test_model.py` checks the feature width, empty frames, and a broken weights file.

## Packet

`veil/packet.py`. Little-endian, magic `0xC7`, always 20 bytes. There is no version field. The magic is the layout id.

| Offset | Field |
| --- | --- |
| 0 | magic u8 |
| 1 | type u8: 1 hazard, 2 pose |
| 2 | sequence u16 |
| 4 | time ms u32 |
| 8 | track id u16 |
| 10 | class u8: 1 rock, 2 pit, 0 on a pose |
| 11 | flags u8 |
| 12 | agreement u8, 0–100 |
| 13 | x centimeters i16 |
| 15 | y centimeters i16 |
| 17 | sigma centimeters u8, 0–2.55 m |
| 18 | radius centimeters u8, 0–2.55 m |
| 19 | XOR of bytes 0–18 |

`encode` raises if a value does not fit. It does not clamp or wrap. `decode` returns nothing for a short buffer, a bad checksum, an unknown type, or a hazard class that is not rock or pit. Example packet for a pit at (−1.25 m, 5.00 m):

```text
c7010400dc050000020002005083fff4010c28e6
```

Full contract: `veil/docs/TELEMETRY_PROTOCOL.md`.

Delivery is at-least-once with receiver collapse. It is not exactly-once. A lost ack causes another copy. Sequence numbers are 16-bit and wrap is not defined. Bytes in the report are application payload only, 20 bytes per send attempt, including drops. UDP and IP headers are not counted.

## Ground and dashboard

Older poses and older hazard fixes are ignored. Duplicate sequence at the same timestamp does not create a second marker. The map grid does not pan or zoom. The lander and the scout can be dragged. Double-click resumes the scripted drive. Pits are one orange ring. Rocks are one teal diamond. Hazard rows show side of path, distance from the lander, sigma, and age in seconds.

## Checks

Command, from the `veil` directory:

```bash
python -m pytest tests -q
python -m veil.accept
```

Latest run: **27 passed**. `python -m veil.accept` writes `veil/artifacts/acceptance/SUMMARY.txt` from the JSON. Exit status is pytest's. A failed assertion is not rewritten into a pass.

Both scenarios use seed 26, step 0.05 m, loss 0.3, budget 80 bytes.

| | Short | Long (same timing as the live drive, without waiting) |
| --- | --- | --- |
| Frames, dt, blackout | 140, 0.1 s, 0.5–6 s | 130, 0.2 s, 4–12 s |
| Hazards | 4/4, 0 false, 0 missed | 4/4, 0 false, 0 missed |
| Tracks | 1 rejected, 0 expired | 1 rejected, 0 expired |
| Mean / max error | 0.033 m / 0.094 m | 0.039 m / 0.085 m |
| Payload vs naive JSON | 860 vs 24373 | 1340 vs 23798 |
| Sent / delivered / dropped | 43 / 28 / 15 | 67 / 43 / 24 |
| Detect-only rate | 775 frames/s | 762 frames/s |
| `passed` | true | true |

Error method: each delivered hazard is matched once to the nearest same-class truth within 0.6 m. The frame rate is mission frames divided by time inside `detect` only. Radio sleep is excluded.

Unit tests also cover geometry, packet limits, track outcomes, ground ordering, seeded loss near 30%, blackout, delivery plus ack, the burst budget, and model loading.

An earlier live file, `veil/outputs/preselection_report.json`, is obsolete. It was overwritten by an old dashboard and reported zero withheld tracks. Do not use it. Current results are the two acceptance JSON files.

## Docs in the repo

- `veil/README.md` — setup, train, test, live demo
- `veil/docs/ENGINEERING_AUDIT.md`
- `veil/docs/ARCHITECTURE.md`
- `veil/docs/TELEMETRY_PROTOCOL.md`
- `veil/docs/TESTING_AND_ACCEPTANCE.md`
- `veil/docs/LIMITATIONS.md`
- `veil/docs/REQUIREMENTS_TRACEABILITY.md`

## Still open

- Pitch slides and the 7-minute talk were intentionally not built.
- The network is not a full-image detector. Proposals are still brightness thresholds.
- Held-out accuracy is synthetic. Eleven lens patches in that set were labeled pits. Geometry, not the class name, keeps them off the map.
- Localization assumes a flat floor and a known mast.
- There is no packet version byte.
- The dashboard's drag and resume path was not fully re-clicked in a browser during the hardening pass. The page text that the picture is not a downlink is asserted by a test.
- `git_revision` in the JSON is null when git cannot name HEAD.

Readiness of the software pipeline: conditionally ready for a technical preselection review. The acceptance suite passes. The pitch is separate, and the limits above should be stated rather than hidden.
