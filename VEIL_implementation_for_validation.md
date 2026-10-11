# VEIL — implementation brief for review

Upload this file together with the ATEC'26 Specification Book. The code is in the `veil` folder next to this file. This brief describes the system as of 11 October 2026, after the systems-integration pass and after the PolarDB network was wired into a photo demo. It is a status report, not the specification.

Ask the reviewer to compare this implementation with the specification for project C.R.A.T.E.R. (Constrained Reconnaissance, Autonomy & Telemetry in Extreme Regions): what is required, what is flexible, what is ambiguous, and what is still missing before a preselection submission. Do not treat this file as proof by itself. The synthetic mission numbers below come from `veil/artifacts/acceptance/` and were not re-run after the photo-demo wiring. The PolarDB number comes from `veil/outputs/polar_train_report.json`, now in this working tree together with `veil/outputs/hazard_net_polar.npz` and `veil/veil/polar.py`. Those files were taken from branch `cursor/polar-hazard-train-9dbc` (draft pull request https://github.com/AHmedHsini/atec/pull/1). The acceptance mission still loads the synthetic network.

## Where we are

Two runs exist. They do not share weights.

| Run | Command | Weights | Pictures | What it is for |
| --- | --- | --- | --- | --- |
| Painted acceptance mission | `python run_mission.py` from `veil/`, http://127.0.0.1:8877 | `veil/outputs/hazard_net.npz` | Simulator, 320×240 | The scored pipeline: 4 known hazards, loss, blackout, 20-byte packets |
| PolarDB photo demo | `python run_mission.py --polar --port 8878 --loss 0` | `veil/outputs/hazard_net_polar.npz` | Real NASA PolarDB frames, 1936×1216 | Show the trained network on held-out terrain photos |

The photo demo was started on this machine and the dashboard at http://127.0.0.1:8878 reported PolarDB held-out accuracy **0.581** and **8** placed hazards. Those 8 are labels on one lab photo, not the four painted hazards, and they are not an acceptance pass.

`python -m veil.polar_demo` fetches a few frames from the Terrain 9 zip by HTTP range. It does not download the 11.7 GB archive set. `dataset/polar/` is gitignored. The live sequence is the no-sun left camera at position A, exposure 0512, written four times so a stationary rig can agree with itself. The sun-at-180° and sun-at-270° frames from the same position are kept beside it in `dataset/polar/demo/lighting/`. They are not in the live sequence: the same rocks change class when the illumination changes, which is what the 0.581 score is warning about.

## Product

VEIL is a software-emulated lunar scout. The camera picture stays on the rover. OpenCV proposes blobs. A small local NumPy network names them. Distance is not a network output: each blob center is a ray from a known mast, intersected with the flat floor. A hazard is radioed only after two views agree on the same ground point. Mission Control plots only what survived a lossy link, and it shows how old and how uncertain each fix is.

The black dot on the left of the camera picture is a smudge stuck on the lens, painted at pixel (36, 110) with radius 7. It is not on the ground. It stays on the same pixel while the scout drives, so its guessed ground point slides. The tracker rejects that track. It is not drawn on the map.

## What the spec fixes, and what this repo chose

Fixed by the spec:

- 100% software-emulated. No rover hardware.
- Data flow: Perception, then Telemetry, then Dashboard.
- Local perception. No cloud inference API. A cloud job trained the PolarDB weight file. Both demos load a local npz. Neither calls a cloud model at runtime.
- Severe bandwidth. This repo chose a fixed 20-byte message. The book does not set a byte count.
- A pixel becomes a spatial coordinate relative to the landing site.
- The dashboard shows data age or uncertainty.
- The lossy link is real sockets, not a comment.

Left open by the spec: model, math, packet layout, language, UI, and whether onboard, radio, and ground are separate programs. This build uses Python, NumPy, OpenCV, UDP, and one HTML/JS page. The live command runs three operating-system processes. Tests can stay in one process.

Ambiguities, recorded rather than invented:

- The book sets no numeric bandwidth cap, loss rate, frame-rate gate, hazard list, or blackout length. 20 bytes, 30% loss, and the blackout windows are design choices.
- "2D pixel to 3D" is implemented as a 3D ray intersected with the flat floor z = 0. The stored fix is x, y, plus the flat distance from the scout. Object height is not estimated. A rock above the plane or a pit below it moves the hit.
- "Video ingestion" means the pipeline reads frames. Simulated frames satisfy it. The same pipeline can also read a PNG directory or an mp4 plus a pose CSV. The radio does not carry images.
- The book does not require NASA imagery, a field camera, or three processes. The PolarDB photo demo and the process split are extra. The acceptance gate is still the painted world.

## What a run does

1. The scout drives from the lander along +Y, 0.05 m per step. The live run is 130 frames at 0.2 s (0.25 m/s, about 6.5 m).
2. A mast camera renders 320×240. The human sees a color JPEG on the dashboard. That JPEG is local only. The badge says "STAYS ON ROVER".
3. Perception proposes dark blobs (value under 60) and bright blobs (value over 175) with area at least 28 px. A trained network then labels each blob as none, rock, pit, or lens artifact. It does not scan the whole image.
4. The blob center is converted to landing-site x, y and a flat range by `pixel_to_ground`. Pixel radius becomes meters with `distance * radius_px / 300`.
5. A tracker confirms a hazard only after the scout has traveled at least 0.7 m and the ground points in that window stay within 0.32 m. Otherwise the outcome is explicit:
   - `rejected`, reason `inconsistent_ground` — the points slide (the lens spot).
   - `expired`, reason `missed_detections` — the track went quiet before it could be judged.
   A rejection is archived. It is not later reported as zero. An expiration does not count as a rejection.
6. Confirmed rocks and pits, a pose at most every 0.5 s, and a heartbeat every 1.0 s are packed into 20-byte datagrams. The link drops about 30% of packets, closes for a blackout, and allows 80 bytes per burst. While the scout knows the link is down it holds the queue and does not spend bytes. It retries until a 2-byte ack. A same-sequence retry with a newer timestamp can tighten the ground fix. An older packet cannot roll it back. Sequence numbers use RFC 1982 serial arithmetic, so sequence 0 is newer than 65535.
7. If perception stalls for more than 1.0 s, the onboard mode becomes SAFE. Heartbeats continue. New hazards are not queued until a fresh observation returns the mode to RUN.
8. The ground map draws a marker only when a valid packet arrives. Heartbeats do not create hazards. Link state is UP if the last heartbeat is under 1.0 s old, DEGRADED under 3.0 s, otherwise DOWN. During silence the map may coast the scout for at most 1.2 s, then it holds the last pose and marks it predicted. Hazard sigma grows by 0.05 m per second of silence. Age is seconds since the packet timestamp.

The send queue holds at most 8 packets and drops the lowest value. A heartbeat is worth 0.001, a pose 0.01, a hazard more. A dropped hazard is counted once and is not put back. A hazard also leaves the queue after 12 real send attempts or after it has waited 30 s (8 s in the stress case). Pose and heartbeat drops are not counted as lost hazards.

Live blackout on the painted mission is 4 s to 12 s. After the drive, when the radio cycle is on, the link closes for 4 s of every 14 s. Dashboard: http://127.0.0.1:8877 (`python run_mission.py`). That command starts onboard, radio, and ground as three processes. `--in-process` keeps the fast single-process path used by most tests. The live report is `veil/artifacts/live/report.json`. It does not overwrite the acceptance files.

The photo demo sets `VEIL_MODEL=polar` and reads `dataset/polar/demo/`. The scout does not drive. The same photo is shown four times at one pose. Confirmation then uses a 0 m travel gate and three repeated views, because the lab camera is fixed. The painted mission still requires 0.7 m of travel. Loss on the photo command above is 0 so the first look at the dataset is not mixed with the 30% drop test.

## Architecture

| Role | File | What it may know |
| --- | --- | --- |
| Onboard computer | `veil/onboard.py` | Camera, tracker, send queue, watchdog. Not the ground map. |
| Radio | `veil/link.py` | UDP on 127.0.0.1, seeded loss, blackout, 80-byte budget, acks. |
| Ground map | `veil/ground.py` | Decoded packets only. Not images. |
| Process split | `veil/procs.py` | Live parent plus onboard, radio, and ground children. |
| Frame source | `veil/source.py` | Simulator, PNG directory plus `poses.csv`, or mp4 plus `{stem}_poses.csv`. |
| Dashboard | `veil/web/index.html` | Ground state and a local camera preview. |

`veil/mission.py` `run()` is the in-process pipeline. `run(..., processes=True)` delegates to `veil/procs.py`. Tests and acceptance use the in-process path unless the scenario name says processes. `run` closes sockets and shuts the HTTP server down. The server does not reuse the address, so a second live process fails instead of hiding behind the first.

## Camera and distance

`veil/camera.py`. Mast height 0.55 m, pitch 18° down, fx = fy = 300, principal point (160, 120). Yaw 0 looks along +Y. Right is +X. Up on the map is +Y. Image v grows downward.

`pixel_to_ground` intersects the pixel ray with z = 0 and returns x, y, and the flat distance from the scout. Non-finite inputs, rays that do not point down, and points behind the camera return no hit. The image center meets the ground about 1.7 m in front of the scout. The pale map wedge is `view_polygon()`, the same rays as the picture, yaw locked to 0.

This is a range measurement under a flat-floor assumption. It is not stereo, and the network does not predict distance. PolarDB's own cameras are a 30 cm stereo pair with LiDAR. That depth was used to label training patches. It is not the ranging method in either demo.

On the photo demo the image stays 1936×1216. `dataset/polar/demo/camera.json` uses the rectified left-camera focal length from Terrain 9 (fx = fy ≈ 2073.6, principal point near (966, 600)) with an assumed mast height of 1.1 m and a 32° downward pitch. Those two mast numbers are not a surveyed pose of the NASA rig. The pixels and the class labels are from the dataset. The meters on the map are this flat-floor ray.

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

The flight trainer is `python -m veil.train`. The simulator renders labeled 16×16 patches (262 features: pixels, mean, spread, dark fraction, bright fraction, and normalized pixel position). Frames invent new rocks and pits, shift brightness, and add a little yaw. Some frames use the flight path. Extra frames paint a pit or rock under the lens and still label that pixel as the lens.

- 1,272 training patches, 318 each of none, rock, pit, and artifact. File: `veil/outputs/train_set.npz`.
- Network: NumPy MLP, 262 inputs, 64 ReLU units, 4 softmax outputs. Not a convolutional detector and not YOLO. Dependencies stay `numpy` and `opencv-python`.
- Flight weights: `veil/outputs/hazard_net.npz`.
- Held-out set is `generate(80, seed=1007)`, not the training seed 7. Accuracy 0.9 on 448 synthetic patches. This is not lunar field performance and it is not the PolarDB score.
- Confusion matrix, rows and columns ordered none, rock, pit, artifact:

```text
[[97, 10, 3, 2],
 [9, 99, 2, 2],
 [1, 3, 106, 2],
 [0, 0, 11, 101]]
```

Eleven held-out lens patches were called pits. Confirmation still requires geometric agreement, so a pit label on the sliding spot does not place a map hazard.

Inference loads the npz. If the file is missing or incomplete it raises. It does not train. `tests/test_model.py` checks the feature width, empty frames, and a broken weights file.

### PolarDB network, now used by the photo demo

A cloud job downloaded the NASA Ames POLAR Stereo set (PolarDB) from https://ti.arc.nasa.gov/dataset/IRG_PolarDB/. The archives are about 11.7 GB: terrains 1–13, radiometric calibration, organized LiDAR range maps, and the readme. They are not in git. This checkout stores the weights, the trainer (`veil/veil/polar.py`), and the report. It stores three tone-mapped demo frames, not the archives.

The images have no rock or pit labels. Labels come from LiDAR height on the rectified range maps: a local high residual is a rock, a low residual or a crater bowl is a pit, and smooth ground is none. PolarDB has no lens-smudge class, so artifact examples are synthetic dark disks. Each sample is a 64×64 window resized to 16×16, then the same six scalars as the flight model, with pixel position divided by 1936 and 1216. That is 262 inputs, matching the flight network's width, on a different crop.

The cloud run wrote `veil/outputs/hazard_net_polar.npz` and `veil/outputs/polar_train_report.json`.

- 4,800 training patches and 960 validation patches, balanced to 1,200 and 240 per class.
- 266 train frames, 166 validation frames. Terrains 9 and 12 are fully held out. Terrain 13 was downloaded and was not in the train or held-out lists in the report.
- Held-out accuracy **0.581**. Confusion, same class order:

```text
[[126, 56, 42, 16],
 [51, 111, 56, 22],
 [63, 57, 110, 10],
 [14, 11, 4, 211]]
```

The artifact row is high because those patches are painted disks. Rock, pit, and none are near chance-to-moderate. This number must not be quoted as the synthetic 0.9.

`python run_mission.py --polar` calls `arm_polar()` and `perceive.use_polar()`. Proposals on those frames are local contrast (darker than a blur by 14, or brighter by 18), area at least 160 px, confidence at least 0.55, artifact labels dropped, and at most 8 kept. The painted mission does not take this path. It still loads `hazard_net.npz`, because the acceptance rule is 4 delivered hazards and 0 false hazards on the painted world. Swapping the 0.581 weights into that world would change the gate that already passes.

## Packet

`veil/packet.py`. Little-endian, magic `0xC7`, always 20 bytes. Format string `<BBHIHBBBhhBBB`.

| Offset | Field |
| --- | --- |
| 0 | magic u8 |
| 1 | `(version << 6) \| type`. Version is 1. Type 1 hazard, 2 pose, 3 heartbeat. |
| 2 | sequence u16 |
| 4 | time ms u32 |
| 8 | track id u16. On a heartbeat this is the confirmed-hazard count. |
| 10 | class u8: 1 rock, 2 pit, 0 on a pose or heartbeat |
| 11 | flags u8. On a heartbeat this is the mode: 0 RUN, 1 SAFE. |
| 12 | agreement u8, 0–100. On a heartbeat this is the queue depth, capped at 100. |
| 13 | x centimeters i16 |
| 15 | y centimeters i16 |
| 17 | sigma centimeters u8, 0–2.55 m |
| 18 | radius centimeters u8, 0–2.55 m |
| 19 | CRC-8 over bytes 0–18, polynomial 0x07, init 0, no reflection, xorout 0 |

`encode` raises if a value does not fit. It does not clamp or wrap. `decode` returns nothing for a short buffer, a bad CRC, a version other than 1, an unknown type, or a hazard class that is not rock or pit.

A 16-byte layout was written down in `veil/docs/TELEMETRY_PROTOCOL.md` and was not implemented. It would drop agreement and would wrap the time field every 65.536 s. The flight packet stays 20 bytes.

Delivery is at-least-once with receiver collapse. It is not exactly-once. A lost ack causes another copy. Bytes in the report are application payload only, 20 bytes per send attempt, including drops. UDP and IP headers are not counted. The first attempt of a hazard, pose, or heartbeat is counted in that bucket. Later attempts are `retransmission_bytes`. Sends while the scout already knows the link is closed would be `wasted_blackout_bytes`. On the acceptance runs that counter is 0, because the scout holds.

Full contract: `veil/docs/TELEMETRY_PROTOCOL.md`. A recorded log is `veil/artifacts/acceptance/packet_log_seed26.hex`.

## Ground and dashboard

Older poses and older hazard fixes are ignored. The same sequence with a newer timestamp refreshes the fix and keeps the original receive time. Duplicate sequence at the same timestamp does not create a second marker. The map grid does not pan or zoom. The lander and the scout can be dragged. Double-click resumes the scripted drive. Pits are one orange ring. Rocks are one teal diamond. Hazard rows show side of path, distance from the lander, sigma, and age in seconds.

## Checks

Command, from the `veil` directory:

```bash
python -m pytest tests -q
python -m veil.accept
```

Latest run on this checkout: **43 passed**. `python -m veil.accept` then writes `veil/artifacts/acceptance/SUMMARY.txt` and exits 0 only if the sweep gate passes. A failed assertion is not rewritten into a pass.

Both in-process scenarios use seed 26, step 0.05 m, loss 0.3, budget 80 bytes.

| | Short | Long (same timing as the live drive, without waiting) |
| --- | --- | --- |
| Frames, dt, blackout | 140, 0.1 s, 0.5–6 s | 130, 0.2 s, 4–12 s |
| Hazards | 4/4, 0 false, 0 missed | 4/4, 0 false, 0 missed |
| Tracks | 1 rejected, 0 expired | 1 rejected, 0 expired |
| Mean / max error | 0.033 m / 0.094 m | 0.038 m / 0.085 m |
| Payload vs naive JSON | 900 vs 24373 | 1600 vs 23798 |
| Payload vs every detection as float32 | 1840 vs 9220 | 3060 vs 9020 |
| Confirmed hazards as float32 | 80 | 80 |
| Hazard / pose / heartbeat / retries | 80 / 520 / 260 / 40 | 80 / 880 / 520 / 120 |
| Bytes per delivered hazard | 20.0 | 20.0 |
| Wasted during blackout | 0 | 0 |
| Sent / delivered / dropped | 92 / 59 / 33 | 153 / 100 / 53 |
| Detect-only rate | 218 frames/s | 169.9 frames/s |
| `passed` | true | true |

The long byte split adds up: 80 + 880 + 520 + 120 = 1600. Pose and heartbeat are sent once and then replaced, so retries fell from 1580 bytes to 120. Detect rate fell because each blob now carries a small uncertainty sample. It is still above 15 frames/s. The large saving versus every detection is onboard confirmation. Packing does not shrink one confirmed hazard below 20 bytes. Pose, heartbeat, and retries make the radio total larger than the 80-byte confirmed-only baseline.

Error method: each delivered hazard is matched once to the nearest same-class truth within 0.6 m. The frame rate is mission frames divided by time inside `detect` only. Radio sleep is excluded. The frame rate moves between machines. The gate is above 15 frames/s.

Loss sweep, no blackout, 10 seeds. The pass gate is: latency changes with loss, and at 30% the minimum delivered hazards is at least 4. Rows at 50% and above are informational.

| Loss | mean / min delivered | receipt median / p95 (s) | retries per hazard | informational |
| --- | --- | --- | --- | --- |
| 0.1 | 4.0 / 4 | 0.0 / 0.1 | 0.175 | no |
| 0.3 | 4.0 / 4 | 0.0 / 0.2 | 1.325 | no |
| 0.5 | 4.0 / 4 | 0.0 / 0.305 | 3.075 | yes |
| 0.7 | 4.0 / 4 | 0.15 / 0.605 | 6.25 | yes |
| 0.9 | 2.9 / 2 | 0.3 / 0.96 | 9.825 | yes |

Medians at 10% and 30% are 0.0 s because a first try that lands is delivered in the same simulated step it was confirmed. The p95 values, and the medians from 50% up, are what move.

Recovery at 30% loss, blackout 0.5–6 s: first queued hazard median 0.0 s, all queued median 0.1 s, minimum delivered 4. Stress blackout that never reopens, pending limit 8 s, seed 1: 4 hazards lost, 0 delivered, 0 wasted bytes.

Other acceptance files:

- `source_compare.json`: PNG playback matches the simulator pixels and also delivers 4/4 with mean error 0.033 m. An mp4 from the `mp4v` codec delivers 4/4 with mean error 0.035 m. At 30% loss the packet log is not byte-identical to the simulator, because ack timing changes retries.
- `acceptance-processes.json`: three processes, 140 frames, dt 0.1 s, blackout 0.5–6 s, loss 0.3, seed 26. 4/4, 0 false, 1 rejected, passed.
- `process_determinism.json`: two process runs at 0% loss, 40 frames, seed 4, identical logs of 28 packets after a 20 ms ack drain. That identity is not claimed at 30% loss.

Unit tests also cover geometry, packet limits, CRC and version, track outcomes, ground ordering, heartbeats and link state, the watchdog, queue trim, seeded loss near 30%, blackout, delivery plus ack, the burst budget, hazard expiry, and model loading.

An earlier live file, `veil/outputs/preselection_report.json`, is obsolete. It was overwritten by an old dashboard and reported zero withheld tracks. Do not use it.

## Docs in the repo

Repository: https://github.com/AHmedHsini/atec (private).

- `veil/README.md` — setup, train, test, live demo
- `veil/docs/ENGINEERING_AUDIT.md`
- `veil/docs/ARCHITECTURE.md`
- `veil/docs/TELEMETRY_PROTOCOL.md`
- `veil/docs/TESTING_AND_ACCEPTANCE.md`
- `veil/docs/LIMITATIONS.md`
- `veil/docs/REQUIREMENTS_TRACEABILITY.md`

## Still open

- Pitch slides and the 7-minute talk were intentionally not built.
- Neither network is a full-image detector. The painted mission still proposes by brightness. The photo demo proposes by local contrast, then classifies those blobs.
- The flight held-out accuracy of 0.9 is synthetic. Eleven lens patches in that set were labeled pits. Geometry, not the class name, keeps them off the map.
- PolarDB held-out accuracy is 0.581. The photo demo uses those weights on Terrain 9, which was held out. The class flips when the sun angle changes, so the live sequence is one lighting, repeated. The acceptance mission does not load this file.
- The photo demo's meters use an assumed mast height and pitch on top of Terrain 9's focal length. They are not LiDAR ranges and not a surveyed rig pose.
- Localization on both runs assumes a flat floor. One camera does not measure object height.
- The full acceptance suite was not re-run after the photo-demo wiring. The tables above are the last synthetic run. Model, track, and geometry unit tests were run after the wiring and passed.
- The dashboard's drag and resume path was not fully re-clicked in a browser during the hardening pass. The page text that the picture is not a downlink is asserted by a test.
- `git_revision` in older acceptance JSON is null when git cannot name HEAD. This folder is now a git checkout.

Readiness of the software pipeline: conditionally ready for a technical preselection review on the painted mission. The pitch is separate. The PolarDB demo is the real-photo path, with a lower score that has to stay labeled 0.581, and the limits above should be stated rather than hidden.
