# Engineering audit

Baseline inspected on 10 October 2026. The specification book is the authority. `VEIL_implementation_for_validation.md` described the code and was not treated as proof. `python -m pytest tests/test_mission.py` passed 4 tests before these changes. The live file `outputs/preselection_report.json` had been overwritten by a realtime run and said `passed: false` with `withheld_inconsistent_tracks: 0`.

## What already worked

- Local 320×240 camera, pinhole projection, ground-plane intersection.
- Brightness proposals classified by a NumPy network loaded from `outputs/hazard_net.npz`.
- 20-byte packets, XOR checksum, UDP loopback, 30% loss, blackout, 80-byte burst, acks.
- Dashboard map with age, sigma, and a local camera preview that is not on the radio.
- Short automated run: 4/4 hazards, 0 false, mean error about 3 cm.

## Must-fix findings

1. Inconsistent tracks were deleted after 6 missed frames and counted with the same `rejected` counter as quiet tracks. The report then looked only at tracks still alive, so a long run could show zero withheld tracks after the lens label flickered. Resolution: outcomes are now `tentative`, `confirmed`, `rejected` (`inconsistent_ground`), and `expired` (`missed_detections`). A rejection stays in the archive.
2. `decode` trusted the checksum but would accept an unknown hazard class and the ground map treated every non-pose type as a hazard. Resolution: unknown type and unknown hazard class return `None`.
3. Sigma and radius were silently clamped into a byte. Coordinates could throw from `struct.pack` on overflow. Resolution: `encode` raises `ValueError` outside the field width.
4. An older packet or a duplicate sequence could replace or freeze a newer fix. A retried hazard keeps its sequence and refreshes the payload, so a same-sequence update with a later timestamp must still be applied. Resolution: ground ignores older timestamps, ignores an identical sequence at the same time, and applies a same-sequence packet only when its timestamp is newer.
5. The live dashboard wrote `outputs/preselection_report.json`, the same path a reader could mistake for the acceptance result. Resolution: `artifacts/acceptance/` and `artifacts/live/` are separate. Writes are atomic.
6. `load()` trained a new network if the file was missing, so inference could overwrite weights. Resolution: a missing or incomplete file raises.

## Should-fix, done in this pass

- Round-trip geometry was one point. There is now a grid of points in front of the scout, plus behind-camera, non-finite, and sky-pixel cases.
- Link loss, blackout, delivery, ack, and the 80-byte budget had no direct tests.
- Held-out confusion was not stored. `write_held_out_metrics` scores the saved weights. It does not retrain them.
- `bytes_sent` counts application payload only. The report says so.
- Perception frames per second exclude radio sleep. The report says so.

## Left as documented limits

- The network classifies threshold proposals. It is not a full-image detector.
- Training and validation images are synthetic. 90% is not lunar field accuracy.
- Onboard, radio, and ground are threads in one process. The spec does not require separate operating-system processes.
- The spec does not set a packet size, a loss percentage, a frame rate, a hazard list, or a blackout length. Those numbers are design choices.
- The spec's "2D pixel to 3D" line is implemented as a ray against the plane z = 0, so the stored coordinates are x and y. Height is not estimated.
- No packet version field. Magic `0xC7` is the layout id. Adding a version would consume a byte that already has a meaning.
- Git revision is recorded when `git` can name `HEAD`. It is null when that command is unavailable.

## Not in scope

Pitch slides, a speaking script, and a presentation site.
