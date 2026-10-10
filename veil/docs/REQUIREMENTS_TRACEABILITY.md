# Requirements traceability

The specification book does not assign numeric targets for bandwidth, frame rate, packet size, or hazard classes. Where this table says PASS, a test in `veil/tests` or an acceptance JSON asserts the behavior. Pitch and Q&A are out of scope and are not graded here.

| Requirement | Where | Test or evidence | Status |
| --- | --- | --- | --- |
| 100% software-emulated | `veil/sim.py`, `veil/mission.py` | Acceptance runs complete without hardware | PASS |
| Flow Perception → Telemetry → Dashboard | `veil/mission.py` | `run()` used by tests and the live command | PASS |
| Severe bandwidth, compact telemetry | `veil/packet.py` 20 bytes, CRC-8 poly 0x07, version in the type byte | `tests/test_packet.py`, `artifacts/acceptance/acceptance-long.json` (`bytes_sent`, `naive_json_bytes`, `float32_bytes`), `artifacts/acceptance/loss_sweep.json` | PASS |
| Local perception, no cloud API | `veil/perceive.py`, `veil/model.py` | `test_model.py` loads a local npz and refuses a missing file | PASS |
| Detect hazards from a visual feed | Threshold proposals plus local network | Acceptance delivers 4/4 hazards | PASS |
| Real-time perception on modest compute | `detect` timer | Acceptance `fps` field, method documented, both runs above 15 | PASS |
| Track across views | `veil/tracks.py` | `test_tracks.py` confirms a stable pit and rejects a sliding one | PASS |
| 2D pixel to spatial coordinates vs landing site | `veil/camera.py`, plane z = 0 | `test_geometry.py` round trip under 2 cm on a grid. Height is not solved | PARTIAL |
| Team-implemented lossy sockets | `veil/link.py` UDP, `veil/sweep.py` | `tests/test_link.py`; `loss_sweep.json` for 10–90% with no blackout, plus a 30% recovery sweep | PASS |
| Dashboard shows spatial data and age or uncertainty | `veil/web/index.html`, `veil/ground.py` | `test_ground.py` age and sigma growth. Page text asserts the preview is not a downlink. Browser click-through of every control was not re-run in this pass | PARTIAL |
| Automated test outputs for preselection | `artifacts/acceptance/` | `python -m veil.accept` | PASS |
| Flight-software habits: fixed messages, checks, bounded bursts, retries, heartbeat, watchdog | packet, link, onboard, ground | `tests/test_systems.py` queue overflow, heartbeat link state, SAFE mode, identical packet log | PASS |
| Separate OS processes for onboard, radio, and ground | `veil/procs.py` | `acceptance-processes.json` and the child-lifetime check in `tests/test_systems.py`. The spec does not require this split. The live command uses it | PASS |
| Video file or image sequence as the camera | `veil/source.py`, `veil/record.py` | PNG sequence delivers 4/4 in `source_compare.json`. CSV columns are `frame,t_s,x_m,y_m,yaw_rad` | PASS |
| Live 7-minute pitch | Not in this engineering pass | No slides | NOT VERIFIED |

## Ambiguities recorded, not invented

- No numeric bandwidth cap, loss rate, blackout, or FPS gate appears in the book. 20 bytes, 30% loss, and the blackout windows are our choices.
- "Video ingestion" in the integration row is satisfied by reading simulated frames locally. The radio does not carry images.
- "2D pixel to 3D" is implemented as a 3D ray hit on a flat floor, stored as 2D map coordinates.
